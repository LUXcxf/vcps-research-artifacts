import sys
import copy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import torch
    from torch.utils.data import DataLoader
    import train_actor
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires the ML Python environment; uses CPU only")
class ActorResumeTest(unittest.TestCase):
    def loader(self, generator):
        return DataLoader(list(range(24)), batch_size=1, shuffle=True, generator=generator)

    def test_resume_reconstructs_second_epoch_order(self):
        generator = torch.Generator().manual_seed(42)
        loader = self.loader(generator)
        list(loader)
        epoch_start = generator.get_state().clone()
        uninterrupted = [int(batch.item()) for batch in loader]
        resumed_generator = torch.Generator().manual_seed(42)
        train_actor.restore_loader_epoch_state(
            resumed_generator, {"loader_epoch_start_state": epoch_start},
        )
        resumed = [int(batch.item()) for batch in self.loader(resumed_generator)]
        self.assertEqual(uninterrupted, resumed)
        self.assertEqual(uninterrupted[8:], resumed[8:])

    def test_checkpoint_records_epoch_start_and_training_identity(self):
        class Saver:
            def save_pretrained(self, path):
                pass

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        generator = torch.Generator().manual_seed(42)
        epoch_start = generator.get_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            train_actor.save_lora_checkpoint(
                Saver(), Saver(), checkpoint,
                {"epoch_index": 1, "next_batch_index": 8, "optimizer_step": 4},
                optimizer=optimizer, loader_epoch_start_state=epoch_start,
                training_identity={"data": "bound"},
            )
            state = torch.load(checkpoint / "training_state.pt", weights_only=False, map_location="cpu")
            self.assertTrue(torch.equal(state["loader_epoch_start_state"], epoch_start))
            self.assertEqual(state["training_identity"], {"data": "bound"})

    def test_incomplete_loader_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            train_actor.restore_loader_epoch_state(torch.Generator(), {})

    def test_resumed_updates_match_uninterrupted_with_dropout(self):
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout(0.2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(42)
        loader = self.loader(generator)

        def update(current, opt, batch):
            opt.zero_grad()
            loss = current(batch.float().reshape(1, 1) / 24).square().mean()
            loss.backward()
            opt.step()

        for batch in loader:
            update(model, optimizer, batch)
        epoch_start = generator.get_state().clone()
        for index, batch in enumerate(loader):
            update(model, optimizer, batch)
            if index == 7:
                saved_weights = copy.deepcopy(model.state_dict())
                saved_optimizer = copy.deepcopy(optimizer.state_dict())
                rng = torch.get_rng_state().clone()
        for batch in loader:
            update(model, optimizer, batch)
        expected = copy.deepcopy(model.state_dict())
        resumed = copy.deepcopy(model)
        resumed.load_state_dict(saved_weights)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_generator = torch.Generator().manual_seed(42)
        train_actor.restore_loader_epoch_state(resumed_generator, {"loader_epoch_start_state": epoch_start})
        torch.set_rng_state(rng)
        resumed_loader = self.loader(resumed_generator)
        for index, batch in enumerate(resumed_loader):
            if index >= 8:
                update(resumed, resumed_optimizer, batch)
        for batch in resumed_loader:
            update(resumed, resumed_optimizer, batch)
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
