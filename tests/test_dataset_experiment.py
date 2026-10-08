import unittest
from types import SimpleNamespace
from unittest.mock import patch
from utils import RecDataset


class DatasetExperimentTests(unittest.TestCase):
    def test_shared_sid_in_history_and_target_and_unique_baseline(self):
        for kind in ('sid', 'sid_nc'):
            args = SimpleNamespace(dataset='synthetic', tokenizer_plm='sentence-t5-base',
                                   K=256, D=3, add_user_prefix=False, item_sep=',',
                                   max_len=20, token_type=kind)
            shared = ['<a_1>', '<b_2>', '<c_3>']
            mapping = {'A': shared[:], 'B': shared[:]}
            if kind == 'sid_nc':
                mapping = {k: v + ['<d_{}>'.format(i)] for i, (k, v) in enumerate(mapping.items())}

            def fake_load(ds):
                ds.inter_data = {'u': ['A', 'B', 'A', 'B', 'A']}
                ds.indices = mapping

            with patch.object(RecDataset, '_load_data', fake_load):
                train = RecDataset(args, 'train')
                valid = RecDataset(args, 'valid')
                test = RecDataset(args, 'test')
            self.assertEqual(len(train), 2)
            self.assertEqual(valid[0]['labels'], ''.join(mapping['B']))
            self.assertEqual(test[0]['labels'], ''.join(mapping['A']))
            self.assertEqual(train[0]['input_ids'], ''.join(mapping['A']))
            self.assertEqual(train[0]['labels'], ''.join(mapping['B']))
            self.assertEqual(len(test.get_all_items()), 1 if kind == 'sid' else 2)
