import math
import unittest
from experiment import CollisionEvaluator, build_catalog, rank_candidates, parse_metrics

RAW = {'A': ['<a_1>', '<b_2>', '<c_3>'], 'B': ['<a_1>', '<b_2>', '<c_3>'],
       'C': ['<a_9>', '<b_8>', '<c_7>']}
UNIQUE = {k: v + ['<d_{}>'.format(int(k == 'B'))] for k, v in RAW.items()}


class ExperimentTests(unittest.TestCase):
    def test_same_sid_wrong_item(self):
        ev = CollisionEvaluator(UNIQUE, 3, 'sid_nc', ['hit@1', 'recall@1', 'ndcg@2'])
        a, b = ''.join(UNIQUE['A']), ''.join(UNIQUE['B'])
        ev.add([b, a], a)
        m = ev.result()['strata']['all']['metrics']
        self.assertEqual(m['item/hit@1'], 0)
        self.assertEqual(m['sid_at_item_rank/hit@1'], 1)
        self.assertEqual(m['sid_hit_item_miss/hit@1'], 1)
        self.assertEqual(m['sid_at_item_rank/ndcg@2'], 1)
        self.assertAlmostEqual(m['item/ndcg@2'], 1 / math.log2(3))

    def test_raw_and_strata(self):
        ev = CollisionEvaluator(RAW, 3, 'sid', ['hit@1'])
        a, c = ''.join(RAW['A']), ''.join(RAW['C'])
        ev.add([a], a)
        ev.add([a], c)
        r = ev.result()
        self.assertEqual(r['catalog']['collision_classes'], 1)
        self.assertEqual(r['catalog']['items_in_collision_classes'], 2)
        self.assertEqual(r['strata']['all']['metrics']['sid/hit@1'], .5)
        self.assertEqual(r['strata']['collision']['metrics']['sid/hit@1'], 1)
        self.assertNotIn('item/hit@1', r['strata']['all']['metrics'])

    def test_dedup_ranking_and_invalid(self):
        self.assertEqual(rank_candidates(['bad', ' a ', 'a', 'b', 'c'], [9, 3, 2, 1, float('-inf')], {'a', 'b', 'c'}), ['a', 'b'])

    def test_unique_class_rank_differs(self):
        ev = CollisionEvaluator(UNIQUE, 3, 'sid_nc', ['hit@2'])
        a, b, c = (''.join(UNIQUE[k]) for k in ('A', 'B', 'C'))
        ev.add([a, b, c], c)
        m = ev.result()['strata']['all']['metrics']
        self.assertEqual(m['sid_at_item_rank/hit@2'], 0)
        self.assertEqual(m['sid_unique/hit@2'], 1)

    def test_validation(self):
        with self.assertRaises(ValueError):
            build_catalog(RAW, 3, 'sid_nc')
        with self.assertRaises(ValueError):
            build_catalog({'A': UNIQUE['A'], 'B': UNIQUE['A']}, 3, 'sid_nc')
        with self.assertRaises(ValueError):
            parse_metrics("__import__('os').system('false')")

    def test_miss_and_empty_stratum(self):
        ev = CollisionEvaluator(RAW, 3, 'sid', ['hit@5', 'ndcg@5'])
        ev.add([], ''.join(RAW['C']))
        r = ev.result()['strata']
        self.assertEqual(r['all']['metrics']['sid/ndcg@5'], 0)
        self.assertEqual(r['collision'], {'samples': 0, 'metrics': {}})


if __name__ == '__main__':
    unittest.main()
