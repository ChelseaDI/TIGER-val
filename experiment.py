"""SID collision experiment helpers (no model dependencies)."""
import ast
import math
from collections import Counter


def normalize_sid(text):
    return ''.join(text.split())


def build_catalog(indices, depth, token_type):
    """Validate mapping; the first D tokens define the shared SID class."""
    unique = token_type.endswith('_nc') or token_type == 'cid'
    expected = depth + int(token_type.endswith('_nc'))
    if not indices or depth < 1:
        raise ValueError('A nonempty mapping and positive D are required')
    catalog = {}
    counts = Counter()
    for item, tokens in indices.items():
        if not isinstance(tokens, list) or len(tokens) != expected:
            raise ValueError('Invalid SID length for item {}: expected {}'.format(item, expected))
        if not all(isinstance(t, str) and t and normalize_sid(t) == t for t in tokens):
            raise ValueError('SID tokens must be nonempty strings without whitespace')
        code, group = ''.join(tokens), ''.join(tokens[:depth])
        if unique and code in catalog:
            raise ValueError('The no-conflict mapping contains duplicate item codes')
        catalog[code] = group
        counts[group] += 1
    stats = {
        'items': len(indices), 'sid_classes': len(counts),
        'collision_classes': sum(n > 1 for n in counts.values()),
        'items_in_collision_classes': sum(n for n in counts.values() if n > 1),
        'collision_rate': 1 - len(counts) / len(indices),
        'max_class_size': max(counts.values()),
    }
    return catalog, counts, stats


def parse_metrics(metrics):
    names = ast.literal_eval(metrics) if isinstance(metrics, str) else metrics
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError('metrics must be a nonempty list')
    for name in names:
        kind, k = name.split('@')
        if kind not in ('hit', 'recall', 'ndcg') or int(k) < 1:
            raise ValueError('Unsupported metric: {}'.format(name))
    return names


def rank_candidates(predictions, scores, allowed):
    """Sort by beam score, remove invalid outputs and repeated candidates."""
    seen, ranked = set(), []
    for text, score in sorted(zip(predictions, scores), key=lambda p: p[1], reverse=True):
        code = normalize_sid(text)
        if math.isfinite(score) and code in allowed and code not in seen:
            seen.add(code)
            ranked.append(code)
    return ranked


def metric_values(rank, names):
    result = {}
    for name in names:
        kind, k = name.split('@')
        hit = rank is not None and rank <= int(k)
        result[name] = (1 / math.log2(rank + 1) if kind == 'ndcg' else 1.0) if hit else 0.0
    return result


class CollisionEvaluator:
    def __init__(self, indices, depth, token_type, metrics):
        self.catalog, self.counts, self.stats = build_catalog(indices, depth, token_type)
        self.item_level = token_type.endswith('_nc') or token_type == 'cid'
        self.metrics = parse_metrics(metrics)
        self.totals = {}
        self.samples = Counter()

    def add(self, ranked, target):
        target = normalize_sid(target)
        group = self.catalog[target]
        groups = [self.catalog[p] for p in ranked]
        # Same item-beam positions: isolates relaxing correctness, not retraining.
        sid_rank = next((i + 1 for i, g in enumerate(groups) if g == group), None)
        distinct = list(dict.fromkeys(groups))
        class_rank = distinct.index(group) + 1 if group in distinct else None
        ranks = {'sid_at_item_rank': sid_rank, 'sid_unique': class_rank} if self.item_level else {'sid': class_rank}
        if self.item_level:
            ranks['item'] = ranked.index(target) + 1 if target in ranked else None
        strata = ('all', 'collision' if self.counts[group] > 1 else 'singleton')
        for stratum in strata:
            self.samples[stratum] += 1
            totals = self.totals.setdefault(stratum, {})
            for level, rank in ranks.items():
                for name, value in metric_values(rank, self.metrics).items():
                    key = level + '/' + name
                    totals[key] = totals.get(key, 0.0) + value
            if self.item_level:
                for name in self.metrics:
                    kind, k = name.split('@')
                    if kind in ('hit', 'recall'):
                        gap = metric_values(sid_rank, [name])[name] - metric_values(ranks['item'], [name])[name]
                        key = 'sid_hit_item_miss/' + name
                        totals[key] = totals.get(key, 0.0) + gap

    def result(self):
        return {'catalog': self.stats, 'strata': {
            s: {'samples': self.samples[s], 'metrics': {
                k: v / self.samples[s] for k, v in self.totals.get(s, {}).items()
            }} for s in ('all', 'collision', 'singleton')
        }}
