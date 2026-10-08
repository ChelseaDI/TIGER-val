"""Convert Amazon Beauty reviews and metadata to the repository's JSON format."""

import argparse
import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def read_records(path):
    """Old Amazon metadata uses Python literals; reviews use JSON lines."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError('expected a dictionary')
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f'{path}:{line_number}: invalid record') from exc
            yield record


def preprocess(reviews_path, metadata_path, output_dir, min_interactions=5):
    if min_interactions < 4:
        raise ValueError('min_interactions must be >= 4 for train/valid/test splitting')
    sequences = defaultdict(list)
    for record in read_records(reviews_path):
        sequences[str(record['reviewerID'])].append(
            (int(record['unixReviewTime']), str(record['asin']))
        )
    original_users = len(sequences)
    sequences = {user: seq for user, seq in sequences.items()
                 if len(seq) >= min_interactions}
    if not sequences:
        raise ValueError('No users remain after filtering')

    # Deterministic IDs; repeated interactions are retained, as in the source data.
    user2id = {user: i for i, user in enumerate(sorted(sequences))}
    asins = sorted({asin for seq in sequences.values() for _, asin in seq})
    item2id = {asin: i for i, asin in enumerate(asins)}
    inter_data = {}
    for user, user_id in user2id.items():
        # Stable sorting preserves source order for equal timestamps.
        sequence = sorted(sequences[user], key=lambda event: event[0])
        inter_data[str(user_id)] = [item2id[asin] for _, asin in sequence]

    metadata = {}
    for record in read_records(metadata_path):
        asin = str(record.get('asin', ''))
        if asin in item2id:
            metadata[asin] = record
    items = {}
    for asin, item_id in item2id.items():
        record = metadata.get(asin, {})
        # Preserve every field (including nested values) for later text selection.
        items[str(item_id)] = dict(record)
        items[str(item_id)].update(asin=asin)
        items[str(item_id)].setdefault('title', '')
        items[str(item_id)].setdefault('description', '')

    lengths = [len(seq) for seq in inter_data.values()]
    stats = {
        'users': len(user2id), 'items': len(item2id),
        'interactions': sum(lengths), 'min_sequence_length': min(lengths),
        'mean_sequence_length': sum(lengths) / len(lengths),
        'filtered_users': original_users - len(user2id),
        'items_missing_metadata': len(item2id) - len(metadata),
        'min_interactions': min_interactions,
        'repeated_interactions': 'retained',
        'timestamp_ties': 'source order',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('inter_data.json', inter_data), ('Beauty.item.json', items),
                       ('item2id.json', item2id), ('user2id.json', user2id),
                       ('preprocess_stats.json', stats)]:
        with (output_dir / name).open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'JSON files saved to {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews_path', type=Path,
                        default=DATA_DIR / 'reviews_Beauty_5.json.gz')
    parser.add_argument('--metadata_path', type=Path,
                        default=DATA_DIR / 'meta_Beauty.json.gz')
    parser.add_argument('--output_dir', type=Path, default=DATA_DIR)
    parser.add_argument('--min_interactions', type=int, default=5)
    args = parser.parse_args()
    preprocess(args.reviews_path, args.metadata_path, args.output_dir, args.min_interactions)
