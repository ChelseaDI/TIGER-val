"""Encode selected Beauty metadata; tensor row i always represents item i."""

import argparse
import html
import json
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_PLM_DIR = DATA_DIR.parents[2] / 'LLM'
TEXT_FIELDS = ('title', 'description', 'brand', 'categories')


def clean_text(value):
    if isinstance(value, (list, tuple)):
        return ' '.join(filter(None, (clean_text(part) for part in value)))
    if isinstance(value, dict):
        return ' '.join(filter(None, (clean_text(part) for part in value.values())))
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    return ' '.join(text.split())


def text_values(value):
    """Flatten only strings, excluding numbers, identifiers and URLs in auto mode."""
    if isinstance(value, dict):
        return ' '.join(filter(None, (text_values(v) for v in value.values())))
    if isinstance(value, (list, tuple)):
        return ' '.join(filter(None, (text_values(v) for v in value)))
    if isinstance(value, str) and not re.match(r'^https?://', value.strip()):
        return clean_text(value)
    return ''


def get_item_text(item_path, features):
    with item_path.open(encoding='utf-8') as stream:
        items = json.load(stream)
    expected_ids = {str(i) for i in range(len(items))}
    if not items or set(items) != expected_ids:
        raise ValueError('Item IDs must be consecutive strings from 0 to N-1')
    available = set().union(*(item.keys() for item in items.values()))
    auto = features == ['all_text']
    if 'all_text' in features and not auto:
        raise ValueError('Use all_text alone, or specify individual field names')
    if auto:
        # Stable semantic order, then any additional textual metadata fields.
        excluded = {'asin', 'imUrl', 'image', 'imageURL', 'imageURLHighRes',
                    'related', 'salesRank', 'price'}
        features = [k for k in TEXT_FIELDS if k in available]
        features += sorted(available - set(features) - excluded)
    else:
        unknown = set(features) - available
        if unknown:
            raise ValueError(f'Unknown metadata fields: {sorted(unknown)}; '
                             f'available fields: {sorted(available)}. '
                             'Rerun preprocess.py to preserve all metadata.')
    print(f'Metadata fields: {features}; automatic text-only mode: {auto}')
    texts = []
    fallback_count = 0
    for item_id in range(len(items)):
        item = items[str(item_id)]
        cleaner = text_values if auto else clean_text
        parts = [cleaner(item.get(feature)) for feature in features]
        text = ' '.join(part.rstrip('.') + '.' for part in parts if part)
        if not text:
            text = f"Product {item.get('asin') or item_id}."
            fallback_count += 1
        texts.append(text)
    print(f'Items: {len(texts)}; items using ASIN fallback: {fallback_count}')
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item_path', type=Path, default=DATA_DIR / 'Beauty.item.json')
    parser.add_argument('--plm_dir', type=Path, default=DEFAULT_PLM_DIR)
    parser.add_argument('--plm_name', default='sentence-t5-base', choices=['sentence-t5-base'])
    parser.add_argument('--device', default=None, help='e.g. cuda:0 or cpu; auto-detect by default')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_sent_len', type=int, default=512)
    parser.add_argument('--features', nargs='+', default=['title', 'price', 'brand', 'categories'],
                        help='Metadata fields in concatenation order, or all_text '
                             'for all textual metadata excluding IDs, URLs and relations')
    parser.add_argument('--output_path', type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_sent_len <= 0:
        parser.error('batch_size and max_sent_len must be positive')

    model_path = args.plm_dir / args.plm_name
    if not model_path.is_dir():
        parser.error(f'Model directory not found: {model_path}')
    texts = get_item_text(args.item_path, args.features)

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(str(model_path.resolve()), device=device)
    model.max_seq_length = args.max_sent_len
    model.eval()
    # Preserve Sentence-T5's pooling, projection and normalization modules.
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=args.batch_size,
                                  show_progress_bar=True, convert_to_tensor=True)
    embeddings = embeddings.detach().cpu().float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(f'Unexpected embedding shape: {tuple(embeddings.shape)}')
    if not torch.isfinite(embeddings).all():
        raise ValueError('Embeddings contain NaN or infinity')
    output_path = args.output_path or DATA_DIR / f'text_embed_{args.plm_name}.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f'Saved {tuple(embeddings.shape)} float32 tensor to {output_path.resolve()}')


if __name__ == '__main__':
    main()
