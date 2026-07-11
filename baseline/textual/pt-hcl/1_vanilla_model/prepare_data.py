"""Prepare data for PT-HCL vanilla model with new data format."""

import json
import argparse
from pathlib import Path


def convert_data_format(input_path, output_path):
    """Convert submission data format to PT-HCL vanilla format."""

    # Load submission format data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to PT-HCL format
    converted_data = []
    for item in data:
        # Split article into sentences (simple approach)
        article_text = item.get('article', '')
        sentences = article_text.split('. ')

        # Create sentences dict
        sentences_dict = {}
        for i, sent in enumerate(sentences):
            if sent.strip():
                sentences_dict[f'sentence_{i}'] = sent.strip()

        converted_item = {
            'id': item.get('id'),
            'issue': item.get('issue', ''),
            'sentences': sentences_dict,
            'article': article_text,
            'stance': item.get('stance', 'neutral'),  # Use English stance directly
            'split': item.get('split', 'train')
        }

        converted_data.append(converted_item)

    # Save converted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted_data)} items")
    print(f"Saved to {output_path}")

    # Print statistics
    splits = {}
    stances = {}
    for item in converted_data:
        split = item['split']
        stance = item['stance']
        splits[split] = splits.get(split, 0) + 1
        stances[stance] = stances.get(stance, 0) + 1

    print("\nSplit distribution:")
    for split, count in splits.items():
        print(f"  {split}: {count}")

    print("\nStance distribution:")
    for stance, count in stances.items():
        print(f"  {stance}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for PT-HCL vanilla model")
    parser.add_argument(
        "--input_path",
        type=str,
        default="../../../data/k-news-stance-mm-acl-arr.json",
        help="Path to input dataset in submission format"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dataset_for_vanilla.json",
        help="Path to output dataset for vanilla model"
    )

    args = parser.parse_args()
    convert_data_format(args.input_path, args.output_path)


if __name__ == "__main__":
    main()