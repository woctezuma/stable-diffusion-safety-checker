import argparse

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/src/parser_utils.py


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_dir",
        "--data-dir",
        "--data",
        "--input",
        type=str,
        default="img",
        help="The path to the input folder where images are stored.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--resize_size",
        "--resize-size",
        "--resize",
        type=int,
        default=256,
        help="Desired image output size after the resize.",
    )
    parser.add_argument(
        "--keep_ratio",
        "--keep-ratio",
        action="store_true",
        help="Whether to keep the image ratio: the smallest image side will match `resize_size`.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="bad_concepts.json",
        help="An output file with a list of IDs of bad concepts detected in each image.",
    )
    parser.add_argument(
        "--output_scores",
        "--output-scores",
        "--scores",
        type=str,
        default="bad_concepts_scores.pth",
        help="An output file with the scores for the bad concepts in PyTorch format.",
    )
    parser.add_argument(
        "--img_list",
        "--img-list",
        "--output_list",
        "--output-list",
        "--list",
        type=str,
        default="img_list.json",
        help="An output file with the list of image paths in the order used by the data loader.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to increase output verbosity.",
    )

    return parser
