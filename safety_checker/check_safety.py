import json
from pathlib import Path

from safety_checker.src.dataloader_utils import collate_fn, get_dataloader
from safety_checker.src.parser_utils import get_parser
from safety_checker.src.transform_utils import get_transform
from safety_checker.src.workflow_utils import apply_workflow

# Reference:
# https://github.com/woctezuma/feature-extractor/blob/minimal/extract_fts.py


def main():
    params = get_parser().parse_args()
    print(f"__log__:{json.dumps(vars(params))}")

    print(">>> Creating dataloader...")
    img_loader = get_dataloader(
        params.data_dir,
        get_transform(params.resize_size, params.keep_ratio),
        batch_size=params.batch_size,
        collate_fn=collate_fn if params.keep_ratio else None,
    )

    print(">>> Detecting bad concepts...")
    aggregate, scores, sample_fnames = apply_workflow(
        img_loader,
        params.batch_size,
        verbose=params.verbose,
    )

    print(">>> Saving image paths and bad concepts...")
    output = {}
    for fname, bad_concepts in zip(sample_fnames, aggregate, strict=True):
        output[fname] = bad_concepts

    with Path(params.output).open("w") as f:
        json.dump(output, f, indent=True)


if __name__ == "__main__":
    main()
