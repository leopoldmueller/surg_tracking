from __future__ import print_function
import argparse
from ast import Try
from src.utils import load_yaml_data, download_data
from src.evaluate import evaluate_method
import warnings
import os

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Tool to label stereo matches")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "-nv",
        "--no-visualization",
        help="no visualization is shown.",
        action="store_true",
    )
    args = parser.parse_args()
    config = load_yaml_data(args.config)

    # Download data (if not downloaded before)
    if not os.path.exists('data'):
        download_data(config)

    print('Start FM-Tracker evaluation')
    evaluate_method(config=config, is_visualization_off=True, tracker_type="fm_tracker")
    print('End FM-Tracker evaluation')
                
if __name__ == "__main__":
    main()
