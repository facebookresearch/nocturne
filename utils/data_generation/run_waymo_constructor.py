# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utils for converting TFRecords into Nocturne compatible JSON."""
import argparse
from pathlib import Path
import os
import multiprocessing

# TODO(ev) remove hardcoding
TRAIN_DATA_PATH = "/scratch/ev2237/waymo_motion/training"
VALID_DATA_PATH = "/scratch/ev2237/waymo_motion/validation"
PROCESSED_TRAIN_NO_TL = "/scratch/ev2237/waymo_processed/train_no_tl"
PROCESSED_VALID_NO_TL = "/scratch/ev2237/waymo_processed/valid_no_tl"
PROCESSED_TRAIN = "/scratch/ev2237/waymo_processed/train_processed"
PROCESSED_VALID = "/scratch/ev2237/waymo_processed/valid_processed"
import waymo_scenario_construction as waymo


def convert_files(args, files, output_dir, rank):
    """Convert the list of files into nocturne compatible JSON.

    Args
    ----
        args (NameSpace): args from the argument parser.
        files ([str]): list of file paths for TFRecords that we should convert
        output_dir (str): output path in which we should store the JSON
        rank (int): rank of the process.
    """
    cnt = 0
    for file in files:
        inner_count = 0
        for data in waymo.load_protobuf(str(file)):
            file_name = os.path.basename(file).split(
                '.')[1] + f'_{inner_count}.json'
            # this file is useful for debugging
            if args.output_txt and cnt == 0 and rank == 0:
                with open(os.path.basename(file).split('.')[1] + '.txt',
                          'w') as f:
                    f.write(str(data))
            waymo.waymo_to_scenario(os.path.join(output_dir, file_name), data,
                                    args.no_tl)
            inner_count += 1
            cnt += 1
            if cnt >= args.num and not args.all_files:
                break
        print(inner_count)


def main():
    """Run the json generators."""
    parser = argparse.ArgumentParser(
        description="Load and show waymo scenario data.")
    parser.add_argument("--file",
                        type=str,
                        default=os.path.join(
                            TRAIN_DATA_PATH,
                            'training.tfrecord-00997-of-01000'))
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--output_txt",
                        action='store_true',
                        help='output a txt version of one of the protobufs')
    parser.add_argument("--all_files",
                        action='store_true',
                        help='If true, iterate through the whole dataset')
    parser.add_argument("--no_tl",
                        action='store_true',
                        help="If true, do not generate JSON files\
             that have a traffic light in them")
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="If true, split the conversion up over multiple processes")
    parser.add_argument("--datatype",
                        default='train',
                        type=str,
                        choices=['train', 'valid'],
                        nargs='+',
                        help="Whether to convert, train or valid data")

    args = parser.parse_args()
    folders_to_convert = []
    if 'train' in args.datatype:
        folders_to_convert.append(
            (TRAIN_DATA_PATH,
             PROCESSED_TRAIN_NO_TL if args.no_tl else PROCESSED_TRAIN))
    if 'valid' in args.datatype:
        folders_to_convert.append(
            (VALID_DATA_PATH,
             PROCESSED_VALID_NO_TL if args.no_tl else PROCESSED_VALID))

    for folder_path, output_dir in folders_to_convert:
        if args.num > 1 or args.all_files:
            files = list(Path(folder_path).glob('*tfrecord*'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not args.all_files:
                files = files[0:args.num]

        else:
            output_dir = os.getcwd()
            files = [args.file]

        if args.parallel:
            # leave some cpus free but have at least one and don't use more than 40
            num_cpus = min(max(multiprocessing.cpu_count() - 2, 1), 40)
            num_files = len(files)
            process_list = []
            for i in range(num_cpus):
                p = multiprocessing.Process(
                    target=convert_files,
                    args=[
                        args, files[i * num_files // num_cpus:(i + 1) *
                                    num_files // num_cpus], output_dir, i
                    ])
                p.start()
                process_list.append(p)

            for process in process_list:
                process.join()
        else:
            convert_files(args, files, output_dir, rank=0)


if __name__ == "__main__":
    main()