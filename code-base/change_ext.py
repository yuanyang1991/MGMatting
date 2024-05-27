import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument("--ext", type=str, required=True)
    args = parser.parse_args()

    files = os.listdir(args.dir)
    for file in files:
        base_file, ext = os.path.splitext(file)
        new_filename = base_file + args.ext
        old = os.path.join(args.dir, file)
        new = os.path.join(args.dir, new_filename)
        os.rename(old, new)
        print(f"Renamed '{old}' to '{new}'")
