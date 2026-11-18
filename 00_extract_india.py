"""
STEP 0.5 - TAKE ONLY THE INDIA PHOTOS OUT OF THE BIG ZIP

The big zip (12 GB) has photos from many countries.
Each country is its own small zip INSIDE the big zip (like a box inside a box).
We only need India (about 0.5 GB), so this script:
   1. takes the India box out of the big zip,
   2. opens the India box,
   3. puts the photos + labels in a new folder called  rdd_india

How to run (Terminal, inside road-project, with .venv on):
    python 00_extract_india.py  ~/Downloads/RDD2022_released_through_CRDDC2022.zip
"""

import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


def is_india_file(info):
    """True for a real file that lives in a folder called India (and is not macOS junk)."""
    parts = PurePosixPath(info.filename).parts
    return (not info.is_dir()) and "India" in parts and "__MACOSX" not in parts


def extract_files(zf, infos, out_dir):
    for n, info in enumerate(infos, 1):
        zf.extract(info, out_dir)
        if n % 2000 == 0:
            print(f"   {n} / {len(infos)}")


def main():
    if len(sys.argv) < 2:
        print("Tell me where the zip is, like this:")
        print("   python 00_extract_india.py ~/Downloads/RDD2022_released_through_CRDDC2022.zip")
        sys.exit(1)

    zip_path = Path(sys.argv[1]).expanduser()
    out_dir = Path("rdd_india")

    if not zip_path.exists():
        print("I cannot find this file:", zip_path)
        print("(If Chrome is still downloading, the name ends with .crdownload - wait until it finishes.)")
        sys.exit(1)

    try:
        big = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        print("This zip is not complete or is broken. Is the download finished?")
        sys.exit(1)

    with big:
        # Case 1: India files are loose inside the big zip
        loose = [i for i in big.infolist() if is_india_file(i)]
        if loose:
            mb = sum(i.file_size for i in loose) / 1e6
            print(f"Found {len(loose)} India files ({mb:.0f} MB). Copying out...")
            extract_files(big, loose, out_dir)
        else:
            # Case 2: India is its own zip inside the big zip (the real dataset looks like this)
            boxes = [i for i in big.infolist()
                     if PurePosixPath(i.filename).name.lower() in ("india.zip", "rdd2022_india.zip")
                     and "__MACOSX" not in i.filename]
            if not boxes:
                print("I found no India files inside. Tell Claude the first 10 names below:")
                for info in big.infolist()[:10]:
                    print("   ", info.filename)
                sys.exit(1)
            box = boxes[0]
            print(f"Found the India box: {box.filename} ({box.file_size / 1e6:.0f} MB)")
            print("Step 1 of 2: taking the India box out of the big zip...")
            out_dir.mkdir(exist_ok=True)
            box_path = Path(big.extract(box, out_dir / "_tmp"))

            print("Step 2 of 2: opening the India box...")
            try:
                inner = zipfile.ZipFile(box_path)
            except zipfile.BadZipFile:
                print("The India box is broken. The big download may be incomplete.")
                sys.exit(1)
            with inner:
                files = [i for i in inner.infolist()
                         if not i.is_dir() and "__MACOSX" not in PurePosixPath(i.filename).parts]
                print(f"   {len(files)} files inside. Copying out...")
                extract_files(inner, files, out_dir)

            # remove the temporary box we made (it is only our own temporary copy)
            shutil.rmtree(out_dir / "_tmp", ignore_errors=True)

    print("\nDone! Everything is in the folder:", out_dir.resolve())
    print("Next step:")
    print("   python 01_look_at_data.py", out_dir)


if __name__ == "__main__":
    main()
