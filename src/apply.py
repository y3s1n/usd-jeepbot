"""
Apply the decisions from review.py

Usage:
    python apply.py           # dry run - shows what WOULD happen
    python apply.py --go      # actually do it

  keep    -> left alone
  move    -> moved to the matching folder in the other class
  reject  -> moved to dataset\\_rejected\\

Nothing is ever deleted. Rejects go to a folder you can inspect or restore.
"""

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
CSV_PATH = ROOT / "review_decisions.csv"
REJECT_DIR = DATASET / "_rejected"

GO = "--go" in sys.argv


def other_class(folder_name):
    """empty_A_up_sun <-> person_A_up_sun"""
    if folder_name.startswith("empty_"):
        return "person_" + folder_name[6:]
    if folder_name.startswith("person_"):
        return "empty_" + folder_name[7:]
    return None


def main():
    if not CSV_PATH.exists():
        print(f"no decisions file at {CSV_PATH} - run review.py first")
        return

    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))

    planned = defaultdict(int)
    moves = []

    for row in rows:
        src = DATASET / row["path"]
        if not src.exists():
            planned["missing"] += 1
            continue

        if row["decision"] == "keep":
            planned["keep"] += 1

        elif row["decision"] == "move":
            dest_folder = other_class(row["folder"])
            if dest_folder is None:
                planned["unmovable"] += 1
                continue
            moves.append((src, DATASET / dest_folder / src.name))
            planned["move"] += 1

        elif row["decision"] == "reject":
            moves.append((src, REJECT_DIR / row["folder"] / src.name))
            planned["reject"] += 1

    print("PLAN")
    for k, v in sorted(planned.items()):
        print(f"  {k:<12}{v:>6}")

    if not GO:
        print("\nDRY RUN - nothing changed.")
        if moves:
            print("\nfirst few moves:")
            for src, dst in moves[:8]:
                print(f"  {src.parent.name}/{src.name}  ->  "
                      f"{dst.parent.parent.name + '/' if dst.parent.parent.name == '_rejected' else ''}"
                      f"{dst.parent.name}/")
        print("\nrerun with  --go  to apply.")
        return

    done = 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = dst
        n = 1
        while target.exists():
            target = dst.with_stem(f"{dst.stem}_dup{n}")
            n += 1
        shutil.move(str(src), str(target))
        done += 1

    print(f"\nmoved {done} files.")
    print("Re-run audit.py to see the cleaned dataset.")


if __name__ == "__main__":
    main()
