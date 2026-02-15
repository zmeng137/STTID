"""
map_test_wp5_1based.py

WikiPeople-5 mapping (relation + 4 entities). Uses 1-based IDs.

Behavior:
- Read ./<data_dir>/{train,valid,test}.txt
- For each line, keep ONLY the first 5 tokens: [rel, e1, e2, e3, e4]; ignore anything after
- Build GLOBAL vocabs (union across all three splits), sorted, 1-based IDs
- Map ONLY test.txt to IDs and write to --out
- Optionally write vocab files via --write_vocab_prefix

Usage example:
  python map_test_wp4_1based.py \
    --data_dir ./data/WikiPeople-4 \
    --out ./data/WikiPeople-4/test_ids_1based.txt \
    --write_vocab_prefix ./data/WikiPeople-4/vocab_1based
"""

import os
import argparse
from typing import List, Dict, Tuple

def read_rows_wp5(path: str) -> List[List[str]]:
    """Return rows as [rel, e1, e2, e3, e4]; skip lines with <5 tokens."""
    rows: List[List[str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            toks = ln.split()
            if len(toks) < 5:
                skipped += 1
                continue
            rows.append([toks[0], toks[1], toks[2], toks[3], toks[4]])
    if skipped:
        print(f"[warn] {path}: skipped {skipped} line(s) with fewer than 5 tokens.")
    return rows

def build_global_vocab(train: List[List[str]],
                       valid: List[List[str]],
                       test:  List[List[str]]
                      ) -> Tuple[Dict[str,int], Dict[str,int], List[str], List[str]]:
    """Union across splits, sorted, 1-based IDs."""
    rel_set = set()
    ent_set = set()
    for rows in (train, valid, test):
        for r, e1, e2, e3, e4 in rows:
            rel_set.add(r)
            ent_set.add(e1); ent_set.add(e2); ent_set.add(e3); ent_set.add(e4)

    relations = sorted(rel_set)
    entities  = sorted(ent_set)

    # 1-based
    rel2id = {tok: i+1 for i, tok in enumerate(relations)}
    ent2id = {tok: i+1 for i, tok in enumerate(entities)}
    return rel2id, ent2id, relations, entities

def map_rows_test(test_rows: List[List[str]],
                  rel2id: Dict[str,int],
                  ent2id: Dict[str,int]) -> List[List[int]]:
    mapped: List[List[int]] = []
    for r, e1, e2, e3, e4 in test_rows:
        mapped.append([rel2id[r], ent2id[e1], ent2id[e2], ent2id[e3], ent2id[e4]])
    return mapped

def write_rows(path: str, rows: List[List[int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")

def write_vocab(prefix: str, relations: List[str], entities: List[str]) -> None:
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    with open(prefix + ".relations.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate(relations, start=1):  # 1-based in file
            f.write(f"{i}\t{tok}\n")
    with open(prefix + ".entities.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate(entities, start=1):
            f.write(f"{i}\t{tok}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Folder containing train.txt, valid.txt, test.txt")
    ap.add_argument("--out", required=True,
                    help="Output mapped file for test.txt (1-based IDs)")
    ap.add_argument("--write_vocab_prefix", default=None,
                    help="If set, writes <prefix>.relations.txt and <prefix>.entities.txt (1-based)")
    args = ap.parse_args()

    train = read_rows_wp5(os.path.join(args.data_dir, "train.txt"))
    valid = read_rows_wp5(os.path.join(args.data_dir, "valid.txt"))
    test  = read_rows_wp5(os.path.join(args.data_dir, "test.txt"))

    rel2id, ent2id, relations, entities = build_global_vocab(train, valid, test)
    mapped_test = map_rows_test(test, rel2id, ent2id)  # Fixed: was mapping 'valid' instead of 'test'

    write_rows(args.out, mapped_test)
    if args.write_vocab_prefix:
        write_vocab(args.write_vocab_prefix, relations, entities)

    print("Done.")
    print(f"Lines -> train:{len(train)}  valid:{len(valid)}  test:{len(test)}")
    print(f"Unique relations (global, 1-based): {len(relations)}")
    print(f"Unique entities  (global, 1-based): {len(entities)}")
    print(f"Mapped test written to: {args.out}")
    if args.write_vocab_prefix:
        print(f"Vocab files: {args.write_vocab_prefix}.relations.txt, {args.write_vocab_prefix}.entities.txt")

if __name__ == "__main__":
    main()