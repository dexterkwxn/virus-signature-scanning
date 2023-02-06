#!/usr/bin/python3

import os
import sys
import random
import itertools
from textwrap import wrap
from typing import Tuple, List

SEED = os.urandom(32)
MIN_FILE_SIZE = 4096
MAX_FILE_SIZE = 8 * 1024 * 1024

def random_bytes(n: int) -> bytes:
	if sys.version_info[1] >= 9:        # random.randbytes only in 3.9+
		return random.randbytes(n)
	else:
		return bytes([ random.getrandbits(8) for _ in range(0, n) ])


def human_readable_size(sz: int) -> str:
	if sz < 1024:
		return f"{sz} bytes"
	elif sz < 1024*1024:
		return f"{sz / 1024:.2f} kB"
	else:
		return f"{sz / (1024*1024):.2f} MB"


def gen_file_size(total_virus_size: int) -> int:
	min_size = max(MIN_FILE_SIZE, total_virus_size + 1024)

	# https://observablehq.com/@herbps10/beta-distribution
	return max(min_size, 2 * int(MAX_FILE_SIZE * random.betavariate(1.1, 10.5)))



def to_bytes(sig: str) -> bytes:
	num_wildcard = sig.count('?')
	replacements = random.choices("0123456789abcdef", k=num_wildcard)
	parts = [x for x in itertools.chain(*itertools.zip_longest(sig.split('?'), replacements)) if x is not None]
	return bytes.fromhex(''.join(parts))


def generate_benign(idx: int, sigs: List[Tuple[str, str]]) -> Tuple[str, bytes]:
	split_into = int(2.3 * len(sigs))
	sigs_in_bytes = [wrap(sig[1], 2) for sig in sigs]
	pieces = list(map(lambda s: [''.join(s[i::split_into]) for i in range(split_into)], sigs_in_bytes))
	pieces = list(itertools.chain.from_iterable(pieces))
	random.shuffle(pieces)

	total_virus_size = sum(map(len, pieces)) // 2
	file_size = gen_file_size(total_virus_size)

	file = bytearray()
	remaining_virus_size = total_virus_size

	p = 0
	just_added = False
	while p < len(pieces):
		remaining = file_size - remaining_virus_size - len(file)
		if random.random() < 0.5 or just_added:
			just_added = False
			file += random_bytes(random.randrange(0, remaining))
		else:
			just_added = True
			p_bytes = to_bytes(pieces[p])
			remaining_virus_size -= len(p_bytes)
			file += p_bytes
			p += 1

	if len(file) < file_size:
		file += random_bytes(file_size - len(file))

	return f"benign-{idx:04}", file


def generate_virus(idx: int, sigs: List[Tuple[str, str]]) -> Tuple[str, bytes]:
	total_virus_size = sum(map(lambda s: len(s[1]), sigs)) // 2
	file_size = gen_file_size(total_virus_size)

	file = bytearray()
	remaining_virus_size = total_virus_size

	v = 0
	while v < len(sigs):
		remaining = file_size - remaining_virus_size - len(file)
		if random.random() < 0.5:
			file += random_bytes(random.randrange(0, remaining))
		else:
			v_bytes = to_bytes(sigs[v][1])
			file += v_bytes
			remaining_virus_size -= len(v_bytes)
			v += 1

	if len(file) < file_size:
		file += random_bytes(file_size - len(file))

	return (f"virus-{idx:04}-" + '+'.join(map(lambda s: s[0], sigs))), file



def main(sig_file: str, output_dir: str, num_files: int, virus_chance: float, max_viruses: int):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	random.seed(SEED)
	print(f"seed = {SEED.hex()}")

	print(f"num files:    {num_files}")
	print(f"virus chance: {100 * virus_chance:.1f}%")
	print(f"max per file: {max_viruses}")

	with open(sig_file, "r") as f:
		all_sigs = [line.strip().split(':') for line in f]

	virus_idx = 1
	benign_idx = 1
	for i in range(0, num_files):
		is_virus = random.random() < virus_chance

		virus_count = int(1 + max_viruses * random.betavariate(2, 5))
		viruses = random.choices(all_sigs, k=virus_count)

		if is_virus:
			filename, file = generate_virus(virus_idx, viruses)
			virus_idx += 1
		else:
			filename, file = generate_benign(benign_idx, viruses)
			benign_idx += 1

		print(f"{human_readable_size(len(file))} - {filename}")
		open(f"{output_dir}/{filename}.in", "wb").write(file)



if __name__ == "__main__":
	if len(sys.argv) < 5:
		print(f"usage: ./gen_tests.py <signatures.txt> <output_dir> <num_files> <virus_chance> <max_viruses_per_file>")
		sys.exit(0)

	main(sys.argv[1], sys.argv[2], int(sys.argv[3]), min(1, max(-1, float(sys.argv[4]))), int(sys.argv[5]))
