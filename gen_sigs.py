import sys
import random
import string

chars = string.hexdigits.lower()[:16] + "?"

with open('my-sigs-double-length.txt', 'w') as f:
    for i in range(10000):
        filename = f'virus.test.myname-{i}'
        num_bytes = random.randrange(32, 4096 + 1)
        num_bytes *= 2
        sig = []
        wild_cnt = 0
        for _ in range(num_bytes * 2):
            nxt_chr = random.choice(chars)
            if nxt_chr == '?':
                wild_cnt += 1
            if wild_cnt == 4:
                nxt_chr = random.choice(chars[:-1])
            if nxt_chr != '?':
                wild_cnt = 0
            sig.append(nxt_chr)
        sig = ''.join(sig)
        f.write(f"{filename}:{sig}\n")
