from __future__ import division

import numpy as np
import random


# We need to ensure that (1 / BANDS_COUNT) ^ (1 / ROWS_COUNT) is approximately
# TARGET_SIMILARITY.
ROWS_COUNT = 22
BANDS_COUNT = 46
MAX_HASH_FNS = ROWS_COUNT * BANDS_COUNT

TARGET_SIMILARITY = 0.85
TOTAL_SHINGLES = 8193
LARGE_PRIME = 1000000007

ass = np.array([random.randint(1, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])
bss = np.array([random.randint(0, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])


def mapper(key, value):
    # key: None
    # value: one line of input file
    docid_and_shingles = value.strip().split(' ')
    docid = int(docid_and_shingles[0].split('_')[1])
    shingles = np.array([int(s) for s in docid_and_shingles[1:]])

    # Compute min hashes.
    min_hashes = np.min(((np.outer(shingles, ass) + bss) % LARGE_PRIME) %
                            TOTAL_SHINGLES, axis=0)
    min_hashes.flags.writeable = False

    for b in range(BANDS_COUNT):
        bucket = hash(min_hashes[b * ROWS_COUNT : (b + 1) * ROWS_COUNT].data)
        # yield (str(bucket) + '_' + str(b)), (docid, min_hashes)
        yield (str(bucket) + '_' + str(b)), (docid, shingles)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    values = sorted(values)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            doc1, shingles1 = values[i]
            doc2, shingles2 = values[j]

            # emipiric_similarity = np.sum(min_hahes1 == min_hahes2) / \
            #                         len(min_hahes1) # MAX_HASH_FNS
            jaccard_similarity = len(np.intersect1d(shingles1, shingles2)) / \
                                    len(np.union1d(shingles1, shingles2))
            if jaccard_similarity >= TARGET_SIMILARITY:
                yield doc1, doc2
