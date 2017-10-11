from __future__ import division

import numpy as np
import random

TARGET_SIMILARITY = 0.85

# We need to ensure that (1 / BANDS_COUNT) ^ (1 / ROWS_COUNT) is approximately
# TARGET_SIMILARITY.
MAX_HASH_FNS = 966
LARGE_PRIME = 1000000007
TOTAL_SHINGLES = 8193

ROWS_COUNT = 23
BANDS_COUNT = MAX_HASH_FNS // ROWS_COUNT
BUCKETS_COUNT = 100000007 # Do we need this?

# NOTE: I think there is a mistake in Krause's slides: it seems that
# a,b ~ Unif(0/1, p), not Unif(0/1, shingles_count).
# Anyway, this did not change much.
ass = np.array([random.randint(1, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])
bss = np.array([random.randint(0, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])


def hash_vector(band):
    # We use regular hashes here.
    # TODO: Why do we need the buckets?
    return hash(band.data) % BUCKETS_COUNT


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
        bucket = hash_vector(min_hashes[b * ROWS_COUNT : (b + 1) * ROWS_COUNT])
        yield (str(bucket) + '_' + str(b)), (docid, min_hashes)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    values = sorted(values)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            doc1, min_hahes1 = values[i]
            doc2, min_hahes2 = values[j]
            # NOTE: 6. Examine each candidate pair’s signatures and determine
            # whether the fraction of components in which they agree is at
            # least t (MMDS (p. 91)).
            emipiric_similarity = np.sum(min_hahes1 == min_hahes2) / \
                                    len(min_hahes1) # MAX_HASH_FNS
            if emipiric_similarity > TARGET_SIMILARITY:
                yield doc1, doc2
