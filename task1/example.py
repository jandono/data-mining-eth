import numpy as np
import random

MAXIMUM_HASH_FNS = 1020
LARGE_PRIME = 1000000007
TOTAL_SHINGLES = 8193

ROWS_COUNT = 30
BANDS_COUNT = MAXIMUM_HASH_FNS / ROWS_COUNT
BUCKETS_COUNT = 100000007 # Check later

ass = np.random.choice(range(1, TOTAL_SHINGLES), MAXIMUM_HASH_FNS)
bss = np.random.choice(range(TOTAL_SHINGLES), MAXIMUM_HASH_FNS)

band_ass = np.random.choice(range(1, TOTAL_SHINGLES), ROWS_COUNT)
band_bss = np.random.choice(range(TOTAL_SHINGLES), ROWS_COUNT)

def hash_vector(band):
    return np.sum(((np.multiply(band, band_ass) + band_bss) % LARGE_PRIME) % \
                BUCKETS_COUNT) % BUCKETS_COUNT


def mapper(key, value):
    # key: None
    # value: one line of input file
    docid_and_shingles = value.strip().split(' ')
    docid = int(docid_and_shingles[0].split('_')[1])
    shingles = np.array([int(s) for s in docid_and_shingles[1:]])

    # Compute min hashes.
    min_hashes = np.min(((np.outer(shingles, ass) + bss) % LARGE_PRIME) %
                            TOTAL_SHINGLES, axis=0)

    for b in range(BANDS_COUNT):
        bucket = hash_vector(min_hashes[b * ROWS_COUNT : (b + 1) * ROWS_COUNT])
        yield (str(bucket) + '_' + str(b)), docid


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    values = sorted(values)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            yield values[i], values[j]
