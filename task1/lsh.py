from __future__ import division

import numpy as np
import random


# We need to ensure that (1 / BANDS_COUNT) ^ (1 / ROWS_COUNT) is approximately
# TARGET_SIMILARITY.
ROWS_COUNT = 20
BANDS_COUNT = 50
MAX_HASH_FNS = ROWS_COUNT * BANDS_COUNT

TARGET_SIMILARITY = 0.85
TOTAL_SHINGLES = 8193
LARGE_PRIME = 1000000007

ass = np.array([random.randint(1, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])
bss = np.array([random.randint(0, LARGE_PRIME - 1)
                for _ in range(MAX_HASH_FNS)])

# Official Solution #1 - Most accurate, but does not scale well
# in a real world scenario, with a large corpora and large number
# of shingles per document.
# Use min hashing to get candidate documents in the mapper and
# explicitly perform Jaccard similarity in the reducer.

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
        yield (str(bucket) + '_' + str(b)), (docid, shingles)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    values = sorted(values)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            doc1, shingles1 = values[i]
            doc2, shingles2 = values[j]

            jaccard_similarity = len(np.intersect1d(shingles1, shingles2)) / \
                                    len(np.union1d(shingles1, shingles2))
            if jaccard_similarity >= TARGET_SIMILARITY:
                yield doc1, doc2


# Solution #2 - Almost as accurate, but better scaling in real world
# scenarios, because the complexity (both time and memory) is a function of the
# number of min hash functions used instead of the number of document shingles.

# def mapper(key, value):
#     # key: None
#     # value: one line of input file
#     docid_and_shingles = value.strip().split(' ')
#     docid = int(docid_and_shingles[0].split('_')[1])
#     shingles = np.array([int(s) for s in docid_and_shingles[1:]])

#     # Compute min hashes.
#     min_hashes = np.min(((np.outer(shingles, ass) + bss) % LARGE_PRIME) %
#                             TOTAL_SHINGLES, axis=0)
#     min_hashes.flags.writeable = False

#     for b in range(BANDS_COUNT):
#         bucket = hash(min_hashes[b * ROWS_COUNT : (b + 1) * ROWS_COUNT].data)
#         yield (str(bucket) + '_' + str(b)), (docid, min_hashes)

# def reducer(key, values):
#     # key: key from mapper used to aggregate
#     # values: list of all value for that key
#     values = sorted(values)
#     for i in range(len(values)):
#         for j in range(i + 1, len(values)):
#             doc1, min_hahes1 = values[i]
#             doc2, min_hahes2 = values[j]

#             emipiric_similarity = np.sum(min_hahes1 == min_hahes2) / \
#                                     len(min_hahes1) # MAX_HASH_FNS
#             if emipiric_similarity >= TARGET_SIMILARITY:
#                 yield doc1, doc2


# Solution #3 - Same approach as Solution #2, with a different metric of the
# similarity in the reducer. This solution computes the Jaccard similarity on
# the min hash vectors of condidate documents.

# def mapper(key, value):
#     # key: None
#     # value: one line of input file
#     docid_and_shingles = value.strip().split(' ')
#     docid = int(docid_and_shingles[0].split('_')[1])
#     shingles = np.array([int(s) for s in docid_and_shingles[1:]])

#     # Compute min hashes.
#     min_hashes = np.min(((np.outer(shingles, ass) + bss) % LARGE_PRIME) %
#                             TOTAL_SHINGLES, axis=0)
#     min_hashes.flags.writeable = False

#     for b in range(BANDS_COUNT):
#         bucket = hash(min_hashes[b * ROWS_COUNT : (b + 1) * ROWS_COUNT].data)
#         yield (str(bucket) + '_' + str(b)), (docid, min_hashes)

# def reducer(key, values):
#     # key: key from mapper used to aggregate
#     # values: list of all value for that key
#     values = sorted(values)
#     for i in range(len(values)):
#         for j in range(i + 1, len(values)):
#             doc1, min_hahes1 = values[i]
#             doc2, min_hahes2 = values[j]

#             jaccard_similarity = len(np.intersect1d(min_hahes1, min_hahes2)) / \
#                                     len(np.union1d(min_hahes1, min_hahes2))
#             if jaccard_similarity >= TARGET_SIMILARITY:
#                 yield doc1, doc2
