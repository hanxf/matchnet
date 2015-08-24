# Generate dataset from the original phototour dataset.
# See data/README.md for instructions to download the dataset.

DATASET=liberty
python generate_patch_db.py data/phototour/${DATASET}/info.txt \
    data/phototour/${DATASET}/interest.txt \
    data/phototour/${DATASET} data/leveldb/${DATASET}.leveldb

DATASET=notredame
python generate_patch_db.py data/phototour/${DATASET}/info.txt \
    data/phototour/${DATASET}/interest.txt \
    data/phototour/${DATASET} data/leveldb/${DATASET}.leveldb

DATASET=yosemite
python generate_patch_db.py data/phototour/${DATASET}/info.txt \
    data/phototour/${DATASET}/interest.txt \
    data/phototour/${DATASET} data/leveldb/${DATASET}.leveldb


