"""This script creates a leveldb database for a given UBC patch dataset. For
each patch we generate a key-value pair:
    key: the patch id (zero-starting line index in the info.txt file).
  value: a Caffe Datum containing the image patch and the metadata.

It will complain if the specified db already exists.

Example:
  python generate_patch_db.py data/phototour/liberty/info.txt \
    data/phototour/liberty/interest.txt \
    data/phototour/liberty data/leveldb/liberty.leveldb

"""

import leveldb, numpy as np, skimage
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from caffe.proto import caffe_pb2
from caffe.io import *


def ParseArgs():
    """Parse input arguments.
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('info_file',
                        help='Path to info.txt file in the dataset.')
    parser.add_argument('interest_file',
                        help='Path to interest.txt file in the dataset.')
    parser.add_argument('container_dir',
                        help='Patch to the directory of .bmp files.')
    parser.add_argument('output_db', help='Path to output database.')
    args = parser.parse_args()
    return args


def GetPatchImage(patch_id, container_dir):
    """Returns a 64 x 64 patch with the given patch_id. Catch container images to
       reduce loading from disk.
    """
    # Define constants. Each container image is of size 1024x1024. It packs at
    # most 16 rows and 16 columns of 64x64 patches, arranged from left to right,
    # top to bottom.
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64

    # Calculate the container index, the row and column index for the given
    # patch.
    container_idx, container_offset = divmod(patch_id, PATCHES_PER_IMAGE)
    row_idx, col_idx = divmod(container_offset, PATCHES_PER_ROW)

    # Read the container image if it is not cached.
    if GetPatchImage.cached_container_idx != container_idx:
        GetPatchImage.cached_container_idx = container_idx
        GetPatchImage.cached_container_img = \
            skimage.img_as_ubyte(skimage.io.imread('%s/patches%04d.bmp' % \
                (container_dir, container_idx), as_grey=True))

    # Extract the patch from the image and return.
    patch_image = GetPatchImage.cached_container_img[ \
        PATCH_SIZE * row_idx:PATCH_SIZE * (row_idx + 1), \
        PATCH_SIZE * col_idx:PATCH_SIZE * (col_idx + 1)]
    return patch_image

# Static variables initialization for GetPatchImage.
GetPatchImage.cached_container_idx = None
GetPatchImage.cached_container_img = None


def main():
    # Parse input arguments.
    args = ParseArgs()

    # Read the 3Dpoint IDs from the info file.
    with open(args.info_file) as f:
        point_id = [int(line.split()[0]) for line in f]

    # Read the interest point from the interest file. The fields in each line
    # are: image_id, x, y, orientation, and scale. We parse all of them as float
    # even though image_id is integer.
    with open(args.interest_file) as f:
        interest = [[float(x) for x in line.split()] for line in f]

    # Create the output database, fail if exists.
    db = leveldb.LevelDB(args.output_db,
                         create_if_missing=True,
                         error_if_exists=True)

    # Add patches to the database in batch.
    batch = leveldb.WriteBatch()
    total = len(interest)
    processed = 0
    for i, metadata in enumerate(interest):
        datum = caffe_pb2.Datum()
        datum.channels, datum.height, datum.width = (1, 64, 64)

        # Extract the patch
        datum.data = GetPatchImage(i, args.container_dir).tostring()

        # Write 3D point ID into the label field.
        datum.label = point_id[i]

        # Write other metadata into float_data fields.
        datum.float_data.extend(metadata)
        batch.Put(str(i), datum.SerializeToString())
        processed += 1
        if processed % 1000 == 0:
            print processed, '/', total

            # Write the current batch.
            db.Write(batch, sync=True)

            # Verify the last written record.
            d = caffe_pb2.Datum()
            d.ParseFromString(db.Get(str(processed - 1)))
            assert (d.data == datum.data)

            # Start a new batch
            batch = leveldb.WriteBatch()
    db.Write(batch, sync=True)


if __name__ == '__main__':
    main()
