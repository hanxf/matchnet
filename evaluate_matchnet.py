"""This script evaluates a given matchnet model (including feature net and metric
   net) on a given ubc test set.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import leveldb
from caffe.proto import caffe_pb2
from matchnet import *
from eval_metrics import *


def ParseArgs():
    """Parse input arguments.
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('feature_net_model',
                        help='Feature network description.')
    parser.add_argument('feature_net_params',
                        help='Feature network parameters.')
    parser.add_argument('metric_net_model', help='Metric network description.')
    parser.add_argument('metric_net_params', help='Metric network parameters')
    parser.add_argument('test_db', help='A leveldb containing test patches.')
    parser.add_argument(
        'test_pairs',
        help=('Test pairs in text format. Patches should be in test_db. ' +
              'Following the original UBC dataset format, each line has ' +
              '6 integers separated by space, 3 for each patch. The three ' +
              'numbers for each point are: patch_id, 3D point_id, 0. ' +
              'Two patches match if their point_id match.'))
    parser.add_argument('output_txt',
                        help='Result file containing the predictions.')
    parser.add_argument('--use_gpu',
                        action='store_true',
                        dest='use_gpu',
                        help=('Switch to use gpu.'))
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        dest='gpu_id',
                        help=('GPU id. Effective only when --use_gpu=True.'))
    args = parser.parse_args()
    return args


def ReadPairs(filename):
    """Read pairs and match labels from the given file.
    """
    pairs = []
    labels = []
    with open(filename) as f:
        for line in f:
            parts = [p.strip() for p in line.split()]
            pairs.append((parts[0], parts[3]))
            labels.append(1 if parts[1] == parts[4] else 0)

    return pairs, labels


def ReadPatches(db, pairs, patch_height=64, patch_width=64):
    """Read patches from the given db handle. Each element in pairs is a
    pair of keys.

    Returns
    -------
    Two N * 1 * W * H array in a list, where N is the number of pairs.
    """
    N = len(pairs)
    patches = [np.zeros((N, 1, patch_height, patch_width),
                        dtype=np.float),
               np.zeros((N, 1, patch_height, patch_width),
                        dtype=np.float)]
    idx = 0  # Index to the next available patch in the patch array.
    parity = 0
    for pair in pairs:
        for key in pair:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(db.Get(key))
            patches[parity][idx, 0, :, :] = \
                np.fromstring(datum.data, np.uint8).reshape(
                patch_height, patch_width)
            parity = 1 - parity

        idx += 1

    return patches


def main():
    args = ParseArgs()

    # Initialize networks.
    feature_net = FeatureNet(args.feature_net_model, args.feature_net_params)
    metric_net = MetricNet(args.metric_net_model, args.metric_net_params)

    if args.use_gpu:
        caffe.set_mode_gpu()
        print "GPU mode"
    else:
        caffe.set_mode_cpu()
        print "CPU mode"

    # Read the test pairs.
    pairs, labels = ReadPairs(args.test_pairs)

    # Open db.
    db = leveldb.LevelDB(args.test_db, create_if_missing=False)
    assert db is not None

    # Compute matching prediction.
    start_idx = 0  # Start index for a batch.
    N = len(labels)  # Total number of pairs.
    scores = np.zeros(N, dtype=np.float)
    while start_idx < N:
        # Index after the last item in the batch.
        stop_idx = min(start_idx + feature_net.GetBatchSize(), N)
        print "Block (%d,%d)" % (start_idx, stop_idx)

        # Read features.
        input_patches = ReadPatches(db, pairs[start_idx:stop_idx])

        # Compute features.
        feats = [feature_net.ComputeFeature(input_patches[0]),
                 feature_net.ComputeFeature(input_patches[1])]

        # # Compute scores.
        scores[start_idx:stop_idx] = \
            metric_net.ComputeScore(feats[0], feats[1])

        start_idx = stop_idx

    # Compute evaluation metrics.
    error_at_95 = ErrorRateAt95Recall(labels, scores)
    print "Error rate at 95%% recall: %0.2f%%" % (error_at_95 * 100)


if __name__ == '__main__':
    main()
