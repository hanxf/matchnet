import numpy as np
import caffe


class FeatureNet(caffe.Net):
    """Feature network
    """

    def __init__(self, model_file, pretrained_file):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.batch_size_ = self.blobs[self.inputs[0]].num

    def ComputeFeature(self, inputs):
        """
        Compute features for given inputs.
        
        Parameters
        ----------
        inputs: N x 1 x H x W array. N is the number of patches.
        W and H should match the input layer's width and height.
        
        Returns
        -------
        feats: (N x F x 1 x 1) array of features. F is the feature dimension.
        """
        # Preprocessing.
        in_ = (inputs.astype(np.float32) - 128) / 160
        # Compute features.
        out_ = self.forward_all(**{self.inputs[0]: in_})
        # Reshape features into a 1-D feature vectors.
        feats = out_[self.outputs[0]].reshape((len(in_), -1, 1, 1))
        return feats

    def GetBatchSize(self):
        return self.batch_size_


class MetricNet(caffe.Net):
    """Metric Network
    """

    def __init__(self, model_file, pretrained_file):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.batch_size_ = self.blobs[self.inputs[0]].num

    def ComputeScore(self, inputs1, inputs2):
        """
        Compute matching scores for input pairs.
        
        Parameters
        ----------
        inputs1, inputs2: N x F x 1 x 1 array. 
        N is the number of patches. F is the feature dimension.
        
        Returns
        -------
        scores: flatten array with N elements as matching scores.
        0: not matching, 1: matching.
        """
        # Check the batch size.
        assert inputs1.shape[0] == inputs2.shape[0]
        in_ = np.concatenate((inputs1, inputs2), axis=1)
        out_ = self.forward_all(**{self.inputs[0]: in_})
        scores = out_[self.outputs[0]][:, 1]
        return scores.flatten()

    def GetBatchSize(self):
        return self.batch_size_
