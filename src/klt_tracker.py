class KLT_Tracker:
    
    def __init__(self, images, feature_params, lk_params):
        self.reference_image = images[0]
        self.images = images[1:]
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.reference_features = self.get_features()
        self.optical_flow = [ [(i.ravel()[0], i.ravel()[1])] for i in self.reference_features]

    def get_features(self):
        return cv2.goodFeaturesToTrack(self.reference_image, 
                                mask = None, 
                                **self.feature_params)

    def generate_optical_flow(self):

        optical_flow = self.optical_flow
        reference_features = self.reference_features
        for current_image in self.images:
            current_features, status, error = cv2.calcOpticalFlowPyrLK(self.reference_image, 
                                                                       current_image, 
                                                                       reference_features, 
                                                                       None, 
                                                                       **self.lk_params)

            reference_features = reference_features[status == 1]
            current_features = current_features[status == 1]

            optical_flow = [optical_flow[i] for i, j in enumerate(status) if j == 1]

            for i, feature in enumerate(current_features):
                optical_flow[i].append((feature.ravel()[0], feature.ravel()[1]))

            reference_features = reference_features.reshape((reference_features.shape[0],1,2))

        self.reference_features = reference_features
        self.optical_flow = optical_flow

        return NULL