class KLT_Tracker:
    
    def __init__(self, images, feature_params, lk_params):
        self.reference_image = images[0]
        self.images = images[1:]
        self.reference_features = None
        self.optical_flow = None
        self.feature_params = feature_params
        self.lk_params = lk_params
        pass

    def get_features(self):

        pass

    def track_features(self):
        pass
    
    def homography_filter(self):
        pass
