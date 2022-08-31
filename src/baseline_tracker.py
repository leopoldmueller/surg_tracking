import cv2

class Baseline_Tracker:
    """ This is a sample tracker, replace this class with your own tracker! """
    def __init__(self, init_im1, init_im2, init_bbox1, init_bbox2, tracker_type="BOOSTING"):
        
        if tracker_type == 'BOOSTING':
            self.t1 = cv2.legacy.TrackerBoosting_create()
            self.t2 = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.t1 = cv2.TrackerMIL_create() 
            self.t2 = cv2.TrackerMIL_create() 
        if tracker_type == 'KCF':
            self.t1 = cv2.TrackerKCF_create() 
            self.t2 = cv2.TrackerKCF_create() 
        if tracker_type == 'TLD':
            self.t1 = cv2.legacy.TrackerTLD_create() 
            self.t2 = cv2.legacy.TrackerTLD_create() 
        if tracker_type == 'MEDIANFLOW':
            self.t1 = cv2.legacy.TrackerMedianFlow_create() 
            self.t2 = cv2.legacy.TrackerMedianFlow_create() 
        if tracker_type == 'GOTURN':
            self.t1 = cv2.TrackerGOTURN_create()
            self.t2 = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.t1 = cv2.legacy.TrackerMOSSE_create()
            self.t2 = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.t1 = cv2.TrackerCSRT_create()
            self.t2 = cv2.TrackerCSRT_create()
        
        self.t1.init(init_im1, init_bbox1)
        self.t2.init(init_im2, init_bbox2)


    def tracker_update(self, new_im1, new_im2):
        
        success1, bbox1 = self.t1.update(new_im1)
        success2, bbox2 = self.t2.update(new_im2)
        if not success1:
            bbox1 = None
        if not success2:
            bbox2 = None
            
        return bbox1, bbox2
