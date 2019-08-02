from Database import Database
from processor.DetectionProcessor import DetectionProcessor


class MainProcessor:

    def __init__(self, camera_id):
        self.cam_ids = camera_id
        #CamAinfo = Database.getInstance().getCaminfo(camera_id[0])
        #CamBinfo = Database.getInstance().getCaminfo(camera_id[1])
        #CamCinfo = Database.getInstance().getCaminfo(camera_id[2])

        self.processor = DetectionProcessor()


    def getProcessedImage(self,yolo,frames=None):
        dicti = self.processor.object_detection(frames,yolo)
        return dicti