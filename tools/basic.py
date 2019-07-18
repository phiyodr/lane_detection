import cv2

def cv2_imread(path):
    """Read image using cv2 as RGB (not BGR)."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)