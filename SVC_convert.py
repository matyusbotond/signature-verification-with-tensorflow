def read_image(path):
    import cv2
    import numpy as np
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.flatten(), [0,1];
    
def createSignatureDataSet(signature_path):
    import numpy as np
    import glob
    
    image_files = glob.glob(signature_path + "*.png")
    
    raw_data = []
    label = []
    
    for image_path in image_files:
        signature = read_image(image_path)
        raw_data.append(signature[0])
        label.append(signature[1])
    
    return raw_data, label
    
#print createSignatureDataSet("SVC/002/")
