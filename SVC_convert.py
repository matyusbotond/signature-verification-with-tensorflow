class SignatureSample:
    def __init__(self, isOriginal, image):
        self.isOriginal = isOriginal
        self.image = image

def read_image(path):
    import cv2
    import numpy as np
    
    image = cv2.imread(path)[0:180,0:600]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("cropped",gray_image)
    #cv2.waitKey(0)
    
    #print path + ", " + str(gray_image.flatten().shape)
    
    return SignatureSample("e" in path, gray_image.flatten());
    
def get_random_batch(datas, count):
    from random import randint
    
    signatures = []
        
    for c in range(0,count):
        i = randint(0,len(datas)-1)
        signatures.append(datas[i])
        
    return signatures
    
def createSignatureDataSet(signature_path):
    import numpy as np
    import glob
    
    image_files = glob.glob(signature_path + "*.png")
    
    signatures = []
    
    for image_path in image_files:
        signature = read_image(image_path)
        signatures.append(signature)

    return signatures
    
def getSignatureImagesAndLabels(signatures):
    import numpy as np

    images = []
    labels = []
    for signature in signatures:
        images.append(signature.image)
        if(signature.isOriginal):
            labels.append([1,0])
        else:
            labels.append([0,1])
        
    return np.array(images), np.array(labels)