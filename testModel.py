from keras.models import load_model
import numpy as np
from PIL import Image

# dimensions of our images.
img_width, img_height = 150, 150

def classify(picPath):
    my_model = load_model('savedModel/model.h5')
    image = Image.open(picPath)
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array / 255.0, axis=0)
    # img_array = img_array / 255.0
    # print(img_array)
    predict = my_model.predict_classes(img_array)
    return predict[0]

# my_model = load_model('model.h5')
# image = Image.open('data/validation/HandSink/62.png')
# image = image.resize((150, 150))
# img_array = np.array(image)
# img_array = np.expand_dims(img_array/ 255.0, axis=0)
# # img_array = img_array / 255.0
# #print(img_array)
# predict = my_model.predict_classes(img_array)
# print(predict)
if __name__ == '__main__':
    aa = classify('data/validation/HandSink/62.png')
    print(aa)