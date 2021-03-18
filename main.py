import cv2 as cv
import numpy as np
import model

waight_path = 'model_waights/waight0020.h5'
n = 500; mrange = 2 ; srange= 1
normalize = lambda x, y: np.array([(x/n)*mrange - srange,
                                   (y/n)*mrange - srange ])
img = np.zeros((n,n))
dec = model.compile_auto_enco()
dec.load_weights(waight_path)
dec = dec.layers[1]

def decode(deco = dec, x = 0,y =0 ):
    point = normalize(x,y)[np.newaxis, ...]
    img = deco.predict(point)[0]
    print(point)
    return img[:,:,0]

def generate_img(event, x, y, flag, param):

    global img

    img = cv.resize(decode(x=x,y=y),(n,n))
    point = normalize(x, y)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, str(point),
               (x,y), font,.5, (0, 255, 0), 1, cv.LINE_AA)



winame = 'MNIST Latent Space'
cv.namedWindow(winame)
cv.setMouseCallback(winame, generate_img)

while True :
    cv.imshow(winame, img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cv.destroyWindow(winame)


