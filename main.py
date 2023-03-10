import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model

drawing = False
erasing = False

width = 20

def drawCallback(e, x, y, flags, params):
    global drawing
    global erasing
    global field

    match e: 
        case cv2.EVENT_LBUTTONDOWN:
            if not erasing:
                drawing = True
        case cv2.EVENT_LBUTTONUP:
            drawing = False
        #Оказывается, правая кнопка мыши уже занята вызовом контекстного меню(
        # case cv2.EVENT_RBUTTONDOWN:
        #     if not drawing:
        #         erasing = True
        # case cv2.EVENT_RBUTTONUP:
        #     erasing = False
        case cv2.EVENT_MOUSEMOVE:
            if drawing:
                field[y: y + width, x:x + width] = 255
            # elif erasing:
            #     field[y:y + width, x:x + width] = 0
    
field = np.zeros((520, 520), dtype='uint8')

cv2.namedWindow("Field", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Field", drawCallback)

model = load_model('mnist.h5')
while True:
    
    key = cv2.waitKey(1)
    
    if key == ord('i'):
        resized = cv2.resize(field, (28, 28))
        reshaped = np.reshape(resized, (1, 28, 28, 1))
        predict = model.predict(reshaped)
        answer = np.argmax(predict)

        print("answer: ", answer)
        
        for i in range(0, 10):
            print(i, predict[0][i] * 100)
    
    if key == ord('q'):
        break
    
    if key == ord('c'):
        field[:] = 0
    
    cv2.imshow("Field", field)

cv2.destroyAllWindows()
