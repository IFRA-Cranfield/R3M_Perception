import cv2

cap = cv2.VideoCapture(4)



while True:

    ret, inputIMG = cap.read()
    if inputIMG is not None:
        cv2.imshow("CAMERA OUTPUT", inputIMG)
    else:
        break

    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyWindow("CAMERA OUTPUT")
        break
