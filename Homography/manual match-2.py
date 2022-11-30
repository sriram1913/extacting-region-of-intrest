import cv2

i=1;

def click_event(event, x, y,flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global i

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.circle(img, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.putText(img, str(x) + ',' +
                    str(y)+'('+str(i)+')', (x, y), font,
                    1, (0, 255, 0), 2)
        cv2.imshow('image', img)
        i +=1;




if __name__ == "__main__":

    img = cv2.imread("left0.png", cv2.IMREAD_GRAYSCALE)  # Image to be aligned.0
    img2 = cv2.imread("right0.png", cv2.IMREAD_GRAYSCALE)  # Reference image.


    cv2.imshow('image1', img)

    cv2.setMouseCallback('image1', click_event)

    cv2.waitKey(0)

    cv2.destroyAllWindows()




