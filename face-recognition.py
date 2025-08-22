import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


zoom_active = False
zoom_x, zoom_y, zoom_w, zoom_h = 0, 0, 0, 0


def mouse_callback(event, x, y, flags, param):
    global zoom_active, zoom_x, zoom_y, zoom_w, zoom_h, faces
    if event == cv2.EVENT_LBUTTONDOWN:  
        for (fx, fy, fw, fh) in faces:
            if fx <= x <= fx + fw and fy <= y <= fy + fh:  
  
                zoom_active = True
                zoom_x, zoom_y, zoom_w, zoom_h = fx, fy, fw, fh
                break
        else:
           
            zoom_active = False


cap = cv2.VideoCapture(0)


cv2.namedWindow('Face Detection')
cv2.setMouseCallback('Face Detection', mouse_callback)

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    

    for (x, y, w, h) in faces:
        padding = 20  
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        
        x = max(x, 0)
        y = max(y, 0)
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    
    if zoom_active:
 
        zoomed_face = frame[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
       
        zoomed_face = cv2.resize(zoomed_face, (zoom_w * 2, zoom_h * 2))  

        cv2.imshow('Zoomed Face', zoomed_face)
    else:

        if cv2.getWindowProperty('Zoomed Face', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Zoomed Face')
    

    cv2.imshow('Face Detection', frame)
    
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()