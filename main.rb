import cv2
import pyvirtualcam
import numpy as np

cam = cv2.VideoCapture(1)

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as vcam:
  while True:
      ret, frame = cam.read()

      if not ret:
        break

      # filter frame
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = cv2.Canny(frame, 100, 200)

      # play nice with pyvirtualcam
      frame = np.stack((frame,)*3, axis=-1)

      #cv2.imshow("frame", frame)

      vcam.send(frame)
      vcam.sleep_until_next_frame()

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cam.release()
cv2.destroyAllWindows()