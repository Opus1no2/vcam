import cv2
import pyvirtualcam
import numpy as np
from effects import *

cam = cv2.VideoCapture(1)

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

if __name__ == "__main__":
    with pyvirtualcam.Camera(width=width, height=height, fps=fps) as vcam:
        while True:
            ret, frame = cam.read()

            if not ret:
                break

            # filter frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(frame, 100, 200)

            # play nice with pyvirtualcam
            frame = np.stack((frame,) * 3, axis=-1)

            frame = apply_block_shift_glitch(frame)
            frame = apply_color_shift_glitch(frame)
            frame = add_scanlines(frame, line_spacing=4, intensity=0.5)
            frame = color_jitter(frame, intensity=0.05)
            frame = add_noise(frame, noise_level=0.02)
            frame = pixel_sort_effect(frame, glitch_chance=0.02)

            cv2.imshow("frame", frame)

            vcam.send(frame)
            vcam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()
