import os, sys, cv2
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('d:/github/Drishya')]
from brisque import BRISQUE
cap = cv2.VideoCapture('testing_videos/video_large.mp4')
ret, frame = cap.read()
cap.release()
print('frame', frame.shape, frame.dtype)
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
obj = BRISQUE(url=False)
try:
    s = obj.score(img)
    print('score', s)
except Exception as e:
    import traceback
    traceback.print_exc()
