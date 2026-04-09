import os, sys, cv2
sys.path=[p for p in sys.path if p and os.path.abspath(p)!=os.path.abspath('d:/github/Drishya')]
from brisque import BRISQUE
cap=cv2.VideoCapture('testing_videos/video_large.mp4')
ret, frame=cap.read(); cap.release()
img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
obj=BRISQUE(url=False)
image=obj.remove_alpha_channel(img)
import skimage.color
gray=skimage.color.rgb2gray(image)
print('gray', gray.shape, gray.dtype)
mscn=obj.calculate_mscn_coefficients(gray,7,7/6)
print('mscn', type(mscn), mscn.shape, mscn.dtype)
cp=obj.calculate_pair_product_coefficients(mscn)
for k,v in cp.items():
    print(k, type(v), getattr(v,'shape', 'scalar'))
features=obj.calculate_brisque_features(gray,7,7/6)
print('features', features.shape, features.dtype)
print('first 10 types', [type(x) for x in features[:10]])
try:
    scaled=obj.scale_features(features)
    print('scaled', scaled)
except Exception as e:
    import traceback
    traceback.print_exc()
