#Programme core
model_name = 'hand_model.sav'

# custom function
def rps(num):
  if num == 0: return 'PAPER'
  elif num == 1: return 'ROCK'
  else: return 'SCISSOR'

# just importing shit
from cv2 import cv2
import hand_detection_module
from data_generation import num_hand
import pickle
from id_distance import calc_all_distance

font = cv2.FONT_HERSHEY_PLAIN
hands = hand_detection_module.HandDetector(max_hands=num_hand)
model = pickle.load(open(model_name,'rb'))
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image, my_list = hands.find_hand_landmarks(cv2.flip(frame, 1),
                                             draw_landmarks=False)
  if my_list:
    height, width, _ = image.shape
    all_distance = calc_all_distance(height,width, my_list)
    pred = rps(model.predict([all_distance])[0])
    pos = (int(my_list[12][0]*height), int(my_list[12][1]*width))
    image = cv2.putText(image,pred,pos,font,2,(0,0,0),2)

  cv2.imshow('Hands', image)
  cv2.waitKey(1)
