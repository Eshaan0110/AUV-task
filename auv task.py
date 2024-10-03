import cv2
import numpy as np
import time

def enhance_image(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img_rgb)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    img_corrected = cv2.merge((b, g, r))
    lab = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 50))
    cl = clahe.apply(l_channel)
    lab_corrected = cv2.merge((cl, a_channel, b_channel))
    img_enhanced = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)
    return img_enhanced
    video_path = r'D:\Opencv\output_video.mp4'  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    enhanced_frame = enhance_image(frame)
    elapsed_time = time.time() - start_time
    current_fps = 1 / elapsed_time
    fps_text = f'FPS: {current_fps:.2f}'
    cv2.putText(enhanced_frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Enhanced Video', enhanced_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
