import os
import tensorflow as tf
import cv2
import numpy as np

model = tf.saved_model.load('/home/sort/ved/MaskED/pomegranate/march6_ckpt3_exp')
infer = model.signatures["serving_default"]

imgDir_path = "/home/sort/ved/MaskED/pomegranate/val_binned/val"
img_list = os.listdir(imgDir_path)
for imgPath in img_list:
    img = cv2.imread( os.path.join(imgDir_path, imgPath) )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,512)).astype(np.float32)
    output = infer(tf.constant(img[None, ...]))

    _h = img.shape[0]
    _w = img.shape[1]

    det_num = output['num_detections'][0].numpy()
    det_boxes = output['detection_boxes'][0][:det_num]
    det_boxes = det_boxes.numpy()*np.array([_h,_w,_h,_w])
    det_masks = output['detection_masks'][0][:det_num].numpy()

    det_scores = output['detection_scores'][0][:det_num].numpy()
    det_classes = output['detection_classes'][0][:det_num].numpy()

    for i in range(det_num):
        score = det_scores[i]
        if score > 0.05:
            box = det_boxes[i].astype(int)
            _class = det_classes[i]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            cv2.putText(img, str(_class)+'; '+str(round(score,2)), (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), lineType=cv2.LINE_AA)
            mask = det_masks[i]
            mask = cv2.resize(mask, (_w, _h))
            mask = (mask > 0.5)
            roi = img[mask]
            blended = roi.astype("uint8")
            # img[mask] = blended*[0,0,1]

    cv2.imwrite(imgDir_path+"/out_"+imgPath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(imgDir_path+"/out_"+imgPath)
    # break