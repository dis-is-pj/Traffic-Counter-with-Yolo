import cv2 as cv
import cv2
import numpy as np
import time


if __name__ == '__main__':
    name = 'obj.names'
    cfg = 'yolo-obj.cfg'
    weights = 'yolo-obj_best.weights'

    labels = open(name).read().strip().split("\n")

    net = cv2.dnn.readNetFromDarknet(cfg, weights)

    image = input('Enter the path of image : ')
    
    img = cv2.imread(image)
    h,w,c = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)

    net.setInput(blob)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    conf = []
    classIDs = []
    c = 0.3

    for op in layerOutputs:
        for det in op:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > c:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = det[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                conf.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, conf, c,0.3)

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),dtype="uint8")

    count = {'t':0,'f':0}

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
            text = "{}: {:.4f}".format(labels[classIDs[i]], conf[i])
            if classIDs[i] == 0:
                count['t'] += 1
            else:
                count['f'] += 1
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

    print('Detected ',count['t'],' two wheelers and ',count['f'],' four wheelers.')

    if(cv2.imwrite('predictions.jpg',img)):
        print('Results saved in predictions.jpg')


