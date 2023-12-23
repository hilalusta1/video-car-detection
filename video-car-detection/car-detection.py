import cv2
import numpy as np

def main():
    video_source = "videoyolunuz"
    video_capture = cv2.VideoCapture(video_source)

    #modelin dosyalarını tanımlama:
    yolo_config = "yolov3.cfg"
    yolo_weights = "yolov3.weights"

    #modeli yükleme:
    net = cv2.dnn.readNet(yolo_config, yolo_weights)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        #görüntünün yükseklik ve genişlik bilgisi:
        height, width = frame.shape[:2]

        #elde edilen görüntünün yolo için uygun formata getirilmesi:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        #en yüksek güvenilirlik ve algılama tanımlamaları:
        max_confidence = 0
        best_detection = None

        #algılamaların işlenmesi:
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                #belirlediğimiz güvenilirliği aşan nesnenin seçilmesi:
                if confidence > 0.9 and confidence > max_confidence:
                    max_confidence = confidence
                    best_detection = obj

        #yüksek güvenilirliğe sahip nesnenin detect edilmesi:
        if best_detection is not None:
            class_id = np.argmax(best_detection[5:])
            center_x = int(best_detection[0] * width)
            center_y = int(best_detection[1] * height)
            w = int(best_detection[2] * width)
            h = int(best_detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            #bounding box çizimi:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Class: {class_id}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or k == ord('Q'):
            exit()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
