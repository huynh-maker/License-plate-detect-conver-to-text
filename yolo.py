from ultralytics import YOLO
import cv2
import easyocr

# Load YOLO model (biển số)
model = YOLO("best.pt")     # <-- thay bằng đường dẫn model của bạn
reader = easyocr.Reader(['en'])   # OCR đọc ký tự tiếng Anh

# Mở video (0 nếu dùng webcam)
cap = cv2.VideoCapture("chrome_nj3serLs4Y.mp4")   # <-- thay video bạn muốn

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bằng YOLO
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            # Lấy bounding box
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Crop vùng biển số
            plate = frame[y1:y2, x1:x2]

            # OCR đọc ký tự
            try:
                text = reader.readtext(plate, detail=0)
                if len(text) > 0:
                    plate_text = text[0]
                    print("License Plate:", plate_text)
            except:
                pass

            # Vẽ bounding box lên video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "Plate", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    resized = cv2.resize(frame, (800, 600))
    cv2.imshow("License Plate Detection", resized)
    #cv2.imwrite("frame.jpg", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
