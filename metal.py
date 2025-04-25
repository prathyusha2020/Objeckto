from ultralytics import YOLO 
import cvzone
import cv2

# Initialize YOLO model
model = YOLO('yolov10n.pt')

# Open default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab a frame from the camera.")
        break

    # Pass the image through the model
    results = model(image)
    
    # Iterate over detections and draw bounding boxes with labels
    for info in results:
        parameters = info.boxes
        for box in parameters:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int') * 100
            class_detected_number = int(box.cls[0])
            
            # Get the original class name from the model's output
            class_detected_name = results[0].names[class_detected_number]
            
            # Override the class label if it is "tooth brush"
            if class_detected_name.lower() == "toothbrush":
                class_detected_name = "metal_object"
            if class_detected_name.lower() == "fork":
                class_detected_name = "metal_object"
            if class_detected_name.lower() == "spoon":
                class_detected_name = "metal_object"  
            if class_detected_name.lower() == "remote":
                class_detected_name = "metal_object" 
            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)
    
    # Display the processed frame
    cv2.imshow('frame', image)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
