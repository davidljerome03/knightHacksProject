import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import gc  # Garbage collection

# Load a faster model (SSD MobileNet)
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'fork', 'spoon', 'knife', 'remote'
]

# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

# Reduce the resolution of the video (lower resolution for better performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_objects(frame, model):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        predictions = model(img)
    return predictions[0] if predictions else None

def draw_boxes(frame, predictions):
    if predictions is None:
        return

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5 and label < len(COCO_INSTANCE_CATEGORY_NAMES):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 10th frame (adjust this value as needed)
    if frame_count % 10 == 0:
        predictions = detect_objects(frame, model)
        draw_boxes(frame, predictions)

        # Explicitly call garbage collection to free memory
        gc.collect()

        # Clear GPU memory cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
