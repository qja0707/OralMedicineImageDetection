from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def detect_and_draw_bboxes(image_path, model_path, is_draw = True):
    """
    Detects objects in an image using the trained YOLO11L single-class detector
    and draws bounding boxes on the detected objects.

    Args:
        image_path (str): The path to the input image.
    """
    # Define the path to the best.pt model based on the user's output
    # model_path = '/content/OralMedicineImageDetection/ml/outputs/checkpoints/yolo11l_kaggle_single_class_detector/weights/best.pt'

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        return

    # Load the YOLO model
    model = YOLO(model_path)

    # Perform inference on the image
    results = model(image_path)

    if not is_draw:
        return results


    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert image from BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process results and draw bounding boxes
    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(box.conf[0].item(), 2)

            # Draw rectangle
            color = (255, 0, 0)  # Red color for bounding box
            thickness = 2
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)

            # Put confidence label
            label = f'{confidence}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            cv2.putText(img_rgb, label, (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Display the image with matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.title(f'Detected Objects in {os.path.basename(image_path)}')
    plt.axis('off')
    plt.show()

    return results

# Example usage (you would replace this with your actual image path)
# detect_and_draw_bboxes('/content/OralMedicineImageDetection/ml/data/raw/images/some_image.jpg')