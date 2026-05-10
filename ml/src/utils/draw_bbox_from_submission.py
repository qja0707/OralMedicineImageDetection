import random
import cv2
import matplotlib.pyplot as plt
import os

def draw_bbox_from_submission(test_dir:str, submission_path:str, sampling_count = 3):

    # Check if the model file exists
    if not os.path.exists(test_dir):
        print(f"Error: test_dir not found at {test_dir}")
        return
    
    if not os.path.exists(submission_path):
        print(f"Error: submission_path not found at {submission_path}")
        return
    

    # image sampling from test_dir
    images = os.listdir(test_dir)

    # random sampling

    for i in range(sampling_count):
        image_name = images[random.randint(0, len(images))]

        print(f"image_name:{image_name}")

        image_id = image_name.split(".")[0]
        
        boxes = []
        # read csv file
        with open(submission_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                datas = line.split(",")
                if image_id == datas[1]:
                    category_id, bbox_x, bbox_y, bbox_w, bbox_h, score = datas[2:]
                    boxes.append({"category_id":category_id, "bbox_x":bbox_x, "bbox_y":bbox_y, "bbox_w":bbox_w, "bbox_h":bbox_h, "score":score})

        print(f"boxes:{boxes}")

        # Load the image using OpenCV
        img = cv2.imread(os.path.join(test_dir, image_name))

        # Check if image was loaded successfully
        if img is None:
            print(f"Error: Could not load image from {image_id}")
            return

        # Convert image from BGR to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process results and draw bounding boxes
        for box in boxes:        
            # Get bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(box["bbox_x"]), int(box["bbox_y"]), int(box["bbox_x"]) + int(box["bbox_w"]), int(box["bbox_y"]) + int(box["bbox_h"])
            confidence = round(float(box["score"]),2)

            # Draw rectangle
            color = (255, 0, 0)  # Red color for bounding box
            thickness = 2
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)

            # Put confidence label
            label = f'{box["category_id"]}_{confidence}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            cv2.putText(img_rgb, label, (x1, y1 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # Display the image with matplotlib
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        plt.title(f'Detected Objects in {os.path.basename(image_id)}')
        plt.axis('off')
        plt.show()
