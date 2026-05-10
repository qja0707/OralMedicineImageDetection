import os
import cv2
from ml.src.utils.detect_and_draw_bboxes import detect_and_draw_bboxes
from ultralytics import YOLO
import csv

def refine_box_results(box_results:list):
    box_list = []

    count = 0
    for r in box_results:
        for box in r.boxes:
            if count > 4:
                break;
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(box.conf[0].item(), 2)

            do_not_add = False

            for in_list_box in box_list:
                p_x1 = in_list_box["bbox_x"]
                p_y1 = in_list_box["bbox_y"]
                p_x2 = p_x1 + in_list_box["bbox_w"]
                p_y2 = p_y1 + in_list_box["bbox_h"]

                if x1 <= p_x1 and x2 >= p_x2 and y1 <= p_y1 and y2 >= p_y2:
                    do_not_add = True
                    break

            if do_not_add:
                continue


            box_list.append({"bbox_x": x1, "bbox_y": y1, "bbox_w": x2-x1, "bbox_h": y2-y1, "score": confidence})
            count += 1

            # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, confidence: {confidence}")

    return box_list

def crop_image(image_path:str, box_coord):
    image = cv2.imread(image_path)
    x1, y1, w, h = box_coord

    #crop image
    cropped_image = image[y1:y1+h, x1:x1+w]

    return cropped_image

def classify_box(image, model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load the YOLO model
    model = YOLO(model_path)

    classify_results = model(image)

    for c in classify_results:
        # print("=============")
        # print(c.names)
        # print(c.probs)
        top1 = c.probs.top1
        top1_name = c.names[top1]
        top1_conf = round(float(c.probs.top1conf),4)

        # print(f"top1: {top1}, top1_name: {top1_name}, top1_conf: {top1_conf}")


    return {"category_id":top1_name, "score":top1_conf}


def double_step_inference(detect_model_path:str, classify_model_path:str, image_path:str, output_path:str):
    print(f"model path:{detect_model_path}")
    print(f"image path:{image_path}")

    images = os.listdir(image_path)

    total_result_list = []

    for image in images:
        image_id = image.split(".")[0]
        
        first_image = os.path.join(image_path, f"{image_id}.png")
        # print(f"first image:{first_image}")

        box_result = detect_and_draw_bboxes(first_image, detect_model_path, is_draw = False)

        box_list = refine_box_results(box_result)

        # print(f"box_list:{box_list}")

        for box in box_list:
            cropped_image = crop_image(first_image, (box["bbox_x"], box["bbox_y"], box["bbox_w"], box["bbox_h"]))

            clssified = classify_box(cropped_image, classify_model_path)
            # print(f"classify cropped_image:{clssified}")

            box["category_id"] = clssified["category_id"]
            box["score"] = box["score"] * clssified["score"]
            box["image_id"] = image_id

        # print(f"final result:{box_list}")

        total_result_list.extend(box_list)

    # make final csv
    fieldnames = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    annotation_id = 1

    for result in total_result_list:
        result["annotation_id"] = annotation_id
        annotation_id += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(total_result_list)




if __name__ == "__main__":
    double_step_inference()