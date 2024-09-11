import cv2
import argparse
import numpy as np
import torch
import threading
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import time

# Argument parsing for YOLO model
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True, help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True, help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True, help='path to text file containing class names')
args = ap.parse_args()

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
# chuyển thành gpt2 
# Function to generate a more meaningful response
def generate_response(prompt):
    # If the input is too short or lacks enough context
    if len(prompt.strip()) < 5:
        return "Please provide a more detailed question."
    
    # Add context to guide the model away from repeating the input
    contextual_prompt = f"Human: {prompt}\nBot:"

    inputs = tokenizer.encode(contextual_prompt, return_tensors='pt')
    attention_mask = torch.ones_like(inputs)

    # Generate response
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=100,
        temperature=0.7,
        top_k=50,
        do_sample=True,  # Allow more diversity in the output
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the response from the model
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-processing to ensure that the response is clean
    return response.strip()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load YOLO model
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a threading event to signal stopping both threads
stop_event = threading.Event()

# Variables to store the latest detection results
latest_detection = {"class_name": None, "confidence": None, "bbox": None}

# Object Detection Function
def run_object_detection():
    global latest_detection
    while not stop_event.is_set():
        ret, image = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.2
        nms_threshold = 0.3

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        latest_detection["class_name"] = None
        latest_detection["confidence"] = None
        latest_detection["bbox"] = None

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            latest_detection["class_name"] = classes[class_ids[i]]
            latest_detection["confidence"] = confidences[i]
            latest_detection["bbox"] = (round(x), round(y), round(x + w), round(y + h))
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        cv2.imshow("Object Detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

# Function to get the latest line from the file
def get_latest_line(filename):
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as file:
        lines = file.readlines()
        if lines:
            latest_line = lines[-1].strip()  # Get the latest line
            if latest_line.startswith("You said: "):
                return latest_line.replace("You said: ", "", 1)  # Remove "You said: "
            return latest_line
    return None

# NLP Function to read from file and export responses to a text file
def run_nlp(filename):
    last_line = ""
    with open("responses.txt", "a") as response_file:  # Open file in append mode
        while not stop_event.is_set():
            user_query = get_latest_line(filename)
            if user_query and user_query != last_line:
                last_line = user_query
                if user_query.lower() == 'exit':
                    stop_event.set()  # Signal both threads to stop
                    break
                if "be my eye" in user_query.lower():
                    # Respond with object detection information
                    if latest_detection["class_name"]:
                        class_name = latest_detection["class_name"]
                        confidence = latest_detection["confidence"]
                        x, y, x_plus_w, y_plus_h = latest_detection["bbox"]
                        response = (f"Detected a {class_name} with a confidence of {confidence:.2f}. "
                                    f"The object is located at approximately ({x:.0f}, {y:.0f}) "
                                    f"with a width of {x_plus_w - x:.0f} pixels and a height of {y_plus_h - y:.0f} pixels.")
                    else:
                        response = "No objects detected."
                else:
                    response = generate_response(user_query)
                print(f"Bot: {response}")
                response_file.write(f"User: {user_query}\nBot: {response}\n\n")  # Write response to file
                #print("ty")
                response_file.flush()
            time.sleep(1)  # Check the file every second

# Specify the path to your file
filename = "temp.txt"

t1 = threading.Thread(target=run_object_detection)
t2 = threading.Thread(target=run_nlp, args=(filename,))

# Start both threads
t1.start()
t2.start()

# Wait for both threads to complete
t1.join()
t2.join()

# Release resources
cap.release()
cv2.destroyAllWindows()
