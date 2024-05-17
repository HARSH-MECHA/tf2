#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
from timeit import default_timer as timer

# Function to draw a rectangle with width > 1
def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color)

# Function to read labels from text files.
def read_label_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def inference_tf(runs, image, model, output, label=None):
    if label:
        labels = read_label_file(label)
    else:
        labels = None
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = input_details[0]['dtype'] == np.float32
    
    img = Image.open(image)
    draw = ImageDraw.Draw(img, 'RGBA')
    helvetica = ImageFont.truetype("./Helvetica.ttf", size=72)
        
    picture = cv2.imread(image)
    initial_h, initial_w, _ = picture.shape
    frame = cv2.resize(picture, (width, height))
    
    # Add N dim
    input_data = np.expand_dims(frame, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    print("Running inferencing for", runs, "times.")
    
    if runs == 1:
        start = timer()
        interpreter.invoke()
        end = timer()
        print('Elapsed time is', ((end - start) / runs) * 1000, 'ms')
    else:
        start = timer()
        print('Initial run, discarding.')
        interpreter.invoke()
        end = timer()
        print('First run time is', (end - start) * 1000, 'ms')
        start = timer()
        for i in range(runs):
            interpreter.invoke()
        end = timer()
        print('Elapsed time is', ((end - start) / runs) * 1000, 'ms')
    
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])
    
    for i in range(int(num_boxes)):
        top, left, bottom, right = detected_boxes[0][i]
        classId = int(detected_classes[0][i])
        score = detected_scores[0][i]
        if score > 0.5:
            xmin = left * initial_w
            ymin = bottom * initial_h
            xmax = right * initial_w
            ymax = top * initial_h
            if labels:
                print(labels[classId], 'score =', score)
            else:
                print('score =', score)
            box = [xmin, ymin, xmax, ymax]
            draw_rectangle(draw, box, (0, 128, 128, 20), width=5)
            if labels:
                draw.text((box[0] + 20, box[1] + 20), labels[classId], fill=(255, 255, 255, 20), font=helvetica)
    img.save(output)
    print('Saved to', output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--input', help='File path of the input image.', required=True)
    parser.add_argument('--output', help='File path of the output image.')
    parser.add_argument('--runs', help='Number of times to run the inference', type=int, default=1)
    args = parser.parse_args()
    
    output_file = args.output if args.output else 'out.jpg'
    label_file = args.label if args.label else None
    
    inference_tf(args.runs, args.input, args.model, output_file, label_file)

if __name__ == '__main__':
    main()
