import cv2
import datetime
import time
import pyfirmata as pm
import numpy as np

board = pm.Arduino('COM6')
it = pm.util.Iterator(board)
it.start()
buzzer_pin = board.get_pin('d:3:o')
green_pin = board.get_pin('d:5:o')
red_pin = board.get_pin('d:4:o')

# Load pre-trained vehicle detection model
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Initialize video capture from webcam
cap = cv2.VideoCapture(1)

# Define boundary line coordinates
line_y = 300

# Initialize crossing count
crossing_count = 0

# Initialize traffic light durations
green_duration = 10 
red_duration = 5

# Initialize traffic light timer
traffic_light_timer = time.monotonic()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    buzzer_pin.write(0)
    
    # Get current time
    current_time = time.monotonic()
    elapsed_time = current_time - traffic_light_timer
    
    if elapsed_time < green_duration:
        green_pin.write(1)
        red_pin.write(0)
        time_remaining = green_duration - elapsed_time
    else:
        green_pin.write(0)
        red_pin.write(1)
        time_remaining = green_duration + red_duration - elapsed_time
        
        if elapsed_time >= green_duration + red_duration:
            traffic_light_timer = current_time
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if time_remaining <= red_duration:
        # Detect vehicles in the frame
        vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around the detected vehicles
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Check if car crosses the line
            if y + h > line_y and y < line_y:
                crossing_count += 1
                print('Car crossed the line! Total crossings:', crossing_count)
                buzzer_pin.write(1)
                
                # Capture a still image of the detected vehicle
                vehicle_img = frame[y:y+h, x:x+w]
                cv2.imwrite("vehicles.jpg", vehicle_img)
                cv2.imwrite("overview.png", frame)
    
    # Draw boundary line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # Add timestamp to the frame
    timestamp = str(datetime.datetime.now())
    cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Vehicle Detection', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

