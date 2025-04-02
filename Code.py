import cv2
import os
import numpy as np
from ultralytics import YOLO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load YOLOv8 model
model_path = r"C:\Users\hp\runs\detect\train5\weights\best.pt"  # model's path
yolo_model = YOLO(model_path)

# Define input image path
input_image_path = r"C:\Users\hp\OneDrive\Desktop\Minor Project\Sentinel\Img 7.jpg"  # image path

# Real-world conversion factor (optional)
real_world_conversion_factor = 1  # Set to the pixel resolution (e.g., meters per pixel)

farming_cost_per_pixel = 0.1  # Example cost per pixel for farming

print("Processing . . .")

# Function to calculate NDVI
def calculate_ndvi(image, x1, y1, x2, y2):
    """Calculate NDVI for a bounding box area."""
    h, w, _ = image.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    NIR = cropped[:, :, 0].astype(float)
    Red = cropped[:, :, 2].astype(float)

    denominator = NIR + Red
    denominator[denominator == 0] = 1e-6
    NDVI = (NIR - Red) / denominator

    return NDVI.mean()

# Process the input image
image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print(f"Error: Could not load image at {input_image_path}")
else:
    results = yolo_model(image, conf=0.25)  # Confidence threshold if needed
    total_area_pixels = image.shape[0] * image.shape[1]
    diseased_area_pixels = 0

    print(f"\nProcessing: {input_image_path}")
    for idx, result in enumerate(results[0].boxes, start=1):
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        confidence = result.conf[0].item()
        label = int(result.cls[0].item())

        mean_ndvi = calculate_ndvi(image, x1, y1, x2, y2)
        if mean_ndvi is None:
            continue

        box_area_pixels = (x2 - x1) * (y2 - y1)
        if mean_ndvi < 0.3:  # Diseased areas
            diseased_area_pixels += box_area_pixels

            # Bounding box around diseased areas
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for diseased areas

            # Adding "Disease" label to bounding box
            label_text = "Disease"
        
            # Text position and font size for better visibility
            font_scale = 0.7  # Slightly larger font scale
            font_color = (0, 0, 0)  # White color 
            font_thickness = 2

            # Position label at the bottom-left corner, 10px above the box
            label_position = (x1, y2 - 10)  # Adjusted to bottom-left

            # A filled background rectangle behind the text for better contrast
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_w, text_h = text_size

            # A red rectangle around the label
            cv2.rectangle(image, (x1 - 5, y2 - text_h - 10), (x1 + text_w + 5, y2), (0, 0, 255), 2)  # Red rectangle around label

            # Put text with adjusted font size and color
            cv2.putText(image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        print(f"  Object {idx}:")
        print(f"    - Label: {yolo_model.names[label]}")
        print(f"    - Confidence: {confidence:.2f}")
        print(f"    - NDVI: {mean_ndvi:.2f}")
        print(f"    - Area: {box_area_pixels} pixels²")

    safe_area_pixels = total_area_pixels - diseased_area_pixels

    print("\nArea Summary:")
    print(f"  Total Area: {total_area_pixels} px sq.")
    print(f"  Diseased Area: {diseased_area_pixels} px sq.")
    print(f"  Safe Area: {safe_area_pixels} px sq.")

    # Loading dataset
    file_path = r"C:\Users\hp\OneDrive\Desktop\Minor Project\Sentinel\Sentinel-2 L2A-3_NDVI - 7.csv"
    data = pd.read_csv(file_path)

    # Preprocessing
    data['C0/date'] = pd.to_datetime(data['C0/date'])

    features = [
        'C0/min', 'C0/max', 'C0/mean', 'C0/stDev', 'C0/sampleCount',
        'C0/noDataCount', 'C0/median', 'C0/p10', 'C0/p90'
    ]

    X = data[features]
    np.random.seed(42)
    data['CropYield'] = 50 + (X['C0/mean'] * 20) + np.random.normal(0, 5, len(X))
    y = data['CropYield']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    #r2 = abs(r2_score(y_test, y_pred))

    print(f"\nModel Evaluation:")
    print(f"  Mean Squared Error: {mse:.2f}")
    #print(f"  R^2 Score: {r2:.2f}")

    # Yield and cost analysis
    total_yield = (total_area_pixels / total_area_pixels) * model.predict(X.mean().to_frame().T)[0]
    undiseased_yield = (safe_area_pixels / total_area_pixels) * model.predict(X.mean().to_frame().T)[0]

    total_cost = total_area_pixels * farming_cost_per_pixel
    undiseased_cost = safe_area_pixels * farming_cost_per_pixel
    cost_savings = total_cost - undiseased_cost

    print("\nYield and Cost Analysis:")
    print(f"  Total Yield (Full Area): {total_yield:.2f}")
    print(f"  Yield from Undiseased Area: {undiseased_yield:.2f}")
    print(f"  Total Cost (Full Area): ${total_cost:.2f}")
    print(f"  Cost of Undiseased Area: ${undiseased_cost:.2f}")
    print(f"  Cost Savings: ${cost_savings:.2f}")

    # Visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_total_area = f"Total Area: {total_area_pixels} px sq."
    text_diseased_area = f"Diseased Area: {diseased_area_pixels} px sq."
    text_safe_area = f"Safe Area: {safe_area_pixels} px sq."
    text_yield_comparison = f"Total Yield: {total_yield:.2f}, Undiseased Yield: {undiseased_yield:.2f}"
    text_cost_comparison = f"Cost Savings: ${cost_savings:.2f}"

    cv2.putText(image, text_total_area, (10, 30), font, font_scale, (0, 128, 255), 2, cv2.LINE_AA)  # Orange
    cv2.putText(image, text_diseased_area, (10, 60), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)  # Red
    cv2.putText(image, text_safe_area, (10, 90), font, font_scale, (255, 0, 0), 2, cv2.LINE_AA)  # Blue
    cv2.putText(image, text_yield_comparison, (10, 120), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)  # Black
    cv2.putText(image, text_cost_comparison, (10, 150), font, font_scale, (128, 0, 255), 2, cv2.LINE_AA)  # Purple
    
    # Displaying the image using matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("YOLO Detection and Yield Analysis")
    plt.show()
