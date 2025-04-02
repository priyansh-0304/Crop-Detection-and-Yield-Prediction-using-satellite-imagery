Project Description:
This project integrates YOLOv8 (You Only Look Once) object detection, NDVI (Normalized Difference Vegetation Index) analysis, and machine learning models to detect crop diseases, estimate affected areas, and predict potential yield losses. It also provides an economic analysis by estimating cost savings based on undiseased crop areas.

Key Components of the Project:
1. YOLOv8 for Crop Disease Detection
A pre-trained YOLOv8 model (best.pt) is used to detect crop diseases in aerial or satellite images.

The model processes an input image, detects diseased regions, and draws bounding boxes around them.

The total area and affected area are calculated to determine the extent of crop disease spread.

2. NDVI Calculation for Health Assessment
NDVI (Normalized Difference Vegetation Index) is calculated for detected regions to determine the severity of crop stress.

NDVI values below a certain threshold indicate diseased areas.

This helps differentiate healthy crops from affected ones.

3. Yield Prediction Using Machine Learning
A Random Forest Regressor is trained on NDVI and other features extracted from Sentinel-2 satellite data.

The model predicts crop yield based on the available dataset.

The predicted yield is used to compare the estimated production in both healthy and diseased areas.

4. Economic Cost Analysis
The farming cost per pixel is considered to estimate the total farming cost.

Cost savings are calculated by estimating the difference between managing the full area vs. focusing on the undiseased regions.

Project Workflow:
Load the YOLOv8 Model → Load the trained YOLO model to detect crop diseases.

Read and Process the Image → Load an input crop image and detect disease-affected areas.

Calculate NDVI for Disease Identification → Use NDVI to confirm diseased regions.

Predict Crop Yield → Train and test a Random Forest model using Sentinel-2 NDVI data.

Analyze the Cost and Yield Loss → Compare estimated yield and calculate financial loss due to disease.

Visualize the Results → Display the final image with bounding boxes and annotations.

Technologies Used:
YOLOv8: Object detection for identifying diseased crops.

OpenCV: Image processing and bounding box visualization.

NumPy & Pandas: Data manipulation and preprocessing.

Scikit-Learn: Machine learning for yield prediction.

Matplotlib: Visualization of results.

Joblib: Model saving and loading for efficient processing.

Applications of the Project:
✅ Precision Agriculture – Helps farmers make data-driven decisions for better crop management.
✅ Early Disease Detection – Identifies diseased regions early to prevent further spread.
✅ Yield Optimization – Predicts expected crop yield under different conditions.
✅ Cost-Effective Farming – Helps optimize farming costs by focusing on healthy crops.

This project is a step toward AI-driven smart agriculture, allowing farmers to maximize productivity while reducing losses.
