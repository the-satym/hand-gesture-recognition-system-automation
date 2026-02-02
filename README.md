# Hand Gesture Recognition & System Automation

A Python-based project that recognizes hand gestures using computer vision and machine learning, then maps those gestures to real-time system automation actions (like mouse control, shortcuts, etc.).

The project follows a **data → model → automation** pipeline and is designed to be modular, extensible, and easy to experiment with.

---

##  Project Workflow

1. **Data Collection**
   - Capture hand landmarks using a webcam
   - Store structured gesture data into a CSV file

2. **Model Training**
   - Train a machine learning model on collected gesture data
   - Save the trained model and label encoder for reuse

3. **System Automation**
   - Load the trained model
   - Perform real-time gesture recognition
   - Trigger system-level actions based on recognized gestures

---

## 🧠 Technologies Used

### Data Collection
- OpenCV
- MediaPipe
- NumPy
- Pandas

### Model Training
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Joblib

### System Automation
- OpenCV
- NumPy
- Joblib
- PyAutoGUI
- Threading
- Collections

---

## How to run the project

1. **Install all the dependencies and required modules/libraries from requirement.txt**

1. **Run data-collector.py**
   - Record your hand gestures by following given instruction in terminal 
     
3. **Run model.py**
   - It will retrain the model with the newly added hand gestures 
5. **Run app.py**
   - Enjoy the final output by controling your system using hand gestures 




## 📂 Project Structure

```text
├── datacollector.py        # Collects hand landmark data using webcam
├── data-collector.csv     # Saved dataset of gestures
├── model.py               # Trains ML model on collected data
├── gesture_model.pkl      # Trained gesture recognition model
├── label_encoder.pkl      # Label encoder for gesture classes
├── app.py                 # Real-time gesture recognition & automation
├── README.md              # Project documentation







