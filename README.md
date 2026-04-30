# SafeGo AI

A safety-focused system that detects distress/help gestures from camera input and triggers alerts.

## 🚀 Overview
SafeGo AI aims to identify emergency hand gestures from camera feeds. When a distress gesture is detected, the system is designed to send alerts to authorities and emergency contacts.

## ✨ Features
- Gesture detection using computer vision
- Basic alert triggering logic
- Local data storage using SQLite
- Simple frontend interface

## 🛠️ Tech Stack
- Python
- OpenCV
- SQLite
- HTML/CSS

## 📁 Project Structure
safego_ai/
│── app.py  
│── detect.py  
│── index.html  
│── requirements.txt  
│── safego.db  
│── .gitignore  

## ⚙️ Setup Instructions

1. Clone the repository:
git clone https://github.com/ashivrma/safeGo_AI_project.git  
cd safeGo_AI_project  

2. Create virtual environment:
python -m venv venv  
venv\Scripts\activate  

3. Install dependencies:
pip install -r requirements.txt  

4. Run the project:
python app.py  

## 📌 Current Status
- Prototype stage  
- Gesture detection logic implemented (basic)  
- Alert system under development  

## 🔮 Future Scope
- Real-time camera integration  
- Alert to nearest police station  
- Emergency contact notification system  
- Improved gesture recognition using AI/ML  

## ⚠️ Disclaimer
This is a prototype and not yet connected to real emergency services.

## 👤 Author
Ashi 
