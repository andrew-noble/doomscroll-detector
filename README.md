# Doomscrolling Detector

My submission for HackCMU 2025. I built this project over 24 hours on September 12-13, 2025. Super fun!

Doomscroll Detector is a computer vision pipeline that helps you reclaim your attention. It recognizes when you’re stuck in endless scrolling, and provides actionable feedback — from gentle reminders to self-imposed monetary penalties. A simple way to break the cycle and build a healthier relationship with your phone.

![Demo GIF](media/best_clip.gif)

Anecdotally, I don't find myself scrolling away the hours while on my feet, or even seated in a chair. It mostly occurs when I am reclined. As such, **the system flags when the person in frame had a phone in their hand AND is reclined**. As you can see in the above GIF, the system doesn't flag when:

- subject is sitting up using their phone, or
- reclined without a phone

The system is divided into 3 parts, a vision pipeline, a webserver, and a simple dashboard.

## Running the Project

I built and ran this project on ordinary, everyday hardware, my Framework 13 laptop and its built-in webcam, and you can too! To run it:

```bash
# Clone the repository
git clone https://github.com/andrew-noble/doomscroll-detector
cd doomscroll-detector

# Set up the computer vision pipeline
cd cv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up the API server
cd ../api
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the vision pipeline (in one terminal)
cd ../cv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py

# Run the API server (in another terminal)
cd ../api
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py

# Open the dashboard
# Navigate to localhost:8000 in your browser
```

## Components

- **Vision Pipeline**

  - Uses OpenCV and YOLO v11 model from ultralytics to determine whether someone is both reclined and has a phone in their hand

- **Web Server**

  - Simple FastAPI server that reads outputs from the pipeline, determines the proper consequences for scrolling behavior, and serves data to be displayed to the dashboard

- **Dashboard**
  - Static dashboard that displays scrolling stats, including consequences. The current consequence is charging a credit card $0.50/second via Stripe

![Dashboard](/media/dashboard.png)

## File Structure

```
.
├── api/
│   ├── main.py
│   └── requirements.txt
├── cv/
│   ├── doomscroll_recording.mp4
│   ├── draw_frame.py
│   ├── main.py
│   ├── opts.py
│   ├── recording.mp4
│   ├── requirements.txt
│   ├── vision.py
│   ├── yolo11n-pose.pt
│   └── yolo11n.pt
├── README.md
└── web/
    ├── index.html
    └── style.css
```
