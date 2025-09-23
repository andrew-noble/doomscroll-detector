# Doomscrolling Detector

A computer vision pipeline that detects when you're reclined and on your phone. I built this because I hate how much I scroll on my phone. My precious life seems to just drain away.

![Demo GIF](media/clip1.gif)

Anecdotally, I don't find myself scrolling away the hours while on my feet, or even seated in a chair. It mostly occurs when I am laying in bed or on a couch.

The system flags doomscrolling when the subject has a phone in their hand **and** is reclined.

It would be easy to add in Stripe to charge your credit card, so I put in a spoofed penalty counter.

## How It Works
- Uses YOLOv11 pose estimation to track keypoints
- Detects phones with a YOLO object detector
- Applies heuristics:
  - Reclined posture → based on hip/shoulder alignment
  - Holding phone → wrist proximity to phone box
- Combines both → flags as doomscrolling
- Overlays results on the webcam feed + spoofed penalty counter

# Running the Project

I built and ran this project on ordinary, everyday hardware, my Framework 13 laptop and its built-in webcam, and you can too! To run it:

```bash
# Clone the repository
git clone https://github.com/andrew-noble/doomscroll-detector
cd doomscroll-detector

# Set up the computer vision pipeline
cd cv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt # this will be beefy with pytorch, etc

# Run the vision pipeline (in one terminal)
cd ../cv
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

Built at HackCMU 2025 in a 24-hour sprint.
