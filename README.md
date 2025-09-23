# Doomscrolling Detector

I built this computer vision pipeline because I hate how much I scroll on my phone. My precious life seems to just drain away.

![Demo GIF](media/clip1.gif)

Anecdotally, I don't find myself scrolling away the hours while on my feet, or even seated in a chair. It mostly occurs when I am reclined.

The system flags doomscrolling when the subject has a phone in their hand **and** is reclined.

It would be easy to add in Stripe to charge your credit card, so I put in a spoofed penalty counter.

I built this for HackCMU 2025 over 24 hours on September 12-13, 2025 and had a blast.

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
