# Doomscrolling Detector

Doomscroll Detector is a computer vision pipeline that helps you reclaim your attention. It recognizes when you’re stuck in endless scrolling, and provides actionable feedback — from gentle reminders to self-imposed penalties. A simple way to break the cycle and build a healthier relationship with your phone.

The basic theory is that most long, undesirable sessions scrolling social media happen when someone is using their phone while also reclined. It's anecdotal,
but I don't personally find myself scrolling away the hours while on my feet, or even seated in a chair, its mostly when I am reclined. As such, the system flags when the person in frame had a phone in their hand AND is reclined. If they're sitting upright with their phone, it doesn't fire, and obviously if they're just
reclined without a phone, it also doesn't fire. Here is a short clip illustrating this:

![Demo GIF](media/best_clip.gif)

This computer vision data is then sent to a webserver that calculates statistics about your doomscrolling habits, and also levies penalties. A simple web dashboard displays these stats:

![Dashboard](/media/dashboard.png)

The system is divided into 3 parts, a vision pipeline, a webserver, and a simple dashboard.

## Components

- **Vision Pipeline**

  - Uses OpenCV and YOLO v11 model from ultralytics to determine whether someone is both reclined and has a phone in their hand

- **Web Server**

  - Simple FastAPI server that reads outputs from the pipeline, determines the proper consequences for scrolling behavior, and serves data to be displayed to the dashboard

- **Dashboard**
  - Static dashboard that displays scrolling stats, including consequences. The current consequence is charging a credit card $0.50/second via Stripe

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
