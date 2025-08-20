import cv2
import asyncio
import threading
import time
from queue import Queue

import numpy as np
from ultralytics import YOLO                                # pip install ultralytics :contentReference[oaicite:1]{index=1}
from aiortc.contrib.media import MediaPlayer
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

# —————————— SETUP ——————————

# Load a pretrained YOLOv5s model
print("Setting up YOLO")
#model = YOLO("yolov5s.pt")                                  # Predict on streams :contentReference[oaicite:2]{index=2}
model = YOLO("yolov5su.pt")

print("YOLO is setup")
# Connect to Go2 (video recv + audio send)
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.124.22.85")
# (or supply serialNumber / credentials as in examples) :contentReference[oaicite:3]{index=3}

async def setup_connection():
    await conn.connect()
    conn.video.switchVideoChannel(True)
    conn.video.add_track_callback(recv_camera_stream)

# Thread to run asyncio
def start_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(setup_connection())
    loop.run_forever()

# Queue & async callback for incoming frames
frame_queue = Queue()
async def recv_camera_stream(track):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)

# —————————— GREETING ACTIONS ——————————

# Play the MP3 once, via the Go2’s audio channel
def play_greeting_sound():
    mp3_path = "welcome.mp3"
    player = MediaPlayer(mp3_path)
    audio = player.audio
    conn.pc.addTrack(audio)

# Placeholder for your “wave” gesture; replace with the real Go2 SDK call:
def perform_greeting_gesture():
    # e.g. conn.action.play("wave") or conn.motion.play_action(ActionID.WAVE)
    pass

# —————————— MAIN ——————————

def main():
    # Start WebRTC in background
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_asyncio_loop, args=(loop,), daemon=True)
    t.start()

    greeted = False

    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                # Run YOLO inference
                results = model(img)[0]
                # Check for any “person” detections (class 0 in COCO)
                person_detected = any(box.cls == 0 for box in results.boxes)
                
                if person_detected and not greeted:
                    greeted = True
                    play_greeting_sound()
                    perform_greeting_gesture()
                elif not person_detected:
                    greeted = False

                # Show for debugging
                cv2.imshow("Go2 Video", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        t.join()

if __name__ == "__main__":
    main()

