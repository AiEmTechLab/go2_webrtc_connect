import cv2
import numpy as np

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# --- YOLO imports ---
from ultralytics import YOLO

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

def main():
    frame_queue = Queue()

    # Load YOLO once (choose a model size: n/s/m/l/x; n is fastest)
    # You can swap 'yolov8n.pt' with 'yolov8s.pt' for a bit more accuracy.
    model = YOLO('yolov8n.pt')
    # Optional: only detect 'person' class (ID 0) to speed up
    PERSON_CLASS_ID = 0

    # Choose a connection method (uncomment the correct one)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.124.22.85")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                # Connect to the device
                await conn.connect()

                # Switch video channel on and start receiving video frames
                conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    try:
        # Warmup (optional, helps avoid first-frame latency)
        _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        while True:
            if not frame_queue.empty():
                img = frame_queue.get()

                # --- YOLO inference (person only) ---
                # conf: confidence threshold; classes=[0] restricts to 'person'
                results = model.predict(
                    img,
                    conf=0.5,
                    classes=[PERSON_CLASS_ID],
                    verbose=False
                )

                # Draw detections
                # Option A: let Ultralytics render overlays for you
                # annotated = results[0].plot()  # returns BGR image with boxes/labels

                # Option B: manual drawing for full control
                annotated = img.copy()
                if len(results):
                    # A person has been detected
                    # we may be able to greet to greet someone
                    r0 = results[0]
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        for box in r0.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            conf = float(box.conf[0]) if box.conf is not None else 0.0
                            label = f"person {conf:.2f}"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated, label, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                            )

                # Display the frame
                cv2.imshow('Video', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
