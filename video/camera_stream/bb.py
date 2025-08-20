import cv2
import numpy as np
import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from ultralytics import YOLO

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

# --- AUDIO HUB (placeholder import, replace with your actual audio_hub module) ---
# from your_audio_module import audio_hub  

BARK_UUID = "feddb267-103d-4873-9758-2feecd8d176c"   # TODO: set your bark sound file UUID

def main():
    frame_queue = Queue()
    model = YOLO('yolov8n.pt')
    PERSON_CLASS_ID = 0
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="10.124.22.85")

    # Async function to receive video frames
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new asyncio event loop in another thread
    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    last_bark_time = 0
    bark_cooldown = 10  # seconds

    try:
        _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                results = model.predict(img, conf=0.5, classes=[PERSON_CLASS_ID], verbose=False)
                annotated = img.copy()

                if len(results):
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

                        # --- Trigger bark if cooldown passed ---
                        now = time.time()
                        if now - last_bark_time > bark_cooldown:
                            last_bark_time = now
                            asyncio.run_coroutine_threadsafe(
                                audio_hub.play_by_uuid(BARK_UUID), loop
                            )

                cv2.imshow('Video', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
