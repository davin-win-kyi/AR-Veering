# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai

import multiprocessing as mp
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

import soundfile   as sf      # pip install soundfile

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""
Relevant hl2ss imports
"""
from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import torch
import time
import cv2
import numpy as np
from PIL import Image

# System related imports
import random
import time

# AR_Veering system imports
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from google import genai
from io import BytesIO
from google.genai import types

# for async calls
import asyncio

# model imports
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO


import sounddevice as sd


# Enviornment variable and class variable setup
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\davin\PycharmProjects\Depth_Testing\google_cloud.json'

# HoloLens Address
host = "192.168.1.8"

# Ports
ports = [
    hl2ss.StreamPort.PERSONAL_VIDEO,
    ]

# PV parameters
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# image size
image_size = (640, 480)

# Maximum number of frames in buffer
buffer_elements = 150

# Global flags (these are fine as is)
cross_boolean = 0
button_boolean = False

# Threshold constants
CROSSWALK_NEAR_THRESHOLD_FACTOR = 0.25  # e.g., 25% of crosswalk width for "near" deviation
CROSSWALK_FAR_THRESHOLD_FACTOR = 0.50   # e.g., 50% of crosswalk width for "far" deviation

###################################################################
# Gemini Client initialization
###################################################################
_client = None
if os.getenv("GOOGLE_API_KEY"):
    _client = genai.Client(
        api_key="<API_KEY>",
        http_options={"api_version": "v1alpha"}, # Specify v1alpha for experimental models
    )
else:
    # Fallback to hardcoded key if env var not set, but strongly discourage this in production
    if "<API_KEY>": # Replace with your actual key if not using env var
        _client = genai.Client(
            api_key="<API_KEY>", # Your hardcoded key here
            http_options={"api_version": "v1alpha"},
        )
        print("WARNING: Using hardcoded API key. Consider setting GOOGLE_API_KEY environment variable.")
    else:
        print("CRITICAL ERROR: GOOGLE_API_KEY not set in environment or hardcoded. Gemini functions will not work.")



####################################################################################################
# GEMINI LIVE
####################################################################################################

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

os.environ["GOOGLE_API_KEY"] = "<API_KEY>"

MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {"response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.image = None

        self.text = ""

        # models to be used
        self.yolo_model = None

        self.pipe = None

        self.index = 0

        self.frame_ready = asyncio.Event()  

        self.turn_done   = asyncio.Event()   # set when Gemini sends turn_complete
        self.audio_done   = asyncio.Event()   # set when last audio chunk is played
    

    async def send_text(self):
            # Send text ---------------------------------------------------------------
        class command_buffer(hl2ss.umq_command_buffer):
            # Command structure
            # id:     u32 (4 bytes)
            # size:   u32 (4 bytes)
            # params: size bytes

            # Send string to Visual Studio debugger
            def debug_message(self, text):
                # Command id: 0xFFFFFFFE
                # Command params: string encoded as utf-8
                self.add(0xFFFFFFFE, text.encode('utf-8')) # Use the add method from hl2ss.umq_command_buffer to pack commands

        # See hl2ss_rus.py and the unity_sample scripts for more examples.


        text = "Only Notify me if there are the following important objects or say nothing: pedestrian curb, pedestrian traffic signal button. If there are important objects only response to " \
            "the user in the following format, say nothing if there is no important objects: " \
            "<important object> <direction>" \

            
        await self.session.send(input=text, end_of_turn=True)

        await asyncio.sleep(15)

        while True:

            await asyncio.sleep(5)

            print("Went into here")

            # Let me know if there are any of the following that are new if available:
            # - Objects and which direction they are coming from.
            # - Walk signal information and how much time is left.
            # - Let me know if I am to the left, right, or am I on the crosswalk.

            # here can get the text along with the prompt built up
            if self.image is not None:

                # YOLO inference
                with torch.inference_mode():
                    results = self.yolo_model(self.image)
                boxes = results[0].boxes.xyxy.tolist()
                classes = results[0].boxes.cls.tolist()
                names = results[0].names

                # Save Relevant Object information
                curbs = []
                curbs_to_detect = ["construction--barrier--curb", "object--manhole"]

                # Depth-Anything v2 inference
                rgb_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                with torch.inference_mode():
                    depth = self.pipe.infer_image(self.image)
                depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_image = np.array(depth)

                # YOLO object information
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    obj_name = names[classes[i]]
                    x1, y1, x2, y2 = map(int, bbox)
                    # Ensure coordinates are within depth_image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(depth_image.shape[1] - 1, x2)
                    y2 = min(depth_image.shape[0] - 1, y2)

                    # Calculate mean depth from the *original* depth_image (grayscale)
                    # Use a region for mean depth for better robustness
                    depth_roi = depth_image[y1:y2, x1:x2]
                    if depth_roi.size > 0:
                        mean_depth = np.mean(depth_roi)
                    else:
                        mean_depth = 0 # Default if bbox is too small or invalid

                    # Yolo Curb information
                    if obj_name in curbs_to_detect:
                        crop_box = (x1, y1, x2, y2)
                        cropped_image = img.crop(crop_box)
                        cropped_np = np.array(cropped_image)
                        # detect_orange_percentage expects BGR image, convert if PIL image is RGB
                        if cropped_np.shape[2] == 3 and cropped_np.dtype == np.uint8:
                            # Check if PIL image is RGB (default), then convert to BGR for cv2
                            if img.mode == 'RGB':
                                cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)

                        if mean_depth >= 90: # Consider if depth threshold is appropriate here
                            curbs.append(bbox)
                                
                            text = "Only Notify me if there are the following important objects or say nothing: pedestrian curb, pedestrian traffic signal button. If there are important objects only response to " \
                                "the user in the following format, say nothing if there is no important objects: " \
                                "<important object> <direction>" \
                                
                            await self.session.send(input=text, end_of_turn=True)

                            print("I went into here!")
                            break


    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # make sure it’s RGB

        # PIL.Image ➜ NumPy array (still RGB)
        frame_rgb = np.array(img_pil)

        # RGB ➜ BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        self.image = frame_bgr

        # print(type(img))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)

            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)


    #############################################################################################
    # NEW: HoloLens related Gemini Live API code
    #############################################################################################
    def _get_hololens_frame(self, sink, port):
        """
        Description: HoloLens Frame feeding method.
                     sending in frames to Gemini Live API
        """

        # Get frame information
        _, data = sink.get_most_recent_frame()
        if data is not None and hasattr(data.payload, 'image') and data.payload.image is not None and data.payload.image.size > 0:

            # Prepare frame information from HoloLens
            img = data.payload.image
            self.image = img
            # img is a numpy array (cv2), convert to PIL.Image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(img)
            pil_img.thumbnail([1024, 1024])
            image_io = io.BytesIO()
            pil_img.save(image_io, format="jpeg")
            image_io.seek(0)
            mime_type = "image/jpeg"
            image_bytes = image_io.read()
            return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        return None

    async def get_hololens_frames(self):
        # hl2ss setup
        host = "192.168.1.8"
        port = hl2ss.StreamPort.PERSONAL_VIDEO
        pv_width, pv_height, pv_framerate = 760, 428, 30
        buffer_elements = 10

        # Start subsystem if needed
        hl2ss_lnm.start_subsystem_pv(host, port)

        # Prepare hl2ss frame reading components
        producer = hl2ss_mp.producer()
        producer.configure(port, hl2ss_lnm.rx_pv(host, port, width=pv_width, height=pv_height, framerate=pv_framerate))
        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sink = consumer.create_sink(producer, port, manager, None)
        sink.get_attach_response()
        while (sink.get_buffered_frame(0)[0] != 0):
            pass

        # Start HoloLens frame reading task
        try:
            while True:
                frame = await asyncio.to_thread(self._get_hololens_frame, sink, port)
                if frame is not None:
                    await self.out_queue.put(frame)
                await asyncio.sleep(1.0)
        finally:
            sink.detach()
            producer.stop(port)
            hl2ss_lnm.stop_subsystem_pv(host, port)


    async def run(self):

        # Clear System Logs
        with open("system_log.txt", "w") as file:
            file.write("")
        with open("veering_log.txt", "w") as file: # Also clear veering log
            file.write("")
        with open("system_a_output.txt", "w") as file: # Also clear output log
            file.write("")
        with open("C:\\Users\\davin\\OneDrive\\Documents\\hl2ss\\viewer\\Depth-Anything-V2\\system_history.txt", 'w') as file:
            file.write("") # Clear system history log

        # YOLO Model
        checkpoint_file = "best.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_model = YOLO(checkpoint_file).to(device)

        # Depth Model
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits'
        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(torch.load(
            f'checkpoints/depth_anything_v2_{encoder}.pth', # Ensure this path is correct
            map_location=device,
            weights_only=True
        ))
        pipe = depth_model.to(device).eval()

        self.yolo_model = yolo_model
        self.pipe = pipe
        
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                # new HoloLens option added into system
                elif self.video_mode == "hololens":
                    tg.create_task(self.get_hololens_frames())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                await send_text_task
                raise asyncio.CancelledError("User requested exit")
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # new HoloLens Option added in
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none", "hololens"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
