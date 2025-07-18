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

#########################################################
# Gemini Request/Response methods
#########################################################
async def generate_content_with_image_live(prompt: str, image: Image.Image) -> str | None:
    """
    Visual question answering with Gemini flash for robust visual reasoning 

    Args:
        prompt (str): Text prompt for the model.
        image (PIL.Image.Image): PIL Image object to send.

    Returns:
        str: Generated text content, or None if an error occurs or client is not initialized.
    """
    # Assuming _client is initialized globally or passed in as an argument.
    if '_client' not in globals() or _client is None:
        print("Error: Gemini Client not initialized. API Key might be missing.")
        return None

    model_name = "gemini-2.0-flash"

    try:
        # Convert PIL Image to bytes
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        # Call the standard generate_content API
        response = _client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                types.Part(text=prompt)
            ]
        )

        return response.text

    except Exception as e:
        print(f"Error generating content with image via standard API: {e}")
        return None
    

async def get_gemini_live_response(user_message: str) -> str:
    """"
    Description: method requests and connects to Gemini Live API for 
                quick and immediate text responses

    Parameters: 
    - user_message: this is the prompt that the user will be sending to Gemini Live
    """
    if _client is None:
        print("Error: Gemini Client not initialized. Cannot get live response.")
        return "Error: API not configured."

    # Gemini request configuration
    model_id = "gemini-2.0-flash-live-001"
    config = {"response_modalities": ["TEXT"]}

    # Send prompt to Gemini and return response
    try:
        async with _client.aio.live.connect(model=model_id, config=config) as session:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": user_message}]}, turn_complete=True
            )

            full_response_text = ""

            async for response in session.receive():
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response_text += part.text
                if response.server_content and response.server_content.turn_complete:
                    break
            return full_response_text

    except Exception as e:
        print(f"An error occurred during get_gemini_live_response: {e}")
        return f"Error during API call: {e}"

#########################################################
# Color Detection methods
#########################################################
def detect_orange_percentage(image):
    """
    Detect the percentage of orange in a given image.

    Args:
        image (np.ndarray): BGR image.

    Returns:
        float: Percentage of orange pixels.
    """
    if image is None:
        return 0.0
    # Assuming image is already HSV if passed from calculate_color_percentage or similar.
    # If it's BGR, convert it first. Assuming BGR for now as per prior usage.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 20], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
    orange_pixels = cv2.countNonZero(mask_orange)
    total_pixels = image.shape[0] * image.shape[1]
    return (orange_pixels / total_pixels) * 100


#########################################################
# Object response methods
#########################################################
def get_object_response(image, mean_depth, object_name, x1, x2):
    """
    Determine if user should be notified about a given object or not

    Args:
        image (PIL.Image.Image): Image in which the object is detected.
        mean_depth (float): Average depth of the object.
        object_name (str): Detected object name.
        x1 (int): Left x-coordinate.
        x2 (int): Right x-coordinate.

    Returns:
        tuple or None: (object_name, direction, close_string) if close enough; else None.
    """

    # Depth threshold for object notifications
    depth_threshold = 70
    if mean_depth < depth_threshold:
        return None

    # Get relative position of object to user
    x_coordinate = (x1 + x2) // 2
    width, _ = image.size
    middle_x = width // 2
    x_diff = middle_x - x_coordinate
    percent_from_middle = abs(x_diff) / width

    if x_diff < 0 and percent_from_middle > 0.05:
        direction = "near-right" if percent_from_middle < 0.10 else "right"
    elif x_diff > 0 and percent_from_middle > 0.05:
        direction = "near-left" if percent_from_middle < 0.10 else "left"
    else:
        direction = "middle"

    # Return object position and closness relative to user
    close_string = "very close" if mean_depth >= 100 else "close by"
    return (object_name, direction, close_string)


def get_objects_message(object_map):
    """
    Build a summary message by grouping detected objects by direction.

    Args:
        object_map (dict): Maps object names to counts for each direction.

    Returns:
        str: A grouped and formatted message.
    """

    # Object mapping from label to name
    mapping = {
        "animal--bird": "bird",
        "animal--ground-animal": "animal",
        "construction--barrier--ambiguous": "barrier",
        "construction--barrier--curb": "curb",
        "construction--barrier--fence": "fence",
        "construction--barrier--guard-rail": "rail",
        "construction--barrier--other-barrier": "barrier",
        "construction--barrier--wall": "wall",
        "construction--flat--bike-lane": "bike lane",
        "construction--flat--crosswalk-plain": "crosswalk",
        "construction--flat--curb-cut": "curb",
        "construction--flat--sidewalk": "sidewalk",
        "human--person--individual": "person",
        "human--person--person-group": "group",
        "human--rider--bicyclist": "bicycler",
        "human--rider--motorcyclist": "motorcyclist",
        "human--rider--other-rider": "vehicle",
        "object--fire-hydrant": "hydrant",
        "object--mailbox": "mailbox",
        "object--manhole": "manhole",
        "object--parking-meter": "parking meter",
        "object--phone-booth": "phone booth",
        "object--pothole": "pothole",
        "object--street-light": "street light",
        "object--support--pole": "pole",
        "object--support--pole-group": "poles",
        "object--support--traffic-sign-frame": "traffic sign",
        "object--support--utility-pole": "pole",
        "object--traffic-cone": "cone",
        "object--trash-can": "trash can",
        "object--vehicle--bicycle": "bicycle",
        "object--vehicle--bus": "bus",
        "object--vehicle--car": "car",
        "object--vehicle--caravan": "van",
        "object--vehicle--motorcycle": "motorcycle",
        "object--vehicle--on-rails": "train",
        "object--vehicle--other-vehicle": "car",
        "object--vehicle--trailer": "trailer",
        "object--vehicle--truck": "truck",
        "object--vehicle--vehicle-group": "cars",
        "object--vehicle--wheeled-slow": "car",
        "void--ego-vehicle": "car"
    }

    # Direction mapping
    direction_map = {
        0: "left",
        1: "near left",
        2: "right",
        3: "near right",
        4: "front"
    }
    grouped_by_direction = {direction: [] for direction in direction_map.values()}

    # Get objects information
    # multiple objects will be plural
    for obj_name, counts in object_map.items():
        readable_name = mapping.get(obj_name, obj_name)
        for idx, count in enumerate(counts):
            if count > 0:
                pluralized = readable_name if count == 1 else readable_name + "s"
                grouped_by_direction[direction_map[idx]].append(pluralized)

    # Build object system response
    messages = []
    for direction, objects in grouped_by_direction.items():
        if objects:
            if len(objects) == 1:
                obj_str = objects[0]
            elif len(objects) == 2:
                obj_str = " and ".join(objects)
            else:
                obj_str = ", ".join(objects[:-1]) + ", and " + objects[-1]
            messages.append(f"{obj_str} {direction}".strip())

    if not messages:
        return ""

    return ". ".join(messages).strip() + "."


#########################################################
# Crosswalk Response methods
#########################################################
def get_crosswalk_message(image, crosswalks):
    """
    Determine the optimal crosswalk and provide a notification message based on the user's
    position relative to the crosswalk. Uses near and far thresholds.
    Also shows the closest crosswalk with OpenCV.

    Args:
        image (PIL.Image.Image): Image being analyzed.
        crosswalks (list): List of tuples (bounding_box, mean_depth).

    Returns:
        str: Notification about the crosswalk position.
    """

    # Get closest crosswalk
    crosswalk_max = None
    max_depth_crosswalk = 0
    for bbox, mean_depth in crosswalks:
        if mean_depth > max_depth_crosswalk:
            max_depth_crosswalk = mean_depth
            crosswalk_max = bbox
    if crosswalk_max is None:
        return ""

    x1, y1, x2, y2 = map(int, crosswalk_max)

    # Determine relative position to crosswalk
    x1, _, x2, _ = map(int, crosswalk_max)
    average_x = (x1 + x2) / 2.0
    width, _ = image.size
    center = width // 2
    distance_from_center = abs(average_x - center)
    crosswalk_width = (x2 - x1)
    near_threshold = crosswalk_width * CROSSWALK_NEAR_THRESHOLD_FACTOR
    far_threshold = crosswalk_width * CROSSWALK_FAR_THRESHOLD_FACTOR
    if distance_from_center >= far_threshold:
        # Adjusted logic for "far left" vs "far right"
        return "far right" if average_x < center else "far left"
    elif distance_from_center >= near_threshold:
        # Adjusted logic for "left" vs "right"
        return "right" if average_x < center else "left"
    else:
        return "On the crosswalk"


#########################################################
# Signal Response methods
#########################################################
async def get_signal_message(image: Image.Image, pedestrian_signals: list) -> str:
    """
    Determine the optimal pedestrian signal and generate a message using the Live API.

    Args:
        image (PIL.Image.Image): Image being analyzed.
        pedestrian_signals (list): List of tuples (bounding_box, mean_depth, object_name).

    Returns:
        str: Signal message or an empty string if none detected or an error occurs.
    """

    # Get closest signal
    pts_max = None
    max_depth = 0
    for bbox, mean_depth, _ in pedestrian_signals:
        if pts_max is None or mean_depth >= max_depth:
            max_depth = mean_depth
            pts_max = bbox

    signal_message = ""
    if pts_max is not None:

        # Signal information prompt
        user_prompt = (
            "Provide concise information about the pedestrian signal:\n"
            "- Indicate if the walking signal is only either red or white.\n"
            "- Include the countdown time (if available).\n"
            "- If the signal is black, return a blank output.\n"
            "- Formatted as (if time available): signal is <color> with <#> seconds left.\n"
            "- Formatted as (if time not available): signal is <color>. \n"
            "Example: signal is red with 7 seconds left."
        )

        # Get Gemini response for signal information
        crop_box = tuple(map(int, pts_max))
        cropped_image = image.crop(crop_box)
        signal_message = await generate_content_with_image_live(user_prompt, cropped_image) or ""

    return signal_message


async def build_system_response(image, boxes, classes, names, depth_image, prev_text):
    """
    Build the complete system response based on object detection, depth estimation, and analysis.

    Args:
        image (PIL.Image.Image): Image to analyze.
        boxes (list): List of bounding boxes.
        classes (list): List of class indices.
        names (dict): Mapping of class indices to names.
        depth_image (np.ndarray): Depth data as an array.
        prev_text (str): Previous response text (if any).

    Returns:
        tuple[str, str]: (Final system response, crosswalk message).
    """

    # Save Relevant Object information
    crosswalks = []
    pedestrian_signals = []
    notify_objects = []
    curbs = []
    curb_message = ""

    # Lists of important objects to detect
    crosswalk_types = [
        "construction--flat--crosswalk-plain",
        "marking--discrete--crosswalk-zebra",
        "marking-only--discrete--crosswalk-zebra"
    ]
    pedestrian_signal_types = [
        "object--traffic-light--pedestrians",
        "object--support--traffic-sign-frame",
        "object--traffic-light--general-single",
        "object--traffic-light--general-upright",
        "object--traffic-light--general-horizontal",
        "object--traffic-light--other"
    ]
    objects_to_detect = [
        "human--person--individual",
        "human--person--person-group",
        "human--rider--bicyclist",
        "human--rider--motorcyclist",
        "human--rider--other-rider",
        "object--vehicle--bicycle",
        "object--vehicle--bus",
        "object--vehicle--car",
        "object--vehicle--caravan",
        "object--vehicle--motorcycle",
        "object--vehicle--other-vehicle",
        "object--vehicle--trailer",
        "object--vehicle--truck",
        "object--vehicle--vehicle-group",
        "object--vehicle--wheeled-slow",
        "void--ego-vehicle"
    ]
    curbs_to_detect = ["construction--barrier--curb", "object--manhole"]

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
            cropped_image = image.crop(crop_box)
            cropped_np = np.array(cropped_image)
            # detect_orange_percentage expects BGR image, convert if PIL image is RGB
            if cropped_np.shape[2] == 3 and cropped_np.dtype == np.uint8:
                # Check if PIL image is RGB (default), then convert to BGR for cv2
                if image.mode == 'RGB':
                    cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)

            orange_percent = detect_orange_percentage(cropped_np) # Pass BGR numpy array
            if orange_percent >= 5.0 and mean_depth >= 90: # Consider if depth threshold is appropriate here
                curbs.append(bbox)
                curb_message = "Curb nearby"

        # Yolo crosswalk information
        if obj_name in crosswalk_types:
            crosswalks.append((bbox, mean_depth))

        # Yolo Walk signal information
        if obj_name in pedestrian_signal_types:
            width_box, height_box = (x2 - x1), (y2 - y1)
            ratio = width_box / height_box if height_box != 0 else 0
            x_center = (x1 + x2) / 2.0
            img_width, _ = image.size
            distance_percentage = abs(x_center - (img_width // 2)) / img_width
            if ratio >= 0.75 and distance_percentage <= 0.2:
                pedestrian_signals.append((bbox, mean_depth, obj_name))

        # YOLO object information
        if (obj_name not in crosswalk_types and obj_name not in pedestrian_signal_types and
            obj_name in objects_to_detect):
            obj_response = get_object_response(image, mean_depth, obj_name, x1, x2)
            if obj_response:
                notify_objects.append(obj_response)

    # Get Gemini Crosswalk information
    crosswalk_msg = get_crosswalk_message(image, crosswalks)

    # Await the async call to get_signal_message
    signal_msg = await get_signal_message(image, pedestrian_signals)

    objects_map = {}
    for obj in notify_objects:
        name_key, direction, _ = obj
        if name_key not in objects_map:
            objects_map[name_key] = np.zeros(5)
        mapping = {"left": 0, "near-left": 1, "right": 2, "near-right": 3, "middle": 4}
        if direction in mapping:
            objects_map[name_key][mapping[direction]] += 1
    objects_msg = get_objects_message(objects_map)

    # Start with the base object message
    final_response = objects_msg

    # Clean and prepend curb message if present
    if curb_message:
        curb_cleaned = curb_message.strip().rstrip(".")
        final_response = f"{curb_cleaned}.\n{final_response}"

    # Clean and prepend signal message if it includes color
    if "red" in signal_msg.lower() or "white" in signal_msg.lower():
        signal_cleaned = signal_msg.strip().rstrip(".")
        final_response = f"{signal_cleaned}.\n{final_response}"

    # Final cleanup: remove empty lines and ensure each line ends with exactly one period
    final_response_cleaned = "\n".join(
        line.strip(" .") + "." for line in final_response.splitlines() if line.strip()
    )

    return final_response_cleaned, crosswalk_msg


#############################################################################
# hl2ss system code
#############################################################################

async def process_frame_async(
    frame, text, index, yolo, depth_anything,
    veering_log_path="C:\\Users\\davin\\OneDrive\\Documents\\hl2ss\\viewer\\Depth-Anything-V2\\veering_log.txt",
    history_path="C:\\Users\\davin\\OneDrive\\Documents\\hl2ss\\viewer\\Depth-Anything-V2\\system_history.txt",
    system_a_output_path="C:\\Users\\davin\\OneDrive\\Documents\\hl2ss\\viewer\\Depth-Anything-V2\\system_a_output.txt"
):
    start_time_overall = time.time()

    # YOLO inference
    with torch.inference_mode():
        results = yolo(frame)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names

    # Depth-Anything v2 inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    with torch.inference_mode():
        depth = depth_anything.infer_image(frame)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_image = np.array(depth)

    # System response
    system_output, crosswalk_output = await build_system_response(
        img, boxes, classes, names, depth_image, text
    )
    text = "\n" + system_output

    # Veering feedback response
    content = ""
    try:
        with open(veering_log_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        print(f"{veering_log_path} not found, creating it.")
        open(veering_log_path, "w").close()
    center = "On the crosswalk"
    feedback = ""
    if crosswalk_output == "" or content == "":
        feedback = ""
    elif center in crosswalk_output and center not in content:
        feedback = "Back on crosswalk"
    elif center in content and center not in crosswalk_output:
        feedback = "Veering " + crosswalk_output + " off crosswalk"
    elif center in crosswalk_output and center in content:
        feedback = "On crosswalk"
    elif center not in crosswalk_output and center not in content:
        feedback = "Still veering " + crosswalk_output
    index = index + 1
    with open(veering_log_path, "w") as file:
        file.write(crosswalk_output)

    # For benchmarking/Debugging
    # end_time_overall = time.time()
    # total_cycle_time = end_time_overall - start_time_overall
    # # print(f"Total 3-sec cycle processing time: {total_cycle_time:.4f}s")

    # line = f"{total_cycle_time:.2f}s, {text}\n"
    # with open(system_a_output_path, "a") as file:
    #     file.write(line)


    # Gemini threshold request/response
    history_content = ""
    try:
        with open(history_path, 'r') as file:
            history_content = file.read()
    except FileNotFoundError:
        print(f"{history_path} not found, creating it.")
        open(history_path, 'w').close()
    text = feedback + "\n" + text
    with open(history_path, 'a') as file:
        file.write(str(index) + ": " + text + "\n")
    threshold_query = (
        "Given the following output: " + text + ", and the following history log: " + history_content + " "
        "what numerical threshold value (from 0 to 100, where 100 means completely different) "
        "would you give which will determine whether the following output is different enough from "
        "the most recent information in the history log? Just give the number, nothing else."
        "If there is information involving changes such as:"
        "red/white signal, new objects nearby, amount of seconds left changed, veering changed, give a higher number."
    )
    threshold_response = await get_gemini_live_response(threshold_query)
    try:
        threshold_value = int(threshold_response.strip())
        # print(f"THRESHOLD PARSED: {threshold_value}")
    except ValueError:
        # print(f"Could not parse threshold response as an integer: '{threshold_response}'")
        threshold_value = -1

    return index, threshold_value, text


############################################################
# HoloLens streaming main code
############################################################
if __name__ == '__main__':

    # Port related code
    if ((hl2ss.StreamPort.RM_DEPTH_LONGTHROW in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous RM Depth Long Throw and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem if PV is selected ------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTLEFT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTLEFT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTRIGHT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTRIGHT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_IMU_ACCELEROMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER))
    producer.configure(hl2ss.StreamPort.RM_IMU_GYROSCOPE, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_GYROSCOPE))
    producer.configure(hl2ss.StreamPort.RM_IMU_MAGNETOMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER))
    producer.configure(hl2ss.StreamPort.MICROPHONE, hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER))

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sinks = {}

    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()
        while (sinks[port].get_buffered_frame(0)[0] != 0):
            pass
        print(f'Started {port}')        
        
    # Create Display Map ------------------------------------------------------
    # def display_pv(port, payload):
    #     if (payload.image is not None and payload.image.size > 0):
    #         cv2.imshow(hl2ss.get_port_name(port), payload.image)
    def display_pv(port, payload):
        if payload.image is None or payload.image.size == 0:
            return

        win_name = hl2ss.get_port_name(port)        # e.g., "PERSONAL_VIDEO"

        # Create the window once, make it resizable
        if not cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE):
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # allow user/manual resize
            cv2.resizeWindow(win_name, 1400, 900)         # or any size you want
            # Optional full screen:
            # cv2.setWindowProperty(win_name,
            #                       cv2.WND_PROP_FULLSCREEN,
            #                       cv2.WINDOW_FULLSCREEN)

        # Show the frame (or an up‑scaled copy if you prefer)
        cv2.imshow(win_name, payload.image)

    def display_basic(port, payload):
        if (payload is not None and payload.size > 0):
            cv2.imshow(hl2ss.get_port_name(port), payload)

    def display_depth_lt(port, payload):
        cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 8) # Scaled for visibility
        cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_depth_ahat(port, payload):
        if (payload.depth is not None and payload.depth.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 64) # Scaled for visibility
        if (payload.ab is not None and payload.ab.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_null(port, payload):
        pass

    DISPLAY_MAP = {
        hl2ss.StreamPort.RM_VLC_LEFTFRONT     : display_basic,
        hl2ss.StreamPort.RM_VLC_LEFTLEFT      : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT    : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTRIGHT    : display_basic,
        hl2ss.StreamPort.RM_DEPTH_AHAT        : display_depth_ahat,
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW   : display_depth_lt,
        hl2ss.StreamPort.PERSONAL_VIDEO       : display_pv,
        hl2ss.StreamPort.RM_IMU_ACCELEROMETER : display_null,
        hl2ss.StreamPort.RM_IMU_GYROSCOPE     : display_null,
        hl2ss.StreamPort.RM_IMU_MAGNETOMETER  : display_null,
        hl2ss.StreamPort.MICROPHONE           : display_null,
        hl2ss.StreamPort.SPATIAL_INPUT        : display_null,
        hl2ss.StreamPort.EXTENDED_EYE_TRACKER : display_null,
    }

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

    client = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE) # Create hl2ss client object
    client.open() # Connect to HL2


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

    # Main loop ---------------------------------------------------------------
    last_send_time = time.time()
    index = 0
    while enable:
        imgL = imgR = None
        for port in ports:
            _, data = sinks[port].get_most_recent_frame()
            if data is not None:
                DISPLAY_MAP[port](port, data.payload)

                # Frame data
                frame = data.payload.image

                # How to determine which frame to run
                current_time = time.time()
                if current_time - last_send_time >= 3.5:
                    updated_index, threshold, text = asyncio.run(process_frame_async(frame, "", index, yolo_model, pipe))
                    index = updated_index

                    print("THRESHOLD: ", threshold)
                    print("RESPONSE: ", text)

                    # Determine when to notify user
                    if threshold >= 60 or index == 1:
                        buffer = command_buffer()

                        buffer.debug_message(str(text))
                        client.push(buffer)
                        response = client.pull(buffer)
                    last_send_time = current_time
        cv2.waitKey(1)

    # Stop streams ------------------------------------------------------------
    for port in ports:
        sinks[port].detach()
        producer.stop(port)
        print(f'Stopped {port}')

    client.close()

    # Stop PV Subsystem if PV is selected -------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()


# Potentisl main tasks
# - potentially get the hl2ss version working. figure out why it is not running properly 
