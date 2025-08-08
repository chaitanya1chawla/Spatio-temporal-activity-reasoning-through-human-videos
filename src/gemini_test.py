import os
import cv2
import enum
import time
import json
from PIL import Image
from collections import Counter

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, create_model

# from run_launch import run_dual_gp4_launch
from env_setup import (
    Arm,
    BrickType,
    LegoCountResponse,
    get_brick_id,
    make_env_launch_config,
    # make_env_setup_config,
)


class Task(enum.Enum):
    PICK = (
        "Pick a brick",
        "A robot arm moves towards a single brick, and lifts it from its starting position.",
    )
    PLACE_CENTER = (
        "Place brick (Center-aligned on brick underneath)",
        "The brick is placed so its side edges are **perfectly flush** with the edges of the brick below it. There is **no overhang** on either side.",
    )
    PLACE_SHIFTED = (
        "Place brick (Shifted left from brick underneath)",
        "The brick is placed with a clear **offset to the left**. A portion of the brick **overhangs the left edge** of the brick below it, **leaving the top surface of the bottom brick exposed on the right**.",
    )
    DEMO_OVER = (
        "Demo over",
        "No more actions are being performed by the arms, and the assembly is complete.",
    )
    # PLACE = ("Place a brick", "An arm holding a brick moves it to the assembly, and releases it.")
    # HANDOVER = ("Handover a brick",
    #             "One arm places a loose brick directly onto the other arm's waiting end-effector, which acts as a platform. **Crucially, the transfer happens arm-to-arm; the brick does not get placed on the main assembly during this action. It must always be followed by a Place action**")
    # SUPPORT = ("Support a brick",
    #            "One arm acts as a static brace, holding an overhanging part of the assembly from underneath, while the second arm places a new brick onto that section. "
    #            "**IMPORTANT: For this 'Support' step, label the 'brick_type' with the brick that is actively being placed, not the stationary brick that is being braced.** "
    #            "This action must always be followed by a Place action.")

    @property
    def display_name(self):
        return self.value[0]

    @property
    def description(self):
        return self.value[1]

TaskType = enum.Enum("TaskType", {task.name: task.display_name for task in Task})

def get_task_definitions():
    """Formats the detailed task descriptions for the Gemini prompt."""
    definitions = []
    for task in Task:
        definitions.append(f"- **{task.display_name}**: {task.description}")
    return "\n".join(definitions)

def createClass_lego_assembly(brick_class, num_steps=15):
    """
    Dynamically creates a Pydantic BaseModel for Lego assembly orders.

    Args:
        num_steps (int): The number of steps the assembly order should have.

    Returns:
        BaseModel: A dynamically generated Pydantic model.
    """
    fields = {}
    for i in range(1, num_steps + 1):
        fields[f"Step{i}"] = (TaskType, ...)
        fields[f"Step{i}_brick_type"] = (brick_class, ...)
        fields[f"Step{i}_execution_arm"] = (
            Arm,
            Field(
                ...,
                description=f"The robot arm (Left or Right) that picked up the brick {i}. **Pick and Place are executed by the same arm.**",
            ),
        )
        fields[f"Step{i}_timestamp"] = (
            int,
            Field(
                ...,
                description=f"The timestamp in seconds when step {i} occurs in the video.",
            ),
        )

    # Create the model dynamically
    DynamicLegoAssemblyOrder = create_model(
        "LegoAssemblyOrder",  # The name of the new model class
        **fields,  # Unpack the dictionary of fields
    )
    return DynamicLegoAssemblyOrder

def make_assembly_task_config(
    num_steps, result: dict, file_path: str = "assembly_task_gemini.json"
):
    x_placement = 24  # Hardcoded for now
    y_placement = 28
    z_placement = 1
    ori = 1

    press_z = 2  # Hardcoded for now
    press_ori = 0
    assembly_task_dict = {}

    step_count = 0
    for i in range(1, num_steps + 1):
        task_name = result.get(f"Step{i}")

        # A new assembly step is created when a brick is PLACED
        if task_name in (TaskType.PLACE_CENTER.value, TaskType.PLACE_SHIFTED.value):
            step_count += 1

            # Check if the placement task was shifted
            is_shifted = "Shifted" in task_name

            assembly_task_dict[f"{step_count}"] = {
                "x": x_placement,
                "y": y_placement,
                "z": z_placement,
                "brick_id": get_brick_id(result[f"Step{i}_brick_type"]),
                "ori": ori,
                "press_side": 1,
                "press_offset": 1,
                "manipulate_type": 0,  # 1 if is_handover else 0, # Check if the PREVIOUS step was a handover
                "press_x": x_placement,
                "press_y": y_placement + 1,
                "press_z": z_placement + 1,
                "press_ori": press_ori,
                # Support brick if it is shifted from bottom brick.
                "support_x": x_placement if is_shifted else -1,
                "support_y": y_placement if is_shifted else -1,
                "support_z": z_placement if is_shifted else 0,
                "support_ori": 1 if is_shifted else -1,
                "brick_seq": -1,
            }
            # IMPORTANT: Only increment z-height after a successful placement
            z_placement += 1

        # Stop processing if the demo is over
        elif task_name == TaskType.DEMO_OVER.value:
            break

    with open(file_path, "w") as f:
        json.dump(assembly_task_dict, f, indent=2)
    return file_path

def initial_scene_understanding(client, video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get first image
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] First image load failed!")
    frame_path = "videos/frame_10.png"
    cv2.imwrite(frame_path, frame)

    # Get last image
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 10)
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Last image load failed!")
    last_frame_path = "videos/frame_last_10.png"
    cv2.imwrite(last_frame_path, frame)
    cap.release()

    image_file = client.files.upload(file=frame_path)
    last_image_file = client.files.upload(file=last_frame_path)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=LegoCountResponse,
            system_instruction="You are an expert at analyzing LEGO bricks. You can identify LEGO brick types (2x2, 2x4, 2x6) and colors (black, blue, green, purple, red, yellow). Count each brick by its type and color. Only count bricks at the bottom of the frame.",
        ),
        contents=[
            types.Part(text="This is first image from demonstration"),
            types.Part(file_data=types.FileData(file_uri=image_file.uri, mime_type="image/png")),
            types.Part(text="This is last image from demonstration"),
            types.Part(file_data=types.FileData(file_uri=last_image_file.uri, mime_type="image/png")),
            types.Part(
                text="Look at the first and the last image, count the number of each type of brick. Compare the first and the last image, and only count the images which move from their place. **Only count bricks at the bottom of the frame.**"
            ),
        ],
    )

    result = {}
    try:
        if response.text:
            result = json.loads(response.text)
            print("---------------------------------------------------------------")
            print("Current Environment Count:")
            print(json.dumps(result, indent=2))
            print("---------------------------------------------------------------\n")
            make_env_launch_config(result, "new_lego_env.json")
        else:
            print("No response text received from Gemini API")
    except json.JSONDecodeError:
        print("Could not parse response as JSON, showing raw response above.")

    ### Run dual_gp4.launch with the new_lego_env.json file
    # print("---------------------------------------------------------------")
    # print("Launch command:")
    # launch_command = run_dual_gp4_launch("new_lego_env.json")
    # print(launch_command)
    print("---------------------------------------------------------------\n")
    print("---------------------------------------------------------------")
    print("Making env setup config...")
    env_setup_file_path = "env_setup_gemini.json"
    # env_setup_file_path = make_env_setup_config(result, "env_setup_gemini.json")
    # print(f"Env setup config file path: {env_setup_file_path}")
    print("---------------------------------------------------------------\n")

    return result

def get_moving_bricks(result):
    """
    Extracts the list of bricks that are part of the assembly from the
    initial scene analysis.
    """
    # The 'order_of_bricks' field from the initial response is the most reliable
    # source for the list of bricks that will be moved.
    movable_bricks = result.get("order_of_bricks", [])

    available_bricks_text = "\n- ".join(movable_bricks)
    print(f"Available and movable bricks in scene:\n- {available_bricks_text}")

    return movable_bricks, available_bricks_text


def main():
    import subprocess

    subprocess.run(["bash", "gemini_api.sh"])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    client = genai.Client(api_key=api_key)

    # Load the video file
    video_path = "videos/video-2025-06-30_00.15.31_zoom.mp4" #(1)
    # video_path = "videos/video-2025-06-29_23.56.09_cropped6.mp4"
    # video_path = "videos/video-2025-06-29_23_51_54_trimmed_zoom.mp4" #(2)
    # video_path = "videos/245_trimmed-zoomed.mp4"   # (3)
    # video_path = "videos/245_trimmed-zoomed2.mp4"
    # video_path = "videos/support+handover2.mp4"
    # video_path = "videos/support2.mp4"
    video_path = "videos/black_a_blue_a_purple_a_yellow_s_green_a_red_s.mp4"

    result = initial_scene_understanding(client, video_path)
    available_bricks, available_bricks_text = get_moving_bricks(result)

    curr_env = {}
    for brick in available_bricks:
        # Assumes brick format is "2x4 color" or "2x4_color"
        color = brick.replace("_", " ").split(" ")[-1].upper()
        curr_env[f"B{get_brick_id(brick)}_{color}"] = brick.replace(" ", "_")

    DynamicCurrEnv = enum.Enum("DynamicCurrEnv", curr_env)
    LegoAssemblyOrder = createClass_lego_assembly(DynamicCurrEnv, 15)

    # For video:
    print("---------------------------------------------------------------")
    print(f"Uploading files for video analysis...")
    video_file = client.files.upload(file=video_path)
    example_image_file = client.files.upload(file="videos/Selection_006.png")
    # example_image_file = client.files.upload(path="videos/Selection_007.png")

    perfect_example_response = """
    {
        "Step1": "Pick a brick",
        "Step1_brick_type": "2x4_purple",
        "Step1_execution_arm": "Right",
        "Step1_timestamp": 3,
        "Step2": "Place brick (Center-aligned on brick underneath)",
        "Step2_brick_type": "2x4_purple",
        "Step2_execution_arm": "Right",
        "Step2_timestamp": 11,
        "Step3": "Pick a brick",
        "Step3_brick_type": "2x4_yellow",
        "Step3_execution_arm": "Right",
        "Step3_timestamp": 20,
        "Step4": "Place brick (Shifted left from brick underneath)",
        "Step4_brick_type": "2x4_yellow",
        "Step4_execution_arm": "Right",
        "Step4_timestamp": 29,
        "Step5": "Pick a brick",
        "Step5_brick_type": "2x4_red",
        "Step5_execution_arm": "Right",
        "Step5_timestamp": 38,
        "Step6": "Place brick (Center-aligned on brick underneath)",
        "Step6_brick_type": "2x4_red",
        "Step6_execution_arm": "Right",
        "Step6_timestamp": 47,
        "Step7": "Pick a brick",
        "Step7_brick_type": "2x4_green",
        "Step7_execution_arm": "Right",
        "Step7_timestamp": 56,
        "Step8": "Place brick (Shifted left from brick underneath)",
        "Step8_brick_type": "2x4_green",
        "Step8_execution_arm": "Right",
        "Step8_timestamp": 65
    }
    """

    while video_file.state == types.FileState.PROCESSING:
        time.sleep(4)
        video_file = client.files.get(name=video_file.name)
        print(f"Waiting for file to be processed... Current state: {video_file.state}")

    if video_file.state != types.FileState.ACTIVE:
        print(f"Error: File did not become active. Current state: {video_file.state}")
        return

    task_definitions = get_task_definitions()
    prompt_text = f"""
    Your goal is to generate a precise, step-by-step assembly order from the video.
    Identify every action performed by the robot arms using the definitions provided.

    **Available Bricks for Assembly:**
    You must only use bricks from the following list for the `brick_type` fields. These are the only bricks that move in the video.
    - {available_bricks_text}

    **Crucial Instructions:**
    1.  **Visually Confirm Every Action:** Only generate a step if you see the action happen. Do not infer or predict actions based on previous patterns. **Not all bricks are moved by the robot arms.**
    2.  **End of Demo:** If the arms stop moving or the video ends, the correct final action is 'Demo over'.

    **Task Definitions:**
    {task_definitions}
    """

    # **Context:**
    # - There is no "Handover" task in this video.
    # **Example of a Correctly Ended Demo:**
    # Here is an example of a correct output where the demo ends after one brick is placed, leaving other bricks unused.
    # ```json
    # {{
    #   "Step1": "Pick a brick",
    #   "Step1_brick_type": "2x4_blue",
    #   "Step1_execution_arm": "Right",
    #   "Step2": "Place brick (Center)",
    #   "Step2_brick_type": "2x4_blue",
    #   "Step2_execution_arm": "Right",
    #   "Step3": "Demo over",
    #   "Step3_brick_type": "2x2_black",
    #   "Step3_execution_arm": "N/A",
    #   "Step4": "Demo over",
    #   "Step4_brick_type": "2x2_black",
    #   "Step4_execution_arm": "N/A"
    # }}
    # ```

    # Generate the assembly order
    response2 = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=LegoAssemblyOrder,
        ),
        contents=[
            # == Turn 1: Example (User) ==
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Here is an example image. Please analyze which bricks are aligned with bricks beneath them."
                    ),
                    types.Part(
                        file_data=types.FileData(
                            file_uri=example_image_file.uri, mime_type="image/png"
                        )
                    ),
                ],
            ),

            # == Turn 1: Example (Model) ==
            types.Content(role="model", parts=[types.Part(text=perfect_example_response)]),

            # == Turn 2: The Real Query (User) ==
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=video_file.uri, mime_type="video/mp4"
                        )
                    ),
                    types.Part(text=prompt_text),
                ],
            ),
        ],
    )

    print("---------------------------------------------------------------")
    print("\nAssembly Order:")
    response2_dict = json.loads(response2.text)
    print(response2.text)
    print("---------------------------------------------------------------\n")

    assembly_task_file_path = make_assembly_task_config(
        num_steps=16, result=response2_dict
    )

    print("---------------------------------------------------------------")
    print(f"Assembly task config file path: {assembly_task_file_path}")
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    main()