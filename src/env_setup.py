import enum
import json
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any


class BrickType(enum.Enum):
    B2_BLACK = "2x4_black"
    B2_BLUE = "2x4_blue"
    B2_GREEN = "2x4_green"
    B2_PURPLE = "2x4_purple"
    B2_RED = "2x4_red"
    B2_YELLOW = "2x4_yellow"
    B3_BLACK = "2x6_black"
    B3_BLUE = "2x6_blue"
    B3_GREEN = "2x6_green"
    B3_PURPLE = "2x6_purple"
    B3_RED = "2x6_red"
    B3_YELLOW = "2x6_yellow"
    B12_BLACK = "2x2_black"
    B12_BLUE = "2x2_blue"
    B12_GREEN = "2x2_green"
    B12_PURPLE = "2x2_purple"
    B12_RED = "2x2_red"
    B12_YELLOW = "2x2_yellow"

class Arm(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"

class LegoCountResponse(BaseModel):
    """
    A Pydantic model to represent the count of each LEGO brick type and color.
    """
    B2_BLACK: int = Field(default=0, description="Count of 2x4 black bricks. This has 4 circular studs along its length.")
    B2_BLUE: int = Field(default=0, description="Count of 2x4 blue bricks. This has 4 circular studs along its length.")
    B2_GREEN: int = Field(default=0, description="Count of 2x4 green bricks. This has 4 circular studs along its length.")
    B2_PURPLE: int = Field(default=0, description="Count of 2x4 purple bricks. This has 4 circular studs along its length.")
    B2_RED: int = Field(default=0, description="Count of 2x4 red bricks. This has 4 circular studs along its length.")
    B2_YELLOW: int = Field(default=0, description="Count of 2x4 yellow bricks. This has 4 circular studs along its length.")

    B3_BLACK: int = Field(default=0, description="Count of 2x6 black bricks. This has 6 circular studs along its length.")
    B3_BLUE: int = Field(default=0, description="Count of 2x6 blue bricks. This has 6 circular studs along its length.")
    B3_GREEN: int = Field(default=0, description="Count of 2x6 green bricks. This has 6 circular studs along its length.")
    B3_PURPLE: int = Field(default=0, description="Count of 2x6 purple bricks. This has 6 circular studs along its length.")
    B3_RED: int = Field(default=0, description="Count of 2x6 red bricks. This has 6 circular studs along its length.")
    B3_YELLOW: int = Field(default=0, description="Count of 2x6 yellow bricks. This has 6 circular studs along its length.")

    B12_BLACK: int = Field(default=0, description="Count of 2x2 black bricks. This has 2 circular studs along its length.")
    B12_BLUE: int = Field(default=0, description="Count of 2x2 blue bricks. This has 2 circular studs along its length.")
    B12_GREEN: int = Field(default=0, description="Count of 2x2 green bricks. This has 2 circular studs along its length.")
    B12_PURPLE: int = Field(default=0, description="Count of 2x2 purple bricks. This has 2 circular studs along its length.")
    B12_RED: int = Field(default=0, description="Count of 2x2 red bricks. This has 2 circular studs along its length.")
    B12_YELLOW: int = Field(default=0, description="Count of 2x2 yellow bricks. This has 2 circular studs along its length.")
    
    order_of_bricks: List[str] = Field(description="An ordered list of strings, where each string represents a brick in order from left to right. Format each string as 'type_color' (e.g., '2x4_black').")
    
def get_brick_id(brick: str) -> int:
    if "2x4" in brick:
        return 2
    elif "2x6" in brick:
        return 3
    elif "2x2" in brick:
        return 12
    else:
        raise ValueError(f"Unknown brick: {brick}")

def make_env_launch_config(result, file_path):
    """
    Creates a new dictionary with cumulative counts for each brick type,
    grouped by color, and saves it as a JSON file.
    """
    cumulative_counts = {}
    
    # Define the brick types and colors for a structured output
    brick_types = ['B2', 'B3', 'B12']
    colors = ['BLACK', 'BLUE', 'GREEN', 'PURPLE', 'RED', 'YELLOW']

    for brick_type in brick_types:
        cumulative_counts[brick_type] = {}
        for color in colors:
            # Construct the key as it appears in the input 'result' dictionary
            key = f"{brick_type}_{color}"
            # Use the count from the 'result' dictionary, defaulting to 0 if the key doesn't exist
            cumulative_counts[brick_type][color.lower()] = result.get(key, 0)

    with open(file_path, "w") as f:
        json.dump(cumulative_counts, f, indent=4)

# def make_env_setup_config():
