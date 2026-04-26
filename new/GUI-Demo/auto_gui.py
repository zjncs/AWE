"""
Auto GUI - Automated Graphical User Interface Operation Module (Optimized Version)
Code organized by execution steps

Execution Flow:
Step 1: Capture screenshot of the operating environment
Step 2: Construct conversation information (system prompt + image)
Step 3: Call model inference
Step 4: Parse action coordinates and mark
Step 5: Parse action commands and execute GUI operations
"""

import os
import time
import base64
import re
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

import pyautogui
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from volcenginesdkarkruntime import Ark

from prompt import COMPUTER_USE_DOUBAO
from ui_tars.action_parser import parse_action_to_structure_output
from parse import execute_pyautogui_action


# ============================================================================
# Global Configuration
# ============================================================================

# API Configuration
API_KEY = os.getenv('ARK_API_KEY')
MODEL_NAME = "doubao-seed-1-6-vision-250815"
TEMPERATURE = 0.0

# Screen Configuration
DEFAULT_SCREENSHOT_DIR = "/Users/bytedance/demo/"
SCREENSHOT_FORMAT = "png"

# Coordinate Configuration
COORDINATE_SCALE = 1000  # Model output coordinate range (0-1000)

# Model Parsing Configuration
MAX_PIXELS = 16384 * 28 * 28
MIN_PIXELS = 100 * 28 * 28
MODEL_TYPE = "doubao"


def validate_config() -> bool:
    """
    Validate configuration
    
    Returns:
        bool: Whether configuration is valid
        
    Raises:
        ValueError: When API Key is not set
    """
    if not API_KEY:
        raise ValueError("❌ ARK_API_KEY environment variable not set")
    return True


# ============================================================================
# Step 1: Capture screenshot of the operating environment
# ============================================================================

def capture_screenshot(
    save_dir: str = None,
    filename: str = None
) -> Tuple[str, int, int]:
    """
    Capture screenshot of the operating environment
    
    Functions:
    1. Get current screen dimensions
    2. Capture full screen area
    3. Save screenshot to specified directory
    
    Args:
        save_dir: Save directory, uses default configured directory if None
        filename: Filename, uses timestamp if None
        
    Returns:
        Tuple[str, int, int]: (file path, width, height)
    """
    # Use default directory
    if save_dir is None:
        save_dir = DEFAULT_SCREENSHOT_DIR
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screen_{timestamp}.{SCREENSHOT_FORMAT}"
    
    filepath = os.path.join(save_dir, filename)
    
    # Get screen dimensions
    pyautogui_width, pyautogui_height = pyautogui.size()
    
    print("=== Step 1: Capture screenshot of the operating environment ===")
    print(f"Screen dimensions: {pyautogui_width}x{pyautogui_height}")
    print("Capturing screen (including full screen area)...")
    
    # Capture screenshot
    screenshot = pyautogui.screenshot(
        region=(0, 0, pyautogui_width, pyautogui_height)
    )
    screenshot.save(filepath)
    
    # Verify screenshot dimensions
    img_width, img_height = screenshot.size
    print(f"Screenshot dimensions: {img_width}x{img_height}")
    print(f"Screenshot saved to: {filepath}")

    return (filepath, img_width, img_height)


def encode_image(image_path: str) -> str:
    """
    Encode image to Base64 string (helper function)
    
    Args:
        image_path: Image path
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============================================================================
# Step 2: Construct conversation information (system prompt + image)
# ============================================================================

def build_conversation(
    instruction: str,
    image_path: str,
    language: str = "Chinese"
) -> List[Dict[str, Any]]:
    """
    Construct conversation information (system prompt + image)
    
    Functions:
    1. Construct system prompt (including operation instruction)
    2. Encode screenshot to Base64
    3. Construct complete messages list
    
    Args:
        instruction: Operation instruction (e.g., "Please open browser")
        image_path: Screenshot path
        language: Language, defaults to Chinese
        
    Returns:
        List[Dict[str, Any]]: messages list containing system prompt and image
    """
    print(f"\n=== Step 2: Construct conversation information (system prompt + image) ===")
    
    # Encode screenshot
    image_format = Path(image_path).suffix[1:]  # Remove dot
    base64_image = encode_image(image_path)
    print(f"Image encoded to Base64 format")

    # Construct system prompt
    system_prompt = COMPUTER_USE_DOUBAO.format(
        instruction=instruction,
        language=language
    )
    print(f"System prompt constructed")
    print(f"   Operation instruction: {instruction}")
    print(f"   Language: {language}")
    
    # Construct messages
    messages = [
        {
            "role": "user",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    print(f"Conversation information constructed")
    
    return messages


# ============================================================================
# Step 3: Call model inference
# ============================================================================

def call_model_inference(
    messages: List[Dict[str, Any]]
) -> str:
    """
    Call model inference
    
    Functions:
    1. Create model client
    2. Call Doubao model for image understanding
    3. Get model response
    
    Args:
        messages: Constructed conversation information
        
    Returns:
        str: Complete model response
    """
    print(f"\n=== Step 3: Call model inference ===")
    
    # Initialize Ark client
    client = Ark(
        base_url="http://ark.cn-beijing.volces.com/api/v3",
        api_key=API_KEY,
    )
    
    # Call model
    print("Calling model for analysis...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )
    
    # Get response
    full_response = response.choices[0].message.content
    print(f"Model inference completed")
    print(f"\nModel response:\n{full_response}\n")

    return full_response


# ============================================================================
# Step 4: Parse action coordinates and mark
# ============================================================================

def parse_coordinates_from_response(
    response: str,
    image_path: str
) -> Tuple[int, int]:
    """
    Parse coordinate information from model response (helper function)
    
    Args:
        response: Model response text
        image_path: Image path (used to get dimensions)
        
    Returns:
        Tuple[int, int]: (x, y) absolute coordinates
        
    Raises:
        ValueError: When coordinates cannot be parsed
    """
    # Get image dimensions
    image = Image.open(image_path)
    original_width, original_height = image.size
    print(f"Image dimensions: {original_width}x{original_height}")
    
    # Parse coordinates (supports <point> format)
    match = re.search(r'<point>(\d+)\s+(\d+)</point>', response)
    if match:
        # Convert relative coordinates (0-1000) to absolute coordinates
        x = int(int(match.group(1)) / COORDINATE_SCALE * original_width)
        y = int(int(match.group(2)) / COORDINATE_SCALE * original_height)
        action_point = (x, y)
        print(f"Coordinate parsing successful: {action_point}")
        return action_point
    else:
        raise ValueError(f"Failed to parse coordinates from response. Original response: {response}")


def mark_position_on_image(
    image_path: str,
    action_point: Tuple[int, int],
    output_path: Optional[str] = None,
    show_image: bool = False
) -> Image.Image:
    """
    Mark target position on screenshot (helper function)
    
    Args:
        image_path: Original screenshot path
        action_point: Coordinates to mark (x, y)
        output_path: Save path for marked image (optional)
        show_image: Whether to display image
        
    Returns:
        Image.Image: Marked image object
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw red circle marker
    x, y = action_point
    radius = 10
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=3)
    
    # Save or display
    if output_path:
        image.save(output_path)
        print(f"Marked image saved to: {output_path}")
    
    if show_image:
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Target position: ({x}, {y})")
        plt.show()
    
    return image


def parse_and_mark(
    response: str,
    screenshot_path: str,
    show_image: bool = True
) -> Tuple[Tuple[int, int], str]:
    """
    Parse action coordinates and mark
    
    Functions:
    1. Parse coordinate information from model response
    2. Convert relative coordinates to absolute coordinates
    3. Mark target position on screenshot
    4. Save marked image
    
    Args:
        response: Model response
        screenshot_path: Original screenshot path
        show_image: Whether to display marked image
        
    Returns:
        Tuple[Tuple[int, int], str]: (action coordinates, marked image path)
    """
    print(f"\n=== Step 4: Parse action coordinates and mark ===")
    
    # Parse coordinates
    action_point = parse_coordinates_from_response(response, screenshot_path)
    
    # Generate marked image path
    mark_path = screenshot_path.replace(
        f".{SCREENSHOT_FORMAT}",
        f"_marked.{SCREENSHOT_FORMAT}"
    )
    
    # Mark position
    mark_position_on_image(screenshot_path, action_point, mark_path)
    
    if show_image:
        print("Displaying marked image...")
        mark_position_on_image(screenshot_path, action_point, show_image=True)
    
    print(f"Coordinate parsing and marking completed")
    
    return (action_point, mark_path)


# ============================================================================
# Step 5: Parse action commands and execute GUI operations
# ============================================================================

def parse_and_execute_action(
    response: str,
    img_height: int,
    img_width: int,
    verbose: bool = True,
    scale_factor: int = 1000
) -> List[str]:
    """
    Parse action commands and execute GUI operations
    
    Functions:
    1. Parse model output to structured action dictionary
    2. Convert relative coordinates to absolute coordinates
    3. Execute pyautogui operations (click, input, scroll, etc.)
    
    Args:
        response: Model response
        img_height: Image height
        img_width: Image width
        verbose: Whether to print detailed information
        
    Returns:
        List[str]: Execution result information list
    """
    print(f"\n=== Step 5: Parse action commands and execute GUI operations ===")
    
    # Parse model output to coordinate dictionary
    parsed_dict = parse_action_to_structure_output(
        response,
        1,
        img_width,
        img_height,
        model_type=MODEL_TYPE,
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS
    )
    
    print(f"Action parsing successful!")
    print(f"Parsing result: {parsed_dict}")
    
    # Execute operations
    print(f"Starting GUI operations execution...")
    result = execute_pyautogui_action(
        parsed_dict,
        img_height,
        img_width,
        verbose,
        scale_factor
    )
    
    print(f"GUI operations execution completed!")
    print(f"Execution result: {result}")
    
    return result


# ============================================================================
# Complete workflow execution
# ============================================================================

def auto_screen_operation(
    instruction: str,
    show_image: bool = True,
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Complete screen automation workflow (executed step by step)
    
    Execution Flow:
    Step 1: Capture screenshot of the operating environment
    Step 2: Construct conversation information (system prompt + image)
    Step 3: Call model inference
    Step 4: Parse action coordinates and mark
    Step 5: Parse action commands and execute GUI operations
    
    Args:
        instruction: Operation instruction (e.g., "Please open browser")
        show_image: Whether to display marked image
        save_dir: Screenshot save directory
        
    Returns:
        Dict[str, Any]: Operation result dictionary containing paths, coordinates, response, etc.
    """
    # Validate configuration
    validate_config()
    
    result = {
        "success": False,
        "screenshot_path": None,
        "marked_path": None,
        "action_point": None,
        "response": None,
        "parsed_dict": None,
        "error": None
    }
    
    try:
        # ========== Step 1: Capture screenshot of the operating environment ==========
        screenshot_path, img_width, img_height = capture_screenshot(save_dir)
        result["screenshot_path"] = screenshot_path
        
        # ========== Step 2: Construct conversation information (system prompt + image) ==========
        messages = build_conversation(
            instruction,
            screenshot_path
        )
        
        # ========== Step 3: Call model inference ==========
        response = call_model_inference(messages)
        result["response"] = response
        time.sleep(2)
        
        # ========== Step 4: Parse action coordinates and mark ==========
        action_point, mark_path = parse_and_mark(
            response,
            screenshot_path,
            show_image
        )
        result["action_point"] = action_point
        result["marked_path"] = mark_path
        
        # ========== Step 5: Parse action commands and execute GUI operations ==========
        time.sleep(2)  # Give user time to view
        parse_and_execute_action(
            response,
            img_height,
            img_width,
            verbose=True,
            scale_factor=COORDINATE_SCALE
        )
        
        result["success"] = True

        print(f"Success: {result['success']}")
        print(f"Screenshot path: {result['screenshot_path']}")
        print(f"Marked path: {result['marked_path']}")
        print(f"Action coordinates: {result['action_point']}")
        
        print(f"\n=== Automation completed! ===")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    print("=== Auto GUI - Automated Graphical User Interface Operation ===")
    
    # Execute automation workflow
    instruction = "打开一个新的浏览器页签"
    
    auto_screen_operation(instruction, show_image=True)