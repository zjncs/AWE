import pyautogui
import time
import pyperclip
from copy import deepcopy

# Key name mapping
KEY_MAPPING = {
    "hotkey": "hotkey",
    "key": "key",
    "press": "press",
    "content": "content",
    "start_box": "start_box",
    "end_box": "end_box",
    "direction": "direction"
}

def escape_single_quotes(text):
    """Escape single quotes"""
    return text.replace("'", "\\'")

def convert_point_to_coordinates(point_str):
    """Convert point marker to coordinates"""
    # Implement specific conversion logic here as needed
    return point_str

def execute_pyautogui_action(responses,
                            image_height: int,
                            image_width: int,
                            input_swap: bool = True,
                            scale_factor: int = 1000):
    '''
    Parse M model output into OSWorld actions and execute pyautogui operations directly
    
    Args:
        responses: Dictionary or list of dictionaries containing model output, structure similar to:
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
        image_height: Image height
        image_width: Image width
        input_swap: Whether to use clipboard input
        scale_factor: Scale factor, default is 1000
    
    Returns:
        List of execution result information
    '''
    
    from copy import deepcopy
    
    # Key name mapping
    KEY_MAPPING = {
        "hotkey": "hotkey",
        "key": "key",
        "press": "press",
        "content": "content",
        "start_box": "start_box",
        "end_box": "end_box",
        "direction": "direction"
    }
    
    def escape_single_quotes(text):
        """Escape single quotes"""
        return text.replace("'", "\\'")
    
    def convert_point_to_coordinates(point_str):
        """Convert point marker to coordinates"""
        return point_str
    
    if isinstance(responses, dict):
        responses = [responses]
    
    result_info = []
    
    for response_id, response in enumerate(responses):
        observation = response.get("observation", "")
        thought = response.get("thought", "")
        
        if response_id == 0:
            print(f"Observation:\n{observation}\n\nThought:\n{thought}")
        else:
            time.sleep(1)
        
        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        # Iterate through action_inputs and replace key names
        old_action_inputs = deepcopy(action_inputs)
        action_inputs = {}
        for key_name, value in old_action_inputs.items():
            new_key_name = KEY_MAPPING.get(key_name, key_name)
            action_inputs[new_key_name] = value
            if "<point>" in value or "<start_point>" in value:
                value = eval(convert_point_to_coordinates(value))
                action_inputs[new_key_name] = value
        
        # Execute corresponding operations based on action type
        if action_type == "hotkey":
            # Execute hotkey operation
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")
            
            # Convert arrow key names
            if hotkey == "arrowleft":
                hotkey = "left"
            elif hotkey == "arrowright":
                hotkey = "right"
            elif hotkey == "arrowup":
                hotkey = "up"
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            if hotkey:
                keys = hotkey.split()
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui.hotkey(*convert_keys)
                result_info.append(f"Execute hotkey: {hotkey}")
        
        elif action_type in ["press", "keydown"]:
            # Execute key press operation
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")
            
            # Convert arrow key names
            if key_to_press == "arrowleft":
                key_to_press = "left"
            elif key_to_press == "arrowright":
                key_to_press = "right"
            elif key_to_press == "arrowup":
                key_to_press = "up"
            elif key_to_press == "arrowdown":
                key_to_press = "down"
            elif key_to_press == "space":
                key_to_press = " "
            
            if key_to_press:
                pyautogui.keyDown(key_to_press)
                result_info.append(f"Press key: {key_to_press}")
        
        elif action_type in ["release", "keyup"]:
            # Execute key release operation
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")
            
            # Convert arrow key names
            if key_to_press == "arrowleft":
                key_to_press = "left"
            elif key_to_press == "arrowright":
                key_to_press = "right"
            elif key_to_press == "arrowup":
                key_to_press = "up"
            elif key_to_press == "arrowdown":
                key_to_press = "down"
            elif key_to_press == "space":
                key_to_press = " "
            
            if key_to_press:
                pyautogui.keyUp(key_to_press)
                result_info.append(f"Release key: {key_to_press}")
        
        elif action_type == "type":
            # Execute text input operation
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            
            if content:
                if input_swap:
                    # Use clipboard input
                    pyperclip.copy(stripped_content)
                    pyautogui.hotkey('ctrl', 'v')
                    time.sleep(0.5)
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui.press('enter')
                    result_info.append(f"Input text: {stripped_content}")
                else:
                    # Direct text input
                    pyautogui.write(stripped_content, interval=0.1)
                    time.sleep(0.5)
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui.press('enter')
                    result_info.append(f"Input text: {stripped_content}")
        
        elif action_type in ["drag", "select"]:
            # Execute drag or select operation
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            
            if start_box and end_box:
                # Parse start box coordinates
                try:
                    if isinstance(start_box, str):
                        start_box = eval(start_box)
                except Exception as e:
                    raise ValueError(f"Start box format error: {start_box}. Error: {e}")
                
                if not isinstance(start_box, (tuple, list)):
                    raise ValueError(f"Start box format error: {start_box}. Expected tuple or list.")
                
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2, y2 = x1, y1
                else:
                    raise ValueError(f"Start box format error: {start_box}. Expected 2 or 4 elements.")
                
                sx = round(float((x1 + x2) / 2) * image_width / scale_factor, 3)
                sy = round(float((y1 + y2) / 2) * image_height / scale_factor, 3)
                
                # Parse end box coordinates
                try:
                    if isinstance(end_box, str):
                        end_box = eval(end_box)
                except Exception as e:
                    raise ValueError(f"End box format error: {end_box}. Error: {e}")
                
                if not isinstance(end_box, (tuple, list)):
                    raise ValueError(f"End box format error: {end_box}. Expected tuple or list.")
                
                if len(end_box) == 4:
                    x1, y1, x2, y2 = end_box
                elif len(end_box) == 2:
                    x1, y1 = end_box
                    x2, y2 = x1, y1
                else:
                    raise ValueError(f"End box format error: {end_box}. Expected 2 or 4 elements.")
                
                ex = round(float((x1 + x2) / 2) * image_width / scale_factor, 3)
                ey = round(float((y1 + y2) / 2) * image_height / scale_factor, 3)
                
                # Execute drag operation
                pyautogui.moveTo(sx, sy)
                pyautogui.dragTo(ex, ey, duration=1.0)
                result_info.append(f"Drag: from ({sx}, {sy}) to ({ex}, {ey})")
        
        elif action_type == "scroll":
            # Execute scroll operation
            start_box = action_inputs.get("start_box")
            
            if start_box:
                try:
                    if isinstance(start_box, str):
                        start_box = eval(start_box)
                except Exception as e:
                    raise ValueError(f"Start box format error: {start_box}. Error: {e}")
                
                if not isinstance(start_box, (tuple, list)):
                    raise ValueError(f"Start box format error: {start_box}. Expected tuple or list.")
                
                if len(start_box) == 4:
                    x = start_box[0]
                    y = start_box[1]
                elif len(start_box) == 2:
                    x, y = start_box
                else:
                    raise ValueError(f"Start box format error: {start_box}. Expected 2 or 4 elements.")
            else:
                x = None
                y = None
            
            direction = action_inputs.get("direction", "")
            
            if x is None:
                # Scroll without specified position
                if "up" in direction.lower():
                    pyautogui.scroll(5)
                    result_info.append("Scroll up")
                elif "down" in direction.lower():
                    pyautogui.scroll(-5)
                    result_info.append("Scroll down")
            else:
                # Scroll at specified position
                if "up" in direction.lower():
                    pyautogui.scroll(5, x=x, y=y)
                    result_info.append(f"Scroll up at ({x}, {y})")
                elif "down" in direction.lower():
                    pyautogui.scroll(-5, x=x, y=y)
                    result_info.append(f"Scroll down at ({x}, {y})")
        
        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Execute mouse click or hover operation
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            
            if start_box:
                try:
                    if isinstance(start_box, str):
                        start_box = eval(start_box)
                except Exception as e:
                    raise ValueError(f"Start box format error: {start_box}. Error: {e}")
                
                if not isinstance(start_box, (tuple, list)):
                    raise ValueError(f"Start box format error: {start_box}. Expected tuple or list.")
                
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2, y2 = x1, y1
                else:
                    raise ValueError(f"Start box format error: {start_box}. Expected 2 or 4 elements.")
                
                x = round(float((x1 + x2) / 2) * image_width / scale_factor, 3)
                y = round(float((y1 + y2) / 2) * image_height / scale_factor, 3)
                
                # Execute corresponding mouse operation based on action type
                if action_type == "left_single" or action_type == "click":
                    pyautogui.click(x, y, button='left')
                    result_info.append(f"Left click: ({x}, {y})")
                elif action_type == "left_double":
                    pyautogui.doubleClick(x, y, button='left')
                    result_info.append(f"Left double click: ({x}, {y})")
                elif action_type == "right_single":
                    pyautogui.click(x, y, button='right')
                    result_info.append(f"Right click: ({x}, {y})")
                elif action_type == "hover":
                    pyautogui.moveTo(x, y)
                    result_info.append(f"Mouse hover: ({x}, {y})")
        
        elif action_type in ["finished"]:
            result_info.append("Task completed")
            return "DONE"
        
        else:
            result_info.append(f"Unrecognized action type: {action_type}")
    
    return result_info