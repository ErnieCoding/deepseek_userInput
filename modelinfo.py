import subprocess
import re

def get_context_length(model_name):
    """
    Retrieve context length attribute of a model
    """
    try:
        result = subprocess.run(
            ["ollama", "show", model_name], 
            capture_output=True, text=True, check=True
        )
        
        match = re.search(r"context length\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Context length not found in output.")
    
    except subprocess.CalledProcessError as e:
        print("Error running ollama show:", e)
        return None
