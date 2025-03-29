
import subprocess
import re

def get_context_length(model_name):
    """
    Retrieve context length attribute of a model with improved error handling
    """
    try:
        # Use shell=True and specify encoding to handle potential encoding issues
        result = subprocess.run(
            ["ollama", "show", model_name], 
            capture_output=True, 
            text=True, 
            check=True,
            encoding='utf-8',  # Explicitly use UTF-8 encoding
            errors='replace'   # Replace any undecodable bytes
        )
        
        # Print the full stdout for debugging
        #print(f"Ollama show output for {model_name}:\n{result.stdout}")
        
        match = re.search(r"context length\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Context length not found for model {model_name}")
            return None
    
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama show for {model_name}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error retrieving context length for {model_name}: {e}")
        return None