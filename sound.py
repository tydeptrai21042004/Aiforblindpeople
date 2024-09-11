import pyttsx3
import time

# Function to extract text after the colon and handle "User:"
def extract_and_modify_text(line):
    # Check if "User:" is in the line and replace it with "did you ask"
    if line.startswith("User:"):
        modified_line = "did you ask" + line.split(":", 1)[1].strip()
        return modified_line
    # Split the line at the colon and return the text after the colon
    elif ':' in line:
        return line.split(':', 1)[1].strip()  # Take only the text after the first colon
    return line.strip()  # In case there's no colon, return the line as is

# Function to read file, extract text after colon and speak it
def process_file_with_pause(filename):
    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Split the content into two blocks, separated by an empty line
    blocks = []
    current_block = []
    for line in lines:
        if line.strip():  # If the line is not empty
            current_block.append(line)
        else:
            if current_block:  # Add the current block to the blocks list
                blocks.append(current_block)
                current_block = []
    if current_block:  # Add the last block if there are remaining lines
        blocks.append(current_block)

    # Process each block of text
    for i, block in enumerate(blocks):
        print(f"Processing block {i + 1}")

        for line in block:
            # Extract and modify text (replace "User:" if needed)
            extracted_text = extract_and_modify_text(line)

            if extracted_text:
                print(f"Speaking: {extracted_text}")  # Print for debug
                # Speak the text
                engine.say(extracted_text)
        
        # Wait for all speech to finish
        engine.runAndWait()

        # Add a 1-minute pause between blocks, but only after the first block
        if i < len(blocks) - 1:
            print("Pausing for 1 minute...")
            time.sleep(60)  # Pause for 1 minute

# Call the function with the input filename
process_file_with_pause('responses.txt')
