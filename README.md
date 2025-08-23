# Gesture Controlled Games

This repo contains some games that I implemented to be playable with body gestures. Upon closely reading the code you may realise that the code in all three games lookes almost similar, that is because the original program is quite reusable. The setup process is also not very hectic.
hectic.


## Requirements
Python 3.10

Start by installing Python 3.10 on your system if not already. Then setup a virtual environment to work in. In your project directory, type the command
 ``` py -3.10 -m venv <EnvironmentName>``` . Then activate the venv by running the command ```source myenv\Scripts\activate``` .

 Now to install the required packages run the command ```pip install -r requirements.txt``` .

 Now you're all setup to run the program. We tested the program on the online implementation of Subway Surfers available on Poki.com . Just run the ```subwaySurfers.py``` file and follow the following controls.

 ## Gestures

Stand in a way that you are visible to the camera waist up (knees up for good measure). Now,

- Make an "X" with your arms to calibrate the program with your body. Do this everytime the person playing the game changes, and also when any input glitches are experienced.
- Horizontally moving either arm away from the body is a gesture to move in that direction in the game. So left arm means "A" key pressed and right arm means "D" key pressed.
- To Jump just raise either or both your arms above your head.
- To duck just duck slightly.

The detection of the gesture depends on the relative coordinate positions of the various landmarks placed in a skeletal fashion over the body. There are certain thresholds(minimum distance needed) setup throughout the program that can be adjusted to finetune the accuracy of gesture detection.

- side_threshold -> How much horizontal distance your wrist needs to have from your shoulder to consider it as input in that direction(left or right).
- duck_threshold -> How much below your normal waist level you need to go to count it as ducking.
- side_overflow_threshold -> How much your wrists need to be below your shoulder to account for side movement in the first place.
- jump_threshold -> How much you need to raise your hands above your head to account as jump.
