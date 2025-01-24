from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class ActionGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def generate_action(self, command):
        system_prompt = """
        Convert drone commands to specific actions. Available actions are:
        - takeoff
        - land
        - move_forward(distance)
        - move_backward(distance)
        - move_left(distance)
        - move_right(distance)
        - rotate_clockwise(degrees)
        - rotate_counter_clockwise(degrees)
        - move_up(distance)
        - move_down(distance)
        - take_photo
        
        Return only the action name and parameters, nothing else.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ],
            temperature=0.0,
        )
        
        return response.choices[0].message.content.strip()