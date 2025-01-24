from djitellopy import Tello
from voice import VoiceController
from action_generator import ActionGenerator
import cv2
import time
from datetime import datetime

class DroneAgent:
    def __init__(self):
        self.tello = Tello()
        self.voice_controller = VoiceController()
        self.action_generator = ActionGenerator()
        self.img_counter = 0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.video_writer = None
        self.face_video_writer = None
        
    def execute_action(self, action):
        try:
            if action.startswith('takeoff'):
                self.tello.takeoff()
            elif action.startswith('land'):
                self.tello.land()
            elif action.startswith('move_forward'):
                distance = int(action.split('(')[1].split(')')[0])
                self.tello.move_forward(distance)
            elif action.startswith('move_backward'):
                distance = int(action.split('(')[1].split(')')[0])
                self.tello.move_backward(distance)
            elif action.startswith('move_left'):
                distance = int(action.split('(')[1].split(')')[0])
                self.tello.move_left(distance)
            elif action.startswith('move_right'):
                distance = int(action.split('(')[1].split(')')[0])
                self.tello.move_right(distance)
            elif action.startswith('rotate_clockwise'):
                degrees = int(action.split('(')[1].split(')')[0])
                self.tello.rotate_clockwise(degrees)
            elif action.startswith('rotate_counter_clockwise'):
                degrees = int(action.split('(')[1].split(')')[0])
                self.tello.rotate_counter_clockwise(degrees)
            elif action.startswith('take_photo'):
                frame = self.tello.get_frame_read().frame
                cv2.imwrite(f'tello_photo_{self.img_counter}.jpg', frame)
                self.img_counter += 1
            
            print(f"Executed action: {action}")
            
        except Exception as e:
            print(f"Error executing action: {e}")
    
    def setup_video_writers(self):
        """Set up video writers for main stream and face detection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(f'tello_stream_{timestamp}.avi', 
                                          fourcc, 30.0, (960, 720))
        self.face_video_writer = cv2.VideoWriter(f'tello_faces_{timestamp}.avi', 
                                                fourcc, 30.0, (960, 720))

    def run(self):
        try:
            self.tello.connect()
            print("Connected to Tello")
            print(f"Battery level: {self.tello.get_battery()}%")
            self.tello.streamon()
            
            self.setup_video_writers()
            
            while True:
                command = self.voice_controller.listen_for_command()
                if command:
                    action = self.action_generator.generate_action(command)
                    self.execute_action(action)
                    
                frame = self.tello.get_frame_read().frame
                if frame is None:
                    continue
                    
                frame = cv2.resize(frame, (960, 720))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                face_frame = frame.copy()
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(face_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(face_frame, 'Face', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                self.video_writer.write(frame)
                self.face_video_writer.write(face_frame)
                
                cv2.imshow("Tello Stream", frame)
                cv2.imshow("Face Detection", face_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            if self.video_writer:
                self.video_writer.release()
            if self.face_video_writer:
                self.face_video_writer.release()
            self.tello.land()
            self.tello.streamoff()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    drone_agent = DroneAgent()
    drone_agent.run()