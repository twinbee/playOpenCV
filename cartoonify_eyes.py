import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Mediapipe Drawing Utils for visualization (optional)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def cartoonify_image(img):
    """
    Applies a cartoon effect to the input image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise
    gray_blur = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=9
    )
    
    # Apply bilateral filter to smooth the image while keeping edges sharp
    color = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)
    
    # Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

def enlarge_eye(frame, eye_landmarks, scale=3.0):
    """
    Enlarges the eye region based on the provided landmarks.
    
    Parameters:
    - frame: The original image frame.
    - eye_landmarks: List of (x, y) tuples representing eye landmarks.
    - scale: Scaling factor (e.g., 3.0 for 300% enlargement).
    
    Returns:
    - Modified frame with the enlarged eye.
    """
    # Convert normalized landmarks to pixel coordinates
    h, w, _ = frame.shape
    eye_points = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in eye_landmarks], np.int32)
    
    # Compute the bounding rectangle for the eye
    x, y, ex, ey = cv2.boundingRect(eye_points)
    
    # Extract the eye region
    eye_region = frame[y:y+ey, x:x+ex]
    
    # Resize the eye region
    eye_region_enlarged = cv2.resize(eye_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Calculate new position to overlay the enlarged eye
    new_x = x - int((eye_region_enlarged.shape[1] - ex) / 2)
    new_y = y - int((eye_region_enlarged.shape[0] - ey) / 2)
    
    # Handle boundary conditions
    if new_x < 0:
        eye_region_enlarged = eye_region_enlarged[:, -new_x:]
        new_x = 0
    if new_y < 0:
        eye_region_enlarged = eye_region_enlarged[-new_y:, :]
        new_y = 0
    if new_x + eye_region_enlarged.shape[1] > w:
        eye_region_enlarged = eye_region_enlarged[:, :w - new_x]
    if new_y + eye_region_enlarged.shape[0] > h:
        eye_region_enlarged = eye_region_enlarged[:h - new_y, :]
    
    # Create a mask for the eye region
    eye_gray = cv2.cvtColor(eye_region_enlarged, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(eye_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Overlay the enlarged eye onto the frame
    frame[new_y:new_y + eye_region_enlarged.shape[0], new_x:new_x + eye_region_enlarged.shape[1]] = cv2.bitwise_and(
        frame[new_y:new_y + eye_region_enlarged.shape[0], new_x:new_x + eye_region_enlarged.shape[1]],
        frame[new_y:new_y + eye_region_enlarged.shape[0], new_x:new_x + eye_region_enlarged.shape[1]],
        mask=cv2.bitwise_not(mask)
    )
    frame[new_y:new_y + eye_region_enlarged.shape[0], new_x:new_x + eye_region_enlarged.shape[1]] += eye_region_enlarged
    
    return frame

def main():
    # Initialize webcam capture (0 is the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Optional: Set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to exit the program.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a mirror effect (optional)
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find face landmarks
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Optionally, draw facial landmarks
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                
                # Define eye landmark indices for left and right eyes
                LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 153, 154, 155]
                RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380]
                
                # Extract left eye landmarks
                left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
                
                # Extract right eye landmarks
                right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]
                
                # Enlarge left eye
                frame = enlarge_eye(frame, left_eye, scale=3.0)  # 300% enlargement
                
                # Enlarge right eye
                frame = enlarge_eye(frame, right_eye, scale=3.0)  # 300% enlargement
        
        # Apply cartoon effect
        cartoon_frame = cartoonify_image(frame)
        
        # Display the resulting frame
        cv2.imshow('Cartoonified Webcam Feed with Enlarged Eyes', cartoon_frame)
        
        # Exit condition - press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
