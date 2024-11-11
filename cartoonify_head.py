import cv2
import numpy as np
import mediapipe as mp

def cartoonify_image(img):
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

def enlarge_head(frame, face_landmarks, scale=1.2):
    """
    Enlarge the head region based on facial landmarks.

    Parameters:
    - frame: The original image frame.
    - face_landmarks: Detected facial landmarks from Mediapipe.
    - scale: Scaling factor for the head enlargement.

    Returns:
    - Modified frame with enlarged head.
    """
    # Extract landmark coordinates
    h, w, _ = frame.shape
    landmarks = face_landmarks.landmark
    
    # Fixed landmark indices for chin and forehead
    chin_index = 152    # Mediapipe landmark for chin
    forehead_index = 10  # Mediapipe landmark for forehead
    
    # Convert normalized coordinates to pixel values
    chin_y = int(landmarks[chin_index].y * h)
    forehead_y = int(landmarks[forehead_index].y * h)
    
    # Calculate the center of the face
    center_x = int(landmarks[1].x * w)  # Using landmark 1 (tip of the nose)
    center_y = int((chin_y + forehead_y) / 2)
    
    # Estimate the radius of the head
    radius = int((chin_y - forehead_y) * scale)
    
    # Define the bounding box for the head
    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius
    y2 = center_y + radius
    
    # Ensure the bounding box is within frame boundaries
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    
    # Extract the head region
    head_region = frame[y1:y2, x1:x2]
    
    # Calculate new size after scaling
    new_width = int(head_region.shape[1] * scale)
    new_height = int(head_region.shape[0] * scale)
    
    # Resize the head region
    resized_head = cv2.resize(head_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate position to place the resized head
    y_start = center_y - new_height // 2
    y_end = y_start + new_height
    x_start = center_x - new_width // 2
    x_end = x_start + new_width
    
    # Handle boundary conditions
    if y_start < 0:
        resized_head = resized_head[-y_start:]
        y_start = 0
    if x_start < 0:
        resized_head = resized_head[:, -x_start:]
        x_start = 0
    if y_end > h:
        resized_head = resized_head[:h - y_start, :]
        y_end = h
    if x_end > w:
        resized_head = resized_head[:, :w - x_start]
        x_end = w
    
    # Create a mask for seamless cloning
    mask = 255 * np.ones(resized_head.shape, resized_head.dtype)
    
    # Define the center for seamless cloning
    center = (x_start + resized_head.shape[1] // 2, y_start + resized_head.shape[0] // 2)
    
    # Seamlessly clone the resized head into the frame
    try:
        frame = cv2.seamlessClone(resized_head, frame, mask, center, cv2.NORMAL_CLONE)
    except:
        # In case seamlessClone fails, fallback to simple overlay
        frame[y_start:y_end, x_start:x_end] = resized_head
    
    return frame

def main():
    # Initialize webcam capture (0 is the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Optional: Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Drawing specifications (optional, for visualization)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
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
                
                # Enlarge the head
                frame = enlarge_head(frame, face_landmarks, scale=1.3)  # Adjust scale as needed
        
        # Apply cartoon effect
        cartoon_frame = cartoonify_image(frame)
        
        # Display the resulting frame
        cv2.imshow('Cartoonified Webcam Feed with Enlarged Head', cartoon_frame)
        
        # Exit condition - press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
