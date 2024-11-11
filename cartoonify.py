import cv2
import numpy as np

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

def main():
    # Initialize webcam capture (0 is the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Optional: Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to exit the program.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Apply cartoon effect
        cartoon_frame = cartoonify_image(frame)
        
        # Display the resulting frame
        cv2.imshow('Cartoonified Webcam Feed', cartoon_frame)
        
        # Exit condition - press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
