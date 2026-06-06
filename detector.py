import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from collections import deque

# Create helper function to display text on image frame
def put_text(img, text, org=(30,40), scale=1.0, color=(0, 255, 0), thick=2):
    # img - frame being drawn on
    # text - text being displayed
    # org - coordinates for where to place text
    # scale - controls font size
    # color - text color
    # thick - controls stroke width of text
    # cv2.LINE_AA: enables anti-aliasing for smoother text edges
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # Focal length = ( width(pixels) * distance(cm) ) / Width(cm)
    # Distance = ( W(cm) * Focal length ) / width(cm)

def run_frame():
    # Start video capture from default webcam 
    cap = cv2.VideoCapture(0)

    # Check that the camera opened correctly
    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    while True:
        # Read single frame from webcam
        success, frame = cap.read()

        # Break if camera reading fails
        if not success: 
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        cv2.imshow("Frame", frame)

        # Wait 1 millisecond for a key press
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Release webcam to free system resources
    cap.release()

    # Close all openCV windows
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    run_frame()