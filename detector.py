import cv2
#import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from collections import deque

# Focal length = ( width(pixels) * distance(cm) ) / KnownWidth(cm)
# Distance = ( KnownWidth(cm) * Focal length ) / width(pixels)

# Known distance from camera to face (cm)
known_distance = 50

# Known width of face IRL (cm)
known_width = 14

# Function to calculate focal length 
def find_focal_length(measured_distance, real_width, width_in_frame):
    # measured distance - distance from camera to face in cm (known_distance)
    # real_width - width of face in real world in cm (known_width) 
    # width_in_frame - width of face in the photo frame (pixels)
    focal_length = (width_in_frame * measured_distance) / real_width

    # Return focal length for distance calculation
    return focal_length

# Function to calculate distance
def find_distance(focal_length, real_width, width_in_frame):
    # focal length - focal length in pixels returned from function
    # real_width - width of face in real world in cm (known_width) 
    # width_in_frame - width of face in the photo frame (pixels) 
    distance = (real_width * focal_length) / width_in_frame

    # return distance
    return distance
    
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


# Use OpenCV's Haar Cascade XML files to detect face
face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Create classifier object from XML files
face_cascade = cv2.CascadeClassifier(face_xml)

def face_data(frame):
    face_width = 0 # Start face width at 0
    
    # Convert frame from color (BGR) to grayscale
    # Haar cascades operate faster + more accurately on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face within the grayscale frame
    # detectMultiScale: scans image for objects that match trained face pattern
    # scaleFactor: how much the image size is reduced at each scale
    # minNeighbors: how many overlapping detections are needed to confirm a face
    # minSize: ignores detections smaller than a given size
    faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
        )
    
    # Process only the first detected face
    for(x, y, w, h) in faces[:1]:
        # Draw a rectangle around the detected face
        # parameters: image, top-left corner, bottom-right, color (BGR), thickness
        cv2.rectangle(frame, (x,y), (x + w, y + h), (100, 200, 255), 1)

        # Extract region of interest (ROI) that contains only the face
        # roi_gray = gray[y: y + h, x:x + w]
        # roi_color = frame[y: y + h, x:x + w]
            
        # Get face width in pixels 
        face_width = w

    # Only process the first face to keep the program simple and consistent
    # Return face width in pixels
    return face_width

def run_frame():

    # Start video capture from default webcam 
    cap = cv2.VideoCapture(0)

    # Store face width (pixels) when posture is ideal
    ideal_face_width = 0

    # Set default deviation to 0
    # Deviation is the difference between ideal face width and person's current face width in the frame
    # "perfect" posture - current posture
    deviation = 0

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

        # Default label & color if no face is detected
        label = "N/A"
        color = (0, 0, 255) # red text

        # Get current face width from frame
        width_in_frame = face_data(frame)

        # Calibrate ideal face width based on current face width in frame
        if ideal_face_width == 0:
            put_text(frame, "Sit up straight and press 'c' to calibrate", (30, 40), 1.0, (180, 180, 180), 2)

            if cv2.waitKey(1) & 0xFF == ord("c"):
                if width_in_frame > 0:
                    ideal_face_width = width_in_frame
                    print(f"Calibrated! Received Perfect posture of {ideal_face_width} pixels from camera.")
                else:
                    # Default label & color if no face is detected
                    label = "no face"
                    color = (0, 0, 255) # red text
                    print("Face could not be detected.")

            elif cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            # Ensure face is in frame before calculating deviation
            if width_in_frame == 0:
                    label = "no face detected"
                    color = (0, 0, 255) # red text
            else:
                # Calculate deviation from ideal face width and current width in frame
                # Deviation is negative: smaller distance away from camera -> too close to screen => poor posture
                # Deviation is positive: larger distance from perfect poster -> far from screen, leaning back => poor posture
                # Deviation is zero: perfect posture
                deviation = ideal_face_width - width_in_frame

                # Set a threshold for "bad" posture or slouching (in pixels)
                slouch_threshold = 20

                # If deviation is less than or equal to threshold, posture is good
                if abs(deviation) <= slouch_threshold:
                    label = "PERFECT"
                    color = (0, 255, 0) # green text
                    put_text(frame, f"Posture Status: {label}", (30, 40), 1.0, color, 2)
                    print(f"Face is {width_in_frame} pixels from camera")
                # If deviation is less, posture is bad
                # elif deviation > slouch_threshold:
                #     label = "OKAY"
                #     color = (0, 255, 255) # red text
                #     put_text(frame, f"Posture Status: {label}", (30, 40), 1.0, color, 2)
                elif deviation < -slouch_threshold:
                    label = "BAD"
                    color = (0, 0, 255) # yellow text
                    put_text(frame, f"Posture Status: {label}", (30, 40), 1.0, color, 2)



        # Display the detected expression on the video feed
        put_text(frame, f"Posture Status: {label}", (30, 40), 1.0, color, 2)

        # Display instructions to quit
        # Calculate y pos dynamically so it stays near bottom
        put_text(frame, "Press q to quit", (30, frame.shape[0] - 20), 0.6, (180, 180, 180), 1)

        # Display the frame in a window
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