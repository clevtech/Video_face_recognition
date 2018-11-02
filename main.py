import face_recognition
import cv2
import numpy as np 

def recognize_faces():
    video_capture = cv2.VideoCapture(0) # Get a reference to the webcam #0 (default)
    known_face_encodings = np.load("stored_face_encodings.npy") # load a numpy array with known face ecnodings
    known_face_names = np.load("stored_face_names.npy") # load a numpy array with known faces' labels 
    live_face_locations = [] # face loactions from a video input
    live_face_encodings = [] # face encodings from a video input
    face_names = [] # list for displaying at the new face
    live_frame = True # boolean variable for structure control
    # while strcuture that will process every other frame from webcam till 'q' key is gonna be pressed
    while True:
        ret, frame = video_capture.read() # Proccess each single frame of the video input
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # Resize each frame to 0.25 of the original image
        rgb_small_frame = small_frame[:, :, ::-1] # Convert the frame from BGR (openCV) to RGB (face_recongition)
        # Only process every other frame of video to save time
        if live_frame:
            # Find all the faces and face encodings in the given frame of video
            live_face_locations = face_recognition.face_locations(rgb_small_frame) # Find face locations
            live_face_encodings = face_recognition.face_encodings(rgb_small_frame, live_face_locations) # Find face encodings
            face_names = [] # A list for storing names
            for face_encoding in live_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.54) # See if the face is a match for the known face(s)
                name = "Unknown" # Default name is unknown
                # If a match was found in the previous step, extract the first one
                if True in matches: 
                    first_match_index = matches.index(True) # First match found in the previous step
                    name = known_face_names[first_match_index] # First match's label is gonna be the name
                face_names.append(name) # Append the name from previous step to face_names list
        live_frame = not live_frame 
        # Display the results
        for (top, right, bottom, left), name in zip(live_face_locations, face_names):
            # Scale up the small frame since previously it was scaled down to 1/4 of the original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Create a box around the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED) # Display a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX # Choose a font
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) # Display a text
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Press 'q' on the keyboard to exit the programm
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release() # Release webcam
    cv2.destroyAllWindows() # Close all related windows

recognize_faces()