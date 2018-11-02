import face_recognition
import cv2
import numpy as np 
import glob
import os

# a function to get face encodings from photos and then store them into numpy arrays
def get_face_encodings(photos_folder):
    stored_face_encodings = [] # create an empty list for face encodings
    stored_names = [] # create an empty list for face labels
    for img in os.listdir(photos_folder): # for every image in folder do
        img_path = photos_folder + "/" + img # store path for each image
        user_img = face_recognition.load_image_file(img_path) # load an image
        user_face_enc = face_recognition.face_encodings(user_img)[0] # extract face encodings 
        user_name = os.path.splitext(img)[0] # get a label from an image
        stored_names.append(user_name) # store the name to the list
        stored_face_encodings.append(user_face_enc) # store the face ecnodings to the list

    np.save("stored_face_encodings", stored_face_encodings) # save all face ecnodings into a numpy array file
    np.save("stored_face_names", stored_names) # save all face labels to a numpy array file


# # a function 
# def load_stored_data(storage_folder):
#     # Create arrays for faces and their names
#     known_face_encodings = []
#     known_face_names = []

#     k = 1
#     for facefile in glob.glob(storage_folder):
#         unit = 'user' + str(k)
#         globals()["unit"] = np.load(facefile)
#         known_face_encodings.append(globals()["unit"])
#         known_face_names.append(unit)
#         k = k + 1

#     print(known_face_encodings)
#     print(known_face_names)
#     return known_face_encodings, known_face_names


# def save_last_data(encodings, names):
#     np.save("stored_face_enc.npy", encodings)
#     np.save("stored_face_names.npy", names)


# face_encs, face_names = load_stored_data("/home/aldos/Desktop/recognizer_aplha_1.0/*.npy")
# save_last_data(face_encs, face_names)
get_face_encodings("/home/aldos/Desktop/recognizer_alpha_1.0/photos")