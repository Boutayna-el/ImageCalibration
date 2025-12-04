import numpy as np
import cv2
import glob
import os

# CONFIGURATION 
taille_carre_mm = 24.0
dims_damier = (9, 6)
data= r"C:\Users\elmou\OneDrive\Desktop\Master VMI FA\cours\seqvideo\TP\data"
path = r"C:\Users\elmou\OneDrive\Desktop\Master VMI FA\cours\seqvideo\TP"
fichier_yaml = "calibration_data.yaml"


cv_file = cv2.FileStorage(fichier_yaml, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
cv_file.release()

print("Calibration chargée depuis le YAML.")
print("Matrice K chargée :", mtx)

# POSE ESTIMATION 
pattern = os.path.join(data, "*.jpg")
images = glob.glob(pattern)

objp = np.zeros((dims_damier[0] * dims_damier[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dims_damier[0], 0:dims_damier[1]].T.reshape(-1, 2)
objp = objp * taille_carre_mm


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * taille_carre_mm

for fname in images:
    try:
        with open(fname, "rb") as f:
            bytes = bytearray(f.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except:
        continue
    
    if img is None: continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, dims_damier, None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

        corner = tuple(map(int, corners2[0].ravel()))
        img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0,255,0), 5) 
        img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0,0,255), 5) 

        output_path = os.path.join(path, "AXES_3D.jpg")
        cv2.imencode(".jpg", img)[1].tofile(output_path)
        
        break 