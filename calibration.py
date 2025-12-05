import numpy as np
import cv2
import glob
import os


# CONFIGURATION
taille_carre_mm = 25.0  
dims_damier = (9, 6)    
data = r"C:\Users\elmou\OneDrive\Desktop\Master VMI FA\cours\seqvideo\TP\ImageCalibration\data"
path = r"C:\Users\elmou\OneDrive\Desktop\Master VMI FA\cours\seqvideo\TP\ImageCalibration"
fichier_yaml = "calibration_data.yaml"

# TRAITEMENT 
objp = np.zeros((dims_damier[0] * dims_damier[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dims_damier[0], 0:dims_damier[1]].T.reshape(-1, 2)
objp = objp * taille_carre_mm 

objpoints = [] 
imgpoints = [] 

pattern = os.path.join(data, "*.jpg")
images = glob.glob(pattern)

print(f"Images trouvées : {len(images)}")

img_shape = None
first_img = None

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
    img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, dims_damier, None)

    if ret:
        if first_img is None: first_img = img.copy()
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        print(".", end="")
    else:
        print("x", end="")

print("Calibration en cours...")

# CALIBRATION
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    print(f"ERREUR RMS: {ret:.4f}")

    #  SAUVEGARDE EN YAML 
    cv_file = cv2.FileStorage(fichier_yaml, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    cv_file.release()
    print(f"Calibration sauvegardée dans : {fichier_yaml}")


    
    fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, img_shape, 4.8, 3.6)
    focale_m = focalLength / 1000.0
    print(f"Focale calculée : {focalLength:.2f} mm")
    print(f"FOV : {fovx:.2f} degrés")
    print(f"FOCALE EN MÈTRES  : {focale_m:.6f} m ")
    

    # Fig 1 : Detection
    img_detect = first_img.copy()
    ret_b, corners_b = cv2.findChessboardCorners(cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY), dims_damier, None)
    cv2.drawChessboardCorners(img_detect, dims_damier, corners_b, ret_b)
    cv2.imencode(".jpg", img_detect)[1].tofile(os.path.join(path, "DETECTION.jpg"))
    
    # Fig 2 : Correction
    h, w = first_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(first_img, mtx, dist, None, newcameramtx)
    cv2.imencode(".jpg", dst)[1].tofile(os.path.join(path, "CORRECTION.jpg"))


else:
    print("Pas assez de points pour calibrer.")