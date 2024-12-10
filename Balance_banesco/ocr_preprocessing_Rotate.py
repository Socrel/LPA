import numpy as np
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage.filters import threshold_otsu, sobel
import cv2
import math

def rotate_image(image, angle):
    # Obtener dimensiones de la imagen original
    (h, w) = image.shape[:2]
    # Calcular el centro de la imagen
    center = (w // 2, h // 2)
    
    # Calcular la matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcular las esquinas de la imagen rotada
    cos_theta = abs(math.cos(math.radians(angle)))
    sin_theta = abs(math.sin(math.radians(angle)))
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))
    
    # Ajustar la matriz de rotación para evitar el recorte
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    # Aplicar la rotación utilizando warpAffine
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    
    return rotated
def binarizeImage(image):
  
  gray_image   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #image = rgb2gray(RGB_image)
  threshold = threshold_otsu(gray_image)
  bina_image = gray_image < threshold
  
  return bina_image
def findEdges(bina_image):
  
  image_edges = sobel(bina_image)
  return image_edges
def check90DegreeRotation(image_edges):
    """ Verifica si la imagen está rotada ±90 grados """
    # Si el ángulo detectado es 0, puede estar rotada en -90 o +90 grados
    # Puedes hacer una detección adicional basada en las características de la imagen
    # Para efectos simples, asumimos que si no hay líneas detectadas, está en 90°
    
    # Análisis simple: ¿más bordes verticales que horizontales?
    vertical_lines = np.sum(np.abs(np.diff(image_edges, axis=1)))  # Diferencias en columnas
    horizontal_lines = np.sum(np.abs(np.diff(image_edges, axis=0)))  # Diferencias en filas
    
    if vertical_lines > horizontal_lines:
        print("La imagen parece estar rotada +90 grados.")
        return 90
    elif horizontal_lines > vertical_lines:
        print("La imagen parece estar rotada -90 grados.")
        return -90
    else:
        print("El ángulo parece correcto en 0 grados.")
        return 0
    
def findTiltAngle(image_edges):
  
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0])
  print(f"angulo de la imagen",{angle})

  if angle == 0:
        print("No se detectó ningún ángulo, podría estar rotada ±90 grados.")
        # Verificamos si la imagen está rotada -90 o +90 grados
        #angle = check90DegreeRotation(image_edges)
        angle=0
  elif (angle < 0):
        angle = angle + 90
  else:
        angle += 90

  return angle  
def rotateImage(image, angle):
  
  rotated_image=rotate_image(image, angle)
  return rotated_image
#funcion principal que llama a las anteriores funciones para corregir el angulo de orientacion del texto
def rotate_page_pdf(image):
    #image       = cv2.imread(path_image)
    # Binarize Image
    img_bina    = binarizeImage(image)
    # Find Edges
    image_edges = findEdges(img_bina)
    # Find Tilt Angle
    angle       = findTiltAngle(image_edges)
    # Rotate Image
    rotated_img = rotateImage(image, angle)

    return rotated_img

def correction_page_pdf(image):
    #normalization and remove Noise 
    norm_img   = np.zeros((image.shape[0], image.shape[1]))
    image      = cv2.normalize(image, norm_img, -125, 300, cv2.NORM_MINMAX)
    #image to gray and binarization
    gray_img   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Desenfoque Gaussiano
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # 3. Adaptive Thresholding
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,3)
    # 4. Erosión y dilatación
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(binary_adaptive, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    #cv2.imwrite("pagina procesada_dil.jpg", dilation)
    processed_image =binary_adaptive
    return processed_image