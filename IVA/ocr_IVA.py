import cv2
import easyocr
import os
from pdf2image import convert_from_path
import math
import re
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import trigger_t_iva

URL='http://172.30.82.190:8220/petersen/api/v1/calificaciones/services/public/calificacion/venta/respuestaMS/'



# Importa la clase SQLConnector desde el archivo database_connector.py
#from connection_IVA_BBDD import SQLConnector
ROOT_FOLDER = os.path.join('E:\MLPA')

def delete_millis(date):
    # Convierte la cadena en un objeto datetime
    date  = date .split('.')[0]  # Elimina los milisegundos
    date_obj  = datetime.strptime(date , '%Y-%m-%d %H:%M:%S')
    
    # Convierte el objeto datetime nuevamente en una cadena en el formato deseado
    Date_ret = date_obj .strftime('%Y-%m-%d %H:%M:%S')
    
    return Date_ret
def Date_IVA(text):
    # Patrón de títulos en mayúsculas
    pattern = r'Fecha de Presentación: (\d{2}/\d{2}/\d{4})'
    # Buscar el patrón en el texto
    result = re.search(pattern, text)
    if result:
        # Extraer y devolver el CUIT encontrado
        date_found = result.group(1)
        print("fecha encontrada: ",date_found)
        date_extr = date_found.replace("-","")
        return date_extr
    else:
        return None
def val_format_810(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bF810|F.810|F 810\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
    
def val_format_2002(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bF.2002|F 2002|2002|F. 2002|F2002\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
def val_formato_731(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bF.731|F 731|F731\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
def val_iva(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bTotal\s+d[ec]l\s+d[ecé]bito\s+[ft][lií]sc[aá]l\s+d[ec]l\s+p[eéc]r[lií][oó]d[oó]|Total del Débito Fiscal|Total del débito fiscal del período|Total del débito tiscal del periodo|Total del débito tiscal del período|del débito fiscal clel periodo|el débito fiscal clel periodo|Total dcl débito fisca] del pcrodo\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
def val_period_810(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bp[eéc]r[lií][oó]d[oó]\s+[ft][lií]sc[aá]l|p[eéc]r[lií][oó]do[_\s][ft][lií]sc[aá]l|Pcríodo tiscal|Pcriodo tiscal\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
def val_period_731(text):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r'\bp[eéc]r[lií][coó]d[coó]\s+[ft][lií]sc[caá]l|p[eéc]r[lií][coó]do[_\s][ft][lií]sc[caá]l|Pcríodo tiscal|Pcriodo tiscal\b', re.IGNORECASE)

    if pattern.search(text):
        return True
    else:
        return False
    
def val_period_2002(text,confidence):
    # Patrón de expresión regular insensible a mayúsculas/minúsculas y tildes
    pattern = re.compile(r"p[eéc]r[lií][coó]d[coó][:;]\s*(\d+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        # Extraer el número del grupo capturado
        period = match.group(1)
        if confidence > 0.6:
            return period
        else:
            return period
    else:
        return None
    
def contains_cuit_810(text):
    # Patrón para buscar el cuit
    pattern = r'\s*CUIT\s*[:;]?\s*(\d{2}-\d{8}-?\d{1})'
    
    # Buscar el patrón en el texto
    result = re.search(pattern, text,re.IGNORECASE)
    
    # Verificar si se encontró la palabra "cuit"
    if result:
        # Extraer y devolver el CUIT encontrado
        cuit_found = result.group(1)
        print("cuit encontrado: ",cuit_found)
        cuit_extr = cuit_found.replace("-","")
        print("cuit extraido: ",cuit_extr)
        return cuit_extr
    else:
        return None
    
def contains_cuit_2002(text):
    # Patrón para buscar el cuit
    pattern = r'CUIT\s*(?:N[Cºo\'\"]|NP|N |N\?)[:;]?\s*(\d{11})'
    # Buscar el patrón en el texto
    result = re.search(pattern, text,re.IGNORECASE)
    
    # Verificar si se encontró la palabra "cuit"
    if result:
        # Extraer y devolver el CUIT encontrado
        cuit_found = result.group(1)
        print("cuit encontrado: ",cuit_found)
        cuit_extr = cuit_found.replace("-","")
        print("cuit extraido: ",cuit_extr)
        return cuit_extr
    else:
        return None

def initialize_easyocr():
    print("Cargando modelo")
    reader = None
    contador =0
    while reader is None and contador < 5:
        try:
            reader = easyocr.Reader(['es', 'en'], gpu=True)
            print("Modelo EasyOCR cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar EasyOCR: {e}. Reintentando en 5 segundos...")
            contador = contador + 1
            
    return reader
def convertir_fecha(yyyymm: str) -> str:
    fecha = datetime.strptime(yyyymm, "%Y%m")  # Convertir a datetime
    return fecha.strftime("%Y-%m-01")

def run(id_entrada, sql_connector, reader):    
    contador = 0
    # Obtener la ubicación actual
    
    # Ruta a la carpeta que contiene los archivos PDF
    carpeta_pdf =os.path.join(ROOT_FOLDER, 'IVA','testing')
    
    print (carpeta_pdf)

    with os.scandir(carpeta_pdf) as carpetas_bancos:
        for fichero in carpetas_bancos:
            carpeta_banco = os.path.join(carpeta_pdf, fichero.name,'IN')
            carpeta_salida =os.path.join(carpeta_pdf, fichero.name,'OUT')
            # Obtener la lista de archivos en la carpeta
            archivos_en_carpeta = os.listdir(carpeta_banco)
            entidad = fichero.name

            # Obtener el tamaño del bucle (número de archivos)
            num_cuentas = len(archivos_en_carpeta)
            #número de columnas base de datos: 'CUIT','Nombre_documento','Periodo','Concepto', 'Neto Gravado','Confianza','Formato','Estado','Fecha_proceso'
            col_cuentas = 12
            #pdf_IVA procesados, inicializamos con valores none, esto permite almancenar distintos tipos de variables
            resultados_cuentas =np.empty((num_cuentas, col_cuentas), dtype=object)
            dim = resultados_cuentas.shape
            dim_str = str(dim)
            print("tamaño cuentas"+dim_str)

            # Crea una instancia de SQLConnector
            #sql_connector = SQLConnector(server, database, user, password)

            # Conecta a la base de datos
            #sql_connector.connect()

            # Iterar sobre los archivos PDF en la carpeta
            for pdf_IVA_index,archivo_pdf in enumerate(archivos_en_carpeta):
                contador = contador+1
                if archivo_pdf.endswith(".PDF") or archivo_pdf.endswith(".pdf"):
                    pdf_path = os.path.join(carpeta_banco, archivo_pdf)
                    pdf_path_out = os.path.join(carpeta_salida, archivo_pdf)
                    print(f"Procesando archivo: {pdf_path}")
                    print(f"index: {pdf_IVA_index}")
                    nombre_doc = archivo_pdf.replace(".pdf","")
                try:
                    resultados_cuentas[pdf_IVA_index][3]=nombre_doc
                    resultados_cuentas[pdf_IVA_index][1]=entidad
                    fecha_proc = str(datetime.now())
                    id_ejecucion = id_entrada
                    resultados_cuentas[pdf_IVA_index][0]=id_ejecucion
                    #eliminamos los milisegundos
                    resultados_cuentas[pdf_IVA_index][11]=delete_millis(fecha_proc)
                    # Convertir el PDF a una lista de imágenes
                    pages = convert_from_path(pdf_path)
                    #cantidad de paginas del documento
                    num_pages=len(pages)
                    #se usa esta variable para validar si ya se leyó el valor de iva y con eso no seguir buscando(ya que el doc puede tener varias hojas)
                    Iva_leido=False
                    #print(f"cantidad de paginas documento:{len(pages)}:")
                    #if(pdf_IVA_index==1):
                    #    break
                    # Crear un diccionario para almacenar los resultados por página
                    resultados_por_pagina = {}
                    estado_python=''
                    I_sub_estado_proceso=''
                    # Procesar cada página como una imagen
                    for index_page, page in enumerate(pages):
                        
                        #condicion para no evaluar todo el documento
                        if(Iva_leido==True):
                            break
                        # Guardar la imagen como archivo temporal
                        temp_image_path = f'pagina_{index_page}.jpg'
                        page.save(temp_image_path, 'JPEG')

                        # Leer la imagen utilizando cv2
                        img = cv2.imread(temp_image_path)
                        
                        
                        # Procesar la imagen con pytesseract

                        # Imprimir los resultados
                        print(f"Resultados de OCR en la página {index_page + 1}:")
                        #resultados_por_pagina=pytesseract.image_to_string(img, lang='spa')
                        #result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='spa')
                        
                        #print(result)
                        #print(resultados_por_pagina)
                        
                        # Detectar líneas de texto en la imagen utilizando HoughLinesP de OpenCV
                        #normalization and remove Noise 
                        norm_img   = np.zeros((img.shape[0], img.shape[1]))
                        image      = cv2.normalize(img, norm_img, -125, 300, cv2.NORM_MINMAX)
                        #image to gray and binarization
                        gray_img   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # Detectar líneas de texto en la imagen utilizando HoughLinesP de OpenCV
                        gray_img   = cv2.bitwise_not(gray_img)
                        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
                        lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=100, minLineLength=100, maxLineGap=5)

                        # Calcular el ángulo promedio de las líneas detectadas
                        total_angle = 0
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                            total_angle += angle

                        average_angle = total_angle / len(lines)
                        print(average_angle)
                        # Determinar si la imagen tiene texto en sentido vertical
                        if average_angle > 80 or average_angle< 0:
                            # Rotar la imagen para poner el texto en sentido horizontal
                            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            img = rotated_img
                            print(f"CORRECCIÓN DE DIRECCCIÓN página {index_page + 1}:")
                        # Ajustar el tamaño de la ventana de visualización al tamaño de la imagen
                        #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                        #cv2.resizeWindow("Image", img.shape[1], img.shape[0])
                        if average_angle > 45 and average_angle< 80:
                            # Rotar la imagen para poner el texto en sentido horizontal
                            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            img = rotated_img
                            print(f"CORRECCIÓN DE DIRECCCIÓN ROTATE_90_CLOCKWISE página {index_page + 1}:")
                        # Procesar la imagen con EasyOCR una vez corregido el sentido de la imagen si es el caso
                        resultados_por_pagina = reader.readtext(img, paragraph=False)

                        print(f"Resultados de OCR en la página {index_page + 1}:")
                        resultados_pagina_actual = []
                        # Crear una lista para almacenar las líneas de texto

                        columnas_texto = []
                        fechas_cuentas = []
                        lineas_texto = []
                        tolerancia_horizontal = 150
                        tolerancia_vertical = 60  # Tolerancia para agrupar textos en la misma línea
                        agrupado=False
                        Format=''
                        
                        next_page=False
                        for detection in resultados_por_pagina:
                            text = detection[1]
                            confidence = detection[2]
                            print(f"Texto: {text}, Confianza: {confidence}")
                            # Agregar el texto y la confianza como una tupla a la lista de resultados de la página actual
                            resultados_pagina_actual.append((text, confidence))
                            
                            #if(agrupado):
                            #    resultados_cuentas.append(text)
                            if val_format_810(text):
                                resultados_cuentas[pdf_IVA_index][8]='810'
                                Format='810'
                            if val_format_2002(text):
                                resultados_cuentas[pdf_IVA_index][8]='2002'
                                Format='2002'
                            if val_formato_731(text):
                                resultados_cuentas[pdf_IVA_index][8]='731'
                                Format='731'

                            x_left = detection[0][0][0]  # Coordenada x del centro del cuadro del texto
                            x_right = detection[0][1][0]  # Coordenada x del centro del cuadro del texto
                            agrupado = False
                            for columna in columnas_texto:
                                columna_x_center = sum(det[0][0][0] for det in columna) / len(columna)
                                columna_x_right = sum(det[0][1][0] for det in columna) / len(columna)
                                
                                if abs(x_left - columna_x_center) <= tolerancia_horizontal or (abs(x_right - columna_x_right)<= tolerancia_horizontal):
                                    columna.append(detection)
                                    
                                    agrupado = True
                                    break
                            if not agrupado:
                                columnas_texto.append([detection])
                        
                        if (Format== ''): 
                            print(f"Formato ::::{Format}")   
                            if(num_pages > 1):
                                next_page=True
                            else:  
                                Format='731'
                        #print columnas
                        print("Columnas encontradas:")
                        for i, columna in enumerate(columnas_texto):
                            print(f"Columna {i + 1}:")
                            if next_page==True:
                                break
                            if Format =='810':
                                for j, detection in enumerate(columna):
                                    text = detection[1]
                                    confidence = detection[2]
                                    print(f"Texto: {text}, Confianza: {confidence}")
                                    #extraer cuit
                                    if contains_cuit_810(text):
                                        resultados_cuentas[pdf_IVA_index][2]=contains_cuit_810(text)
                                    #extraer periodo
                                    if val_period_810(text):
                                        numero_ext = str(columnas_texto[i][j+1][1]).replace("/","")
                                        numero_ajustado = numero_ext[2:] + numero_ext[:2]
                                        resultados_cuentas[pdf_IVA_index][4]=numero_ajustado
                                    if val_iva(text) and j < len(columna) - 1:
                                        resultados_cuentas[pdf_IVA_index][5]=("Total del débito fiscal del período")
                                        for k, otra_columna in enumerate(columnas_texto):
                                            if k != i:
                                                for col,det in enumerate(otra_columna):

                                                    if (abs(detection[0][0][1] - det[0][0][1]) <= 15 or
                                                        abs(detection[0][1][1] - det[0][1][1]) <= 15):
                                                        print(f"index : {col}, inf:{det}")
                                                        result_IVA = det[1]
                                                        result_IVA_proc = result_IVA.replace("$ ","")
                                                        result_IVA_proc = result_IVA.replace(".","")
                                                        resultados_cuentas[pdf_IVA_index][6]=(result_IVA_proc)
                                                        resultados_cuentas[pdf_IVA_index][7]=round(det[2], 3)
                                                        print(f"Valor resultado confianza leido :{round(det[2], 3)}")
                                                        Iva_leido=True
                                                        #fechas_cuentas.append(det[1])
                                print()
                            if Format =='2002':
                                for j, detection in enumerate(columna):
                                    text = detection[1]
                                    confidence = detection[2]
                                    print(f"Texto: {text}, Confianza: {confidence}")
                                    #extraer cuit
                                    if contains_cuit_2002(text):
                                        print(f"el valor CUIT de texto es ::::::: {text}")
                                        resultados_cuentas[pdf_IVA_index][2]=contains_cuit_2002(text)
                                    #extraer periodo    
                                    if val_period_2002(text,confidence):
                                        numero_ext = str(val_period_2002(text,confidence)).replace("/","")
                                        #print(f"el valor leido de texto es ::::::: {text}")
                                        resultados_cuentas[pdf_IVA_index][4]=numero_ext
                                    if val_iva(text) and j < len(columna) - 1:
                                        resultados_cuentas[pdf_IVA_index][5]=("Total del débito fiscal del período")
                                        for k, otra_columna in enumerate(columnas_texto):
                                            if k != i:
                                                for col,det in enumerate(otra_columna):

                                                    if (abs(detection[0][0][1] - det[0][0][1]) <= 15 or
                                                        abs(detection[0][1][1] - det[0][1][1]) <= 15):
                                                        print(f"index : {col}, inf:{det}")
                                                        result_IVA = det[1]
                                                        result_IVA_proc = result_IVA.replace("$ ","")
                                                        #result_IVA_proc = result_IVA.replace(".","")
                                                        resultados_cuentas[pdf_IVA_index][6]=(result_IVA_proc)
                                                        resultados_cuentas[pdf_IVA_index][7]=round(det[2], 3)
                                                        print(f"Valor resultado confianza leido :{round(det[2], 3)}")
                                                        Iva_leido=True
                                                        #fechas_cuentas.append(det[1])
                                print() 
                            if Format =='731':
                                for j, detection in enumerate(columna):
                                    text = detection[1]
                                    confidence = detection[2]
                                    print(f"Texto: {text}, Confianza: {confidence}")
                                    print(f"...............................{contains_cuit_810(text)}")
                                    #extraer cuit
                                    if contains_cuit_810(text):
                                        resultados_cuentas[pdf_IVA_index][2]=contains_cuit_810(text)
                                    #extraer periodo    
                                    if val_period_731(text):
                                        numero_ext = str(columnas_texto[i][j+1][1]).replace("-","")
                                        numero_ajustado = numero_ext[2:] + numero_ext[:2]
                                        if (numero_ajustado.isdigit()):
                                            resultados_cuentas[pdf_IVA_index][4]=numero_ajustado
                                        else:
                                            numero_ext = str(columnas_texto[i][j+2][1]).replace("-","")
                                            numero_ajustado = numero_ext[2:] + numero_ext[:2]

                                            if (numero_ajustado.isdigit()):
                                                resultados_cuentas[pdf_IVA_index][4]=numero_ajustado
                                    
                                        print(f"el valor leido de texto es ::::::: {text}")
                                    if val_iva(text) and j < len(columna) - 1:
                                        resultados_cuentas[pdf_IVA_index][5]=("Total del débito fiscal del período")
                                        for k, otra_columna in enumerate(columnas_texto):
                                            if k != i:
                                                for col,det in enumerate(otra_columna):

                                                    if (abs(detection[0][0][1] - det[0][0][1]) <= 15 or
                                                        abs(detection[0][1][1] - det[0][1][1]) <= 15):
                                                        print(f"index : {col}, inf:{det}")
                                                        result_IVA = det[1]
                                                        result_IVA_proc = result_IVA.replace("$ ","")
                                                        result_IVA_proc = result_IVA.replace(".","")
                                                        resultados_cuentas[pdf_IVA_index][6]=(result_IVA_proc)
                                                        resultados_cuentas[pdf_IVA_index][7]=round(det[2], 3)
                                                        print(f"Valor resultado confianza leido :{round(det[2], 3)}")
                                                        Iva_leido=True
                                                        #fechas_cuentas.append(det[1])
                                print()
                        # Definir los nombres de las columnas
                        column_names = ['id_ejecucion','Banco','CUIT','Nombre Documento','Periodo','Concepto', 'Valor_IVA_leido','confianza','formato','estado','observacion','fecha_creacion']
                        #insertar datos BBDD
                        if (Iva_leido==True or (index_page+1==num_pages)):
                            I_id_ejecucion = id_ejecucion
                            I_Banco = resultados_cuentas[pdf_IVA_index][1]
                            I_Cuit= resultados_cuentas[pdf_IVA_index][2]
                            I_Nombre_documento= str(resultados_cuentas[pdf_IVA_index][3])
                            I_Fecha_IVA= str(resultados_cuentas[pdf_IVA_index][4])
                            I_Valor_IVA_leido= str(resultados_cuentas[pdf_IVA_index][6]).replace(",",".")
                            I_Valor_IVA_leido_validacion=str(I_Valor_IVA_leido).replace(".","")
                            I_Confianza= resultados_cuentas[pdf_IVA_index][7]
                            I_Formato=resultados_cuentas[pdf_IVA_index][8] 
                            I_FechaPro =resultados_cuentas[pdf_IVA_index][11]
                            I_Observacion=''
                            print(f"Cuit leido es.......................{I_Cuit}")
                            if(I_Confianza==None):
                                I_Confianza=0
                            if(I_Formato==None):
                                    I_estado ='F'
                                    I_Formato='0'
                                    I_sub_estado_proceso ='ERROR_LECTURA'
                                    I_Observacion='No se pudo identificar el formato del documento, validar si el documento es correcto'
                            elif(I_Cuit!=None):
                                I_estado ='A'
                                if(I_Cuit==I_Nombre_documento[:11]):

                                    if I_Fecha_IVA==None or not (I_Fecha_IVA.isdigit()):
                                        I_sub_estado_proceso ='CUIT'
                                        I_Fecha_IVA = '0'
                                        if( I_Valor_IVA_leido!=None and I_Confianza>0.6):
                                            estado_python='CUIT_IVA'
                                        else:
                                            if not I_Valor_IVA_leido_validacion.isdigit():
                                                I_Valor_IVA_leido='-1'
                                            estado_python='CUIT' 
                                    else:
                                        I_sub_estado_proceso ='CUIT_PERIODO'
                                        """I_sub_estado_proceso ='CUIT'"""
                                        if( I_Valor_IVA_leido!=None and I_Confianza>0.6):
                                            estado_python='CUIT_PERIODO_IVA'
                                        else:
                                            if not I_Valor_IVA_leido_validacion.isdigit():
                                                I_Valor_IVA_leido='-1'
                                            estado_python='CUIT_PERIODO'
                                else:
                                    if (I_Fecha_IVA==None) or not (I_Fecha_IVA.isdigit()):
                                        I_Fecha_IVA = '0'
                                        I_estado ='F'
                                        I_Cuit=I_Nombre_documento[:11]
                                        if not (I_Cuit.isdigit()):
                                            I_Cuit = '0'
                                        I_sub_estado_proceso ='ERROR_LECTURA'
                                        I_Observacion='No se pudo validar el cuit y el periodo del documento,validar.'
                                    else:
                                        I_estado ='A'
                                        I_sub_estado_proceso ='PERIODO'
                                        if( I_Valor_IVA_leido!=None and I_Confianza>0.6):
                                            estado_python='PERIODO_IVA'
                                        else:
                                            if not I_Valor_IVA_leido_validacion.isdigit():
                                                I_Valor_IVA_leido='-1'
                                            estado_python='PERIODO'
                            elif(I_Cuit==None):
                                I_Cuit=I_Nombre_documento[:11]
                                if not (I_Cuit.isdigit()):
                                    I_Cuit = '0'
                                
                                if I_Fecha_IVA==None or not (I_Fecha_IVA.isdigit()):
                                    I_Fecha_IVA = '0'
                                    I_estado ='F'
                                    I_sub_estado_proceso ='ERROR_LECTURA'
                                    I_Observacion='No se pudo obtener el cuit y el el periodo del documento,validar.'
                                    estado_python=''
                                else:
                                    I_estado ='A'
                                    I_sub_estado_proceso ='PERIODO'
                                    if( I_Valor_IVA_leido!=None and I_Confianza>0.6):
                                        estado_python='PERIODO_IVA'
                                    else:
                                        if not I_Valor_IVA_leido_validacion.isdigit():
                                            I_Valor_IVA_leido='-1'
                                        estado_python='PERIODO'

                            if(estado_python=='CUIT'):
                                I_Observacion='No se encontró periodo e iva.'
                            elif(estado_python=='CUIT_PERIODO'):
                                I_Observacion='No se pudo leer el valor de iva.'
                            elif(estado_python=='CUIT_IVA'):
                                I_Observacion='No se pudo leer el periodo.'
                            elif(estado_python=='CUIT_PERIODO_IVA'):
                                I_Observacion=''
                            elif(estado_python=='PERIODO'):
                                I_Observacion='No se encontró cuit e iva.'
                            elif(estado_python=='PERIODO_IVA'):
                                I_Observacion='No se pudo extraer el cuit'
                                    
                            resultados_cuentas[pdf_IVA_index][10]=I_Observacion
                            resultados_cuentas[pdf_IVA_index][9]=I_estado
                            
                            #if(I_estado =='A' or I_estado =='C'):
                            print(f"-------------------")
                            print(f"{I_Valor_IVA_leido_validacion}")
                            if I_Valor_IVA_leido == 'None' or I_Valor_IVA_leido is None:
                                I_Valor_IVA_leido =0
                                print(f"{I_Valor_IVA_leido}")

                            if I_Cuit == 'None' or I_Cuit is None:
                                I_Cuit =0
                                print(f"{I_Cuit}")

                            if I_Fecha_IVA == 'None' or I_Fecha_IVA is None:
                                I_Fecha_IVA =0
                                print(f"{I_Fecha_IVA}")
                            
                            insert_statement = "Insert into T_IVA(id_ejecucion,Banco,CUIT, Nombre_documento, Periodo, Valor_IVA_leido, Confianza, formato,estado,sub_estado_proceso,observacion,fecha_creacion) " \
                                            f"values({I_id_ejecucion},'{I_Banco}',{I_Cuit}, '{I_Nombre_documento}',{I_Fecha_IVA}, {I_Valor_IVA_leido}, {I_Confianza},{I_Formato},'{I_estado}','{I_sub_estado_proceso}','{I_Observacion}',CONVERT(datetime, '{I_FechaPro}', 120));"
                            insert_statement_historico = "Insert into T_IVA_H(id_ejecucion,Banco,CUIT, Nombre_documento, Periodo, Valor_IVA_leido, Confianza, formato,estado,sub_estado_proceso,observacion,fecha_creacion) " \
                                            f"values({I_id_ejecucion},'{I_Banco}',{I_Cuit}, '{I_Nombre_documento}',{I_Fecha_IVA}, {I_Valor_IVA_leido}, {I_Confianza},{I_Formato},'{I_estado}','{I_sub_estado_proceso}','{I_Observacion}',CONVERT(datetime, '{I_FechaPro}', 120));"

                            data_workflow={
                                "idSolicitudVentas":0,
                                "observacion":f"{I_Observacion}",
                                "proceso":"A",
                                "status":f"{I_estado}"}

                            headers ={
                                "Content-Type":"application/json",
                                "accept":"*/*"}

                            url_workflow = URL + I_Nombre_documento
                            ####################################
                            #descomentar cuando se corra en local

                            sql_connector.insert_data(insert_statement_historico)
                            sql_connector.insert_data(insert_statement)
                            
                            fecha_iva_completa=convertir_fecha(I_Fecha_IVA)
                            trigger_t_iva.trigger_logic(I_id_ejecucion,I_sub_estado_proceso,I_estado,I_Cuit,I_Fecha_IVA,I_Valor_IVA_leido,I_Confianza,
                                                        I_FechaPro,I_Banco,I_Nombre_documento,I_Formato,I_Observacion,fecha_iva_completa)




                            response = requests.post(url_workflow, json=data_workflow, headers=headers)
                            print(data_workflow)
                            print(url_workflow)
                            print(response)
                            print(estado_python)
                            ##############################################
                            I_Banco = None
                            I_Cuit= None
                            I_Nombre_documento= None
                            I_Fecha_IVA= None
                            I_Valor_IVA_leido= None
                            I_Confianza= None
                            I_Formato= None 
                            I_FechaPro = None

                            pd.set_option('display.max_columns', None)
                            pd.set_option('display.max_rows', None)
                            pd.set_option('display.width', None)
                            pd.set_option('display.max_colwidth', None)
                            # Crear un DataFrame a partir de la lista y los nombres de las columnas
                            df = pd.DataFrame(resultados_cuentas, columns=column_names)

                            # Mostrar el DataFrame
                            print(df)

                            #Mover archivo
                            os.replace(pdf_path,pdf_path_out)

                except Exception as e:
                    print("error al abrir el archivo")
                    data_workflow={
                                "idSolicitudVentas":0,
                                "observacion":"Error al leer el documento",
                                "proceso":"F",
                                "status":"F"}

                    headers ={
                                "Content-Type":"application/json",
                                "accept":"*/*"}

                    url_workflow = URL + nombre_doc

                    print(data_workflow)
                    print(url_workflow)
                    response = requests.post(url_workflow, json=data_workflow, headers=headers)
                    print(response)
                    
                    #Mover archivo
                    os.replace(pdf_path,pdf_path_out)
    return contador
