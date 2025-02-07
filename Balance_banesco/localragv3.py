import torch
import ollama
import os
from openai import OpenAI
import argparse
import json
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import easyocr
import os
from ocr_preprocessing_Rotate import rotate_page_pdf
import cv2
from datetime import datetime
import easyocr
import cv2
import re
import logging

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
import unicodedata
# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

logging.basicConfig(
    filename='errores.log',  # Archivo donde se guardarán los registros
    level=logging.ERROR,     # Nivel de registro (ERROR captura errores y excepciones)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Formato del mensaje
)

#temporal para visualzar pdf
#import matplotlib.pyplot as plt
# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()

    # Normalizar el texto eliminando tildes y caracteres diacríticos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

    # Eliminar puntuación y símbolos no deseados
    text = re.sub(r'[^\w\s]', '', text)  # Quitar puntuación

    # Quitar espacios adicionales
    text = text.strip()
    
    return text
#solo procesa imagenes ya en blanco y negro
def procesar_pagina_con_ocr(image, palabras_clave_regex, alto_region_superior=1000):
    # Definir la región superior
    altura_pagina, ancho_pagina = image.shape
    region_superior = image[0:alto_region_superior, 0:ancho_pagina]  # Región desde el tope hasta alto_region_superior
    # Usar EasyOCR en la región superior
    reader = easyocr.Reader(["es"], gpu=True)
    resultados_por_pagina = reader.readtext(region_superior, paragraph=False)

    # Combinar todo el texto detectado en una sola cadena para buscar con regex
    texto_detectado = " ".join([resultado[1].lower() for resultado in resultados_por_pagina])

    # Normalizar texto detectado (eliminar caracteres no deseados si es necesario)
    texto_detectado = re.sub(r'[^\w\s]', '', texto_detectado)  # Elimina caracteres especiales

    # Buscar cualquier coincidencia de las palabras clave usando regex
    for regex in palabras_clave_regex:
        # Modificar la expresión regular para permitir errores tipográficos
        regex_modificado = regex.replace("situacion", "(situacion|sltuacion)")  # Ejemplo de error tipográfico permitido
        regex_modificado = regex_modificado.replace("resultado", "(resultado|resultados)")

        if re.search(regex_modificado, texto_detectado):
            return True, region_superior  # Retorna la región de la imagen además del resultado

    return False, region_superior
def procesar_cuentas_simple(response_data, categoria_identificada, diccionario_cuentas, categoria_cuenta, orden_inicial, resultados, nombre_cliente, cuit,id_cliente,periodo_archivo,moneda,sql_connector):
    orden_cuenta = orden_inicial
    
    # Dividir las líneas del resultado
    lineas = response_data.strip().split("\n")
    
    for linea in lineas:
        if linea.strip() == "SIN CUENTAS":
            continue  # Si no hay cuentas, pasa al siguiente paso
        
        # Separar los campos usando el delimitador "|"
        try:
            id_balance= consultar_balance(id_cliente,periodo_archivo,moneda,sql_connector)
            if id_balance is not -1:
                cuentas_previas = consultar_cuentas_previas(id_balance,sql_connector)
                if cuentas_previas is not None:
                    for cuenta in cuentas_previas:
                        cuenta_reclasificada, categoria_reclasificada = cuenta
                else:
                    partes = linea.split("|")
                    cuenta = partes[1].split(":")[1].strip()
                    valor = partes[2].split(":")[1].strip()
                    periodo = partes[3].split(":")[1].strip()
                    
                    # Limpiar valor y convertir a número
                    valor = valor.replace(",", "").replace("-", "")
                    
                    # Reclasificación
                    cuenta_reclasificada, categoria_reclasificada = obtener_reclasificacion(cuenta, categoria_cuenta, diccionario_cuentas)
                estado = 0
                diferencia_periodos = 'pendiente'
                tipo_periodo = 'pendiente'
                
                # Añadir la cuenta al resultado
                resultados.append((
                    nombre_cliente,
                    cuit,
                    categoria_cuenta,
                    orden_cuenta,
                    cuenta,
                    cuenta_reclasificada,
                    categoria_reclasificada,
                    valor,
                    periodo,
                    tipo_periodo,
                    diferencia_periodos,
                    estado
                ))
                orden_cuenta += 1
            else:
                print(f"Cliente con id: {id_cliente}, tiene más de un balance para la misma fecha")
                continue
        except (IndexError, ValueError):
            print(f"Línea mal formateada: {linea}")
            continue  # Salta líneas con formato incorrecto
    
    return resultados, orden_cuenta

palabras_clave_regex = [
    r"estado de situación financiera",
    r"estado de situacion",
    r"estado de situacion patrimonial",
    r"estado consolidado de situación financiera",
    r"estado consolidado de resultados y otras utilidades integrales",
    r"estado de situacion financiera",
    r"estado de resultados",
    r"estado de resultados integrales",
    r"estado de resultado",
    r"Balance Sheet",
    r"balance shcet ",
    r"pasivos corrientes", 
    r"estado de situacion"
]

diccionario_cuentas_activos_corrientes =[]
diccionario_cuentas_activos_no_corrientes =[]
diccionario_cuentas_pasivos_corrientes =[]
diccionario_cuentas_pasivos_no_corrientes =[]
diccionario_cuentas_patrimonio=[]
diccionario_cuentas_estado_resultados=[]


#Crea diccionario de variables
diccionario_var_activo_corriente={}
diccionario_var_pasivo_corriente={}
diccionario_var_activo_no_corriente={}
diccionario_var_pasivo_no_corriente={}
diccionario_var_patrimonio={}
diccionario_var_estado_resultados={}
diccionario_resultantes={}


# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=8):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context
   
def ollama_chat(user_input, system_messages, vault_embeddings, vault_content, ollama_model, conversation_history):
  
    conversation_history.append({"role": "user", "content": user_input})
    
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content)
    
    if relevant_context:
        context_str = "\n".join(relevant_context)
        #print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    # Combine system messages into a single system role entry
    combined_system_message = " - ".join(system_messages)
    
    messages = [
        {"role": "system", "content": combined_system_message},
        *conversation_history
    ]
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=20000,
        temperature=0.4,
    )
    #print(conversation_history)
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content


def obtener_reclasificacion(cuenta, categoria, diccionario_cuentas, umbral_similitud=0.9):
    """
    Función para obtener la reclasificación de cuenta y categoría basándose en la similitud de embeddings.

    Parámetros:
    - cuenta (str): Cuenta a comparar.
    - categoria (str): Categoría a comparar.
    - diccionario_cuentas (list of tuples): Lista de tuplas con el formato 
      (cuenta_diccionario, cat_diccionario, reclasificacion, categoria_reclasificacion).
    - umbral_similitud (float): Umbral de similitud de coseno (valor entre 0 y 1), por defecto 0.9.

    Retorna:
    - tuple: (cuenta_reclasificada, categoria_reclasificada) o (None, None) si no se encuentra coincidencia.
    """

    
    cuenta= preprocess_text(cuenta)
    #categoria = preprocess_text(cuenta)
    # Genera embeddings para la cuenta y categoría proporcionadas
    cuenta = f"El significado de '{cuenta}' en el contexto financiero."
    #categoria = f"El significado de '{categoria}' en el contexto financiero."
    cuenta_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cuenta)["embedding"])
    #categoria_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=categoria)["embedding"])
    
    #model = SentenceTransformer('all-mpnet-base-v2')
    #cuenta_emb_alterna = torch.tensor(model.encode(cuenta))

    #batch_size = 64  # Ajusta el tamaño del lote según la memoria disponible
    #embeddings = []
    
    max_similitud_cuenta = 0
    cuenta_reclasificada = None
    categoria_reclasificada = None


    # Itera sobre el diccionario para buscar una coincidencia
    for cuenta_diccionario, cat_diccionario, reclasificacion, categoria_reclasificacion in diccionario_cuentas:
            # Genera embeddings para la cuenta y categoría del diccionario
            cuenta_diccionario= preprocess_text(cuenta_diccionario)
            cuenta_diccionario = f"El significado de '{cuenta_diccionario}' en el contexto financiero."
            cuenta_dic_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cuenta_diccionario)["embedding"])
            #cat_dic_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cat_diccionario)["embedding"])
            #cuenta_dic_emb_alterna = torch.tensor(model.encode(cuenta_diccionario))
            # Calcula la similitud de coseno para cuenta y categoría
            similitud_cuenta = F.cosine_similarity(cuenta_emb, cuenta_dic_emb, dim=0)
            #similitud_categoria = F.cosine_similarity(categoria_emb, cat_dic_emb, dim=0)

            ##    
            ##
            #Usar similitud del 80% en esta alternativa
            ##
            ##
            ##
            ##
            #similitud_alterna = F.cosine_similarity(cuenta_emb_alterna, cuenta_dic_emb_alterna, dim=0)
            ##
            ##
            ##
            ##
            ##
            ##
            print (f'cuenta :{cuenta}')
            print (f'cuenta_diccionario :{cuenta_diccionario}')
            print (f'similitud_cuenta :{similitud_cuenta}')
            #print (f'similitud_alterna :{similitud_alterna}')
        # Verifica si ambas similitudes superan el umbral
            if similitud_cuenta >= umbral_similitud:
                #and similitud_categoria >= umbral_similitud:

                if similitud_cuenta > max_similitud_cuenta:
                    max_similitud_cuenta = similitud_cuenta
                    cuenta_reclasificada = reclasificacion
                    categoria_reclasificada = categoria_reclasificacion

            # Retorna la mejor coincidencia encontrada
                
            if max_similitud_cuenta > 0.98:
                print(f'maxima paridad encontrada {max_similitud_cuenta}')
                break
    return cuenta_reclasificada, categoria_reclasificada

    # Si no se encuentra coincidencia
    return None, None

def ollama_chat_with_validation(user_input, system_messages, vault_embeddings, vault_content, ollama_model, conversation_history, max_retries=4):
    """
    Llama al modelo y valida que la respuesta sea un JSON correcto con la estructura esperada.
    Reintenta en caso de error hasta `max_retries` veces.
    """
    system_message_extraction = [
        "You are a financial assistant specialized in extracting financial accounts from statements in JSON format.",
        "Your task is to extract **all** relevant accounts and organize them into JSON format without additional text or explanations.",
        """
        Only use the following categories:
        - For "estado_de_situacion_financiera": {"Activo Corriente", "Activo No Corriente", "Pasivo Corriente", "Pasivo No Corriente", "Patrimonio"}.
        - For "estado_resultado_integral": {"Ingresos", "Gastos"}.
        """,
        """
        Formatting rules:
        1. Respond only in JSON, with no extra text or explanations.
        2. Follow this structure:
        {
            "estado_de_situacion_financiera": [
                {
                    "categoria": "<one of the valid categories>",
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for each period in yyyy-mm-dd>" ,

                }
            ],
            "estado_resultado_integral": [
                {
                    "categoria": "<one of the valid categories>",
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for each period in yyyy-mm-dd>" ,
                }
            ]
        }
        3. Do not include totals like "Total Activo" or "Total Pasivo" or "total de pavisos no corrientes" or " Total del Patrimonio".
        """,
        """
        Special rules:
        - If an account like "Activo Fijo" is found, classify it as "Activo Corriente."
        - If an account like "Pasivo no circulante" is found, classify it as "Pasivo No Corriente."
         -If an account like "capital" is found, classify it as "Patrimonio" category.
        """,
        "Respond in Spanish, using the JSON format described, and ensure accuracy in field assignment."
        ]
    attempts = 0
    while attempts < max_retries:
        
        # Paso 1: Solicitar la lista inicial de cuentas
        
        consulta_inicial = """
        Identifica y construye una lista en formato Json con todas las cuentas en las categorías disponibles, con el siguiente formato y sin texto adicional:
        "estado_de_situacion_financiera": [
                {
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for the most recent period in yyyy-mm-dd>", ,
                }
            ],
            "estado_resultado_integral": [
                {
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for the most recent period in yyyy-mm-dd>" ,
                }
            ]

        -The expected output is a JSON structured according to the described format. 
        -No additional text should be included before or after the JSON, nor labels like "Here is the JSON" or explanatory comments.
        Special rules:
        - Validate that the total of "Activo" matches the sum of all identified accounts classified as "Activo Corriente" and "Activo No Corriente."
        - Validate that the total of "Pasivo" matches the sum of all identified accounts classified as "Pasivo Corriente" and "Pasivo No Corriente."
        - Validate that the total of "Patrimonio" matches the sum of all identified accounts classified under the "Patrimonio" category.

        """
        cuentas_detectadas = ollama_chat(consulta_inicial, system_message_extraction, vault_embeddings, vault_content, ollama_model, conversation_history)
        print(NEON_GREEN + "Response: \n\n" + cuentas_detectadas + RESET_COLOR)
        print(f"Cuentas detectadas inicialmente: {cuentas_detectadas}")
        if  is_valid_json(cuentas_detectadas):
            cuentas_detectadas_json = json.loads(cuentas_detectadas)
            consulta_validacion= """
            hacer la sumatoria del balance, activo = pasivo + patrimonio y responder con un array : [[valor patrimonio]['es correcta la cuenta']]
            """
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
            response = ollama_chat(user_input, system_messages, vault_embeddings, vault_content, ollama_model, conversation_history)
            print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
            
            return cuentas_detectadas_json

        if not is_valid_json(cuentas_detectadas):
            print(f"Respuesta inválida. Reintentando... ({attempts + 1}/{max_retries})")
            attempts += 1
            continue

        #response = ollama_chat(user_input, system_messages, vault_embeddings, vault_content, ollama_model, conversation_history)        
        print(f"Respuesta inválida. Reintentando... ({attempts + 1}/{max_retries})")
        attempts += 1
    
    # Si después de los intentos no se logra una respuesta válida, lanza una excepción o maneja el error
    raise ValueError("No se obtuvo un JSON válido después de varios intentos.")

def is_valid_json(response):
    """
    Verifica si el JSON tiene la estructura esperada.
    """
    try:
        data = json.loads(response)
        print(f"JSON correcto")
        # Validación exitosa
        return True

    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        return False
    except TypeError as e:
        print(f"Error de tipo: {e}")
        return False

def run(id_lote,id_doc,id_onbase,cuit,sql_connector,carpeta_entrada,periodo):

    # Parse command-line arguments
    print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
    parser = argparse.ArgumentParser(description="Ollama Chat")
    #parser.add_argument("--model", default="financial-model-lpa", help="Ollama model to use (default: financial-model-lpa)")
    parser.add_argument("--model", default="llama3.1", help="Ollama model to use (default: financial-model-lpa)")
    args = parser.parse_args()
    cargar_diccionarios(sql_connector)
    #id ejecucion
    id_ejecucion=0

    # Iterar sobre los archivos PDF en la carpeta
    ubicacion_actual= os.getcwd()
    
    if id_onbase is None:
        carpeta_pdf = obtener_carpeta_petersen(ubicacion_actual)
        #Consulta cliente 
        sql_cuit='codigociiu'
        sql_nombre='nombrecomercial'
        
        print(carpeta_pdf)
        with os.scandir(carpeta_pdf) as carpetas_bancos:
            for fichero in carpetas_bancos:
                carpeta_banco = os.path.join(carpeta_pdf,fichero.name,'IN')
                carpeta_salida = os.path.join(carpeta_pdf,fichero.name,'OUT')
                archivos_en_carpeta= os.listdir(carpeta_banco)
                entidad=fichero.name
                for index,archivo_pdf in enumerate(archivos_en_carpeta):
                    if archivo_pdf.endswith('.PDF') or archivo_pdf.endswith('.pdf'):
                        nombre_doc = archivo_pdf.replace(".pdf","")
                        cuit = archivo_pdf.split("_")[0]
                        pdf_path = os.path.join(carpeta_banco,archivo_pdf)
                        pdf_path_out =os.path.join(carpeta_salida,archivo_pdf)
                        print(f"Procesando archivo: {pdf_path}")
                        id_ejecucion = id_ejecucion+1
                        cuit,id_cliente,nombre_cliente = consultar_cliente(cuit,sql_cuit,sql_nombre,sql_connector)
                        try:
                            procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_cliente,nombre_cliente)
                            sql_connector.commit()
                        except Exception as e:
                            sql_connector.rollback()
                            insert_error(id_lote,id_doc,cuit,nombre_doc,sql_connector)
                            sql_connector.commit()
                            logging.error(f"Error en localragv3.run para el documento {nombre_doc}: {e}", exc_info=True)

    else:
        pdf_path,pdf_path_out,archivo_pdf = obtener_archivo_banesco(carpeta_entrada,id_onbase,cuit,periodo,ubicacion_actual)
        nombre_doc = archivo_pdf.replace(".pdf","")
        sql_cuit='num_doc'
        sql_nombre='nombre'
        cuit,id_cliente,nombre_cliente = consultar_cliente(cuit,sql_cuit,sql_nombre,sql_connector)
        print(cuit,id_cliente,nombre_cliente)
        print(f"Procesando archivo: {pdf_path}")
        try:
            procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_cliente,nombre_cliente)
        except Exception as e:
            insert_error(id_lote,id_doc,cuit,nombre_doc,sql_connector)
            logging.error(f"Error en localragv3.run para el documento {nombre_doc}: {e}", exc_info=True)

def procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_cliente,nombre_cliente):
    
    ruta_texto_reconstruido = None
    paginas_encontradas =''
    if os.path.exists(pdf_path):
        if (nombre_doc =='ejemplo extraction cuentas'):
            cuit=30708622555
            nombre_cliente='Pueba'
            pages = convert_from_path(pdf_path,first_page=1,last_page=2)
        elif (nombre_doc=='Balance Club  San Telmo pdf'):
            cuit=30529803563
            nombre_cliente='Club Atlético San Telmo'
            pages = convert_from_path(pdf_path,first_page=21,last_page=22)
        else:
            pages = convert_from_path(pdf_path)
        # Procesar cada página como una imagen
        info_extraida = []
        # Variable para contar las páginas revisadas después de encontrar una palabra clave
        contador_paginas = 0
        limite_paginas = 4  # Limitar a revisar solo las 5 páginas siguientes
        palabra_clave_encontrada = False  # Nueva variable para rastrear si se encontró alguna palabra clave
        for i, page in enumerate(pages):
            # Si ya revisamos las 5 páginas después de encontrar la palabra clave, continuar con el siguiente documento,
            # pero solo si ya hemos encontrado al menos una palabra clave, de lo contrario recorre todo el documento
            if contador_paginas >= limite_paginas and palabra_clave_encontrada:
                print(f"Se ha alcanzado el límite de {limite_paginas} páginas revisadas después de encontrar la palabra clave.")
                break
            
            #condicion para no evaluar todo el documento
            temp_image_path = f'pagina_{i}.jpg'
            page.save(temp_image_path, 'JPEG')

            img = cv2.imread(temp_image_path)

            image=rotate_page_pdf(img)
            gray_img   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Llamar a la función que realiza OCR selectivo en la parte superior
            palabra_encontrada,region_superior = procesar_pagina_con_ocr(gray_img, palabras_clave_regex, alto_region_superior=600)
            # Mostrar la imagen de la región superior para verificar visualmente
            #plt.imshow(cv2.cvtColor(region_superior, cv2.COLOR_BGR2RGB))
            #plt.title(f"Región superior de la página {i + 1}")
            #plt.show()
            if palabra_encontrada:
                print(NEON_GREEN+f"Palabra clave encontrada en la página {i + 1}. Procesando contenido."+ RESET_COLOR)
                
                # Marcar que se ha encontrado al menos una palabra clave
                palabra_clave_encontrada = True
                # Si se encontró una palabra clave, reiniciar el contador de páginas procesadas
                contador_paginas = 0

                reader = easyocr.Reader(["es"], gpu=True)
                resultados_por_pagina = reader.readtext(gray_img, paragraph=False)

                for detection in resultados_por_pagina:
                    text = detection[1]
                    confidence = detection[2]
                    box = detection[0]

                    info = {
                        "text": text,
                        "confidence": confidence,
                        "box": box,
                        # Otras propiedades útiles
                    }
                    info_extraida.append(info)

                # Inicializa variables para seguir la posición
                last_y = info_extraida[0]["box"][0][1]
                last_x = info_extraida[0]["box"][0][0]
                lines = []

                # Tolerancia vertical para considerar textos en el mismo renglón
                tolerancia_vertical = 10
                tolerancia_horizontal= 150
                separador = "\t"
                # Establece hay_espaciado_inicial en True al inicio
                hay_espaciado_inicial = True
                # Recorre la información para determinar los saltos de línea
                for info in info_extraida:
                        y = info["box"][0][1]  # Tomar el valor Y de uno de los puntos, ya que son aproximadamente iguales
                        x = info["box"][0][0]  # Tomar el valor X del punto izquierdo


                        if abs(y - last_y) > tolerancia_vertical:
                            lines.append("\n")
                            hay_espaciado_inicial = True
                            last_x = x 

                        #if((x - last_x) > tolerancia_horizontal):
                        # Agregar espaciado entre palabras basado en las coordenadas X
                        #    espaciado = max(0, int((x - last_x) / 20))  # Ajusta el divisor según sea necesario
                        #    lines.append(" " * espaciado)
                        
                        lines.append(info["text"].lower() + " ")
                        last_y = y
                        last_x = x + len(info["text"].lower()) * 20  # Ajusta el multiplicador según sea necesario para la longitud promedio de las palabras
                        
                # Combina las líneas para obtener el texto final
                texto_reconstruido = "".join(lines)
                # Normaliza espacios en blanco y limpia el texto
                texto_reconstruido = re.sub(r'\s+', ' ', texto_reconstruido)
                sentences = re.split(r'(?<=[.!?]) +', texto_reconstruido)
                #sentences = re.split(r'(?<!\d)(?<!\d\.\d)(?<=[.!?]) +', texto_reconstruido)  # split on spaces following sentence-ending punctuation
                chunks = []
                current_chunk = ""
                for sentence in sentences:
                    # Check if the current sentence plus the current chunk exceeds the limit
                    if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                        current_chunk += (sentence + " ").strip()
                    else:
                        # When the chunk exceeds 1000 characters, store it and start a new one
                        chunks.append(current_chunk)
                        current_chunk = sentence + " "
                if current_chunk:  # Don't forget the last chunk!
                    chunks.append(current_chunk)
                # Guarda el texto reconstruido en una carpeta específica
                carpeta_texto_reconstruido = os.path.join(ubicacion_actual, 'src', 'pdfs','text_reconstruido')
                if not os.path.exists(carpeta_texto_reconstruido):
                        os.makedirs(carpeta_texto_reconstruido)

                ruta_texto_reconstruido = os.path.join(carpeta_texto_reconstruido, f'{nombre_doc}_reconstruido.txt')
                with open(ruta_texto_reconstruido, "w", encoding="utf-8") as file:
                        for chunk in chunks:
                            file.write(chunk.strip() + "\n")
                        # file.write("\n"+texto_reconstruido+"\n")

                print(f"Texto reconstruido guardado en: {ruta_texto_reconstruido}")
            else:
                print(YELLOW+f"La página {i + 1} no contiene información relevante."+ RESET_COLOR)  
                if palabra_clave_encontrada:
                    contador_paginas += 1
            # Eliminar la imagen temporal
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        
        #with open(ruta_texto_reconstruido, 'r', encoding='utf-8') as file:
        #        text = file.read()

        # Load the vault content
        print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
        vault_content = []
        if os.path.exists(ruta_texto_reconstruido):
            with open(ruta_texto_reconstruido, "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
        # Generate embeddings for the vault content using Ollama
        if vault_content:
            print(NEON_GREEN + "Generating embeddings content..." + RESET_COLOR)
            vault_embeddings = []
            for content in vault_content:
                response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
                vault_embeddings.append(response["embedding"])

            # Convert to tensor and print embeddings
            print("Converting embeddings to tensor...")
            vault_embeddings_tensor = torch.tensor(vault_embeddings) 
            print("Embeddings for each line in the vault:")
            print(vault_embeddings_tensor)

            conversation_history = []
            
            system_message = [
        "You are a financial assistant specialized in extracting financial accounts from statements.",
        "Your task is to extract **all** relevant accounts and organize them using a STRICT structured format without additional text, comments, or explanations.",
        "Respond in Spanish, using the following format for each account: 'Categoría: <categoría>' | Cuenta: <nombre> | Valor: <valor numérico sin separadores de miles> | Fecha: <fecha en formato yyyy-mm-dd>'.",
        "Do not use commas for thousand separators. Only use a dot '.' as a decimal separator.",
        "Include only accounts with a non-zero value for the latest period under the specified category.",
        "Exclude aggregated accounts such as 'Total Activo Corriente' or 'Total Activo No Corriente'.",
        "If no valid accounts can be identified, respond with 'SIN CUENTAS'.",
        "Ensure the result strictly adheres to the format and instructions provided."
    ]


            lpa_input = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is a text containing only the accounts of "Activo Corriente" that have a value associated for the latest period, ensuring that the sum of these accounts matches the "Total Activo Corriente." Exclude any accounts with a value of 0 for the latest period.

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Activo corriente
            - Cuenta: <Nombre de la cuenta>
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Activo Corriente | Cuenta: Prestamos por cobrar | Valor: 123 | Fecha: 2023-12-31
            Categoría: Activo Corriente | Cuenta: Activos bajo contratos | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts under the specified category with a non-zero value for the latest period.
            2. Exclude aggregated accounts like "total" or "Total Current Assets" or similar totals.
            3. Ensure the sum of the values corresponds to the total for the specified category.
            4. If you cannot identify valid accounts, respond with "SIN CUENTAS."
            5. The result must adhere strictly to the described format without additional text, comments, or explanations.
            Special Rules:
            - If an account like "Activo Fijo" is found, classify it under "Activo Corriente."
            - If the "Total Activo" cannot be matched due to missing or inconsistent values, return an empty array as the value of "activo."
            """

            lpa_input_activo_no_corriente = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is a text containing only the accounts of "Activo No Corriente" that have a value associated for the latest period, ensuring that the sum of these accounts matches the "Total Activo No Corriente." Exclude any accounts with a value of 0 for the latest period.

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Activo no Corriente
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Activo no Corriente | Cuenta: prueba | Valor: 123 | Fecha: 2023-12-31
            Categoría: Activo no Corriente | Cuenta: prueba2 | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Activo No Corriente" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Activo No Corriente" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Activo No Corriente" or similar totals.
            4. The result must adhere strictly to the described format without additional text, comments, or explanations.
            5. Skip any accounts without a numeric value or where the value for the latest period is missing or zero.
            6. The output must only include accounts under the "Activo No Corriente" category.
            7. If you cannot identify valid accounts, respond with "SIN CUENTAS."
            Special Rules:
            - If the "Total Activo No Corriente" cannot be matched due to missing or inconsistent values, return an empty array as the value of "activo_no_corriente."
            """

            lpa_input_pasivo_corriente = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is text containing only the accounts of "Pasivo Corriente" (also known as "Pasivo Circulante") that have a value associated for the latest period, ensuring that the sum of these accounts matches the "Total Pasivo Corriente." Exclude any accounts with a value of 0 for the latest period.

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Pasivo Corriente
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Pasivo Corriente | Cuenta: Prestamos por cobrar | Valor: 123 | Fecha: 2023-12-31
            Categoría: Pasivo Corriente | Cuenta: Activos bajo contratos | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Pasivo Corriente" or "Pasivo Circulante" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Pasivo Corriente" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Pasivo Corriente" or similar totals.
            4. The result must adhere strictly to the described format without additional text, comments, or explanations.
            5. If you cannot identify valid accounts, respond with "SIN CUENTAS."
            Special Rules:
            - If the "Total Pasivo Corriente" cannot be matched due to missing or inconsistent values, return an empty array as the value of "pasivo_corriente."
            """

            lpa_input_pasivo_no_corriente = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is text containing only the accounts of "Pasivo No Corriente" (also known as "Pasivo No Circulante" or "Otros Pasivos") that have a value associated for the latest period, ensuring that the sum of these accounts matches the "Total Pasivo No Corriente." Exclude any accounts with a value of 0 for the latest period.

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Pasivo no Corriente
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Pasivo no Corriente | Cuenta: Prestamos por cobrar | Valor: 123 | Fecha: 2023-12-31
            Categoría: Pasivo no Corriente | Cuenta: Activos bajo contratos | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Pasivo No Corriente," "Pasivo No Circulante," or "Otros Pasivos" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Pasivo No Corriente" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Pasivo No Corriente" or similar totals.
            5. Skip any accounts without a numeric value or where the value for the latest period is missing or zero.
            6. The output must only include accounts under the "Pasivo No Corriente," "Pasivo No Circulante," or "Otros Pasivos" category.
            7.The result must adhere strictly to the described format without additional text, comments, or explanations.
            8. If the "Total Pasivo No Corriente" cannot be matched due to missing or inconsistent values, return an empty array as the value of "pasivo_no_corriente."
            """

            lpa_input_patrimonio = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is text containing only the accounts of "Patrimonio" (including "Patrimonio No Controladora" and other names associated with this category) that have a value associated for the latest period, ensuring that the sum of these accounts matches the "Total Patrimonio." Exclude any accounts with a value of 0 for the latest period.  
            ***Do not include aggregated accounts such as "Total Patrimonio," "Total de patrimonio atribuible a los propietarios", or similar totals.***

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Patrimonio
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Patrimonio | Cuenta: Prestamos por cobrar | Valor: 123 | Fecha: 2023-12-31
            Categoría: Patrimonio | Cuenta: Activos bajo contratos | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Patrimonio" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Patrimonio" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Patrimonio" or similar totals.
            4. The result must adhere strictly to the described format without additional text, comments, or explanations.
            5. Skip any accounts without a numeric value or where the value for the latest period is missing or zero.
            6. The output must only include accounts under the "Patrimonio" category.
            7. If you cannot identify valid accounts, respond with "SIN CUENTAS."
            Special Rules:
            - If the "Total Patrimonio" cannot be matched due to missing or inconsistent values, return an empty array as the value of "patrimonio."
            """

            lpa_input_ingresos = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is text containing only the accounts of "Ingresos" that appear explicitly in the "Estado de Resultados" or similars and have a value associated for the latest period.  
            ***Do not include aggregated accounts such as "Total Ingresos" or similar totals.***  

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Ingresos
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Ingresos | Cuenta: ventas | Valor: 123 | Fecha: 2023-12-31
            Categoría: Ingresos | Cuenta: Costo de venta | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Ingresos" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Ingresos" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Ingresos" or similar totals.
            4. The result must adhere strictly to the described format without additional text, comments, or explanations.
            5. Skip any accounts without a numeric value or where the value for the latest period is missing or zero.
            6. The output must only include accounts under the "Ingresos" category.
            7. If you cannot identify valid accounts, respond with "SIN CUENTAS."

            """
            lpa_input_gastos = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured format and WITHOUT ADDITIONAL TEXT.

            The expected output is text containing only the accounts of "Gastos" that appear explicitly in the "Estado de Resultados" or similars and have a value associated for the latest period.  
            ***Do not include aggregated accounts such as "Total Gastos" or similar totals.***  

            Respond only based on the information extracted in the Relevant Context using the following STRICT format:

            Each account must be represented as a single line, with the following fields separated by the "|" character:

            - Categoría: Gastos
            - Cuenta: Nombre de la cuenta
            - Valor: (numeric, without commas or thousand separators, only digits)
            - Fecha: <fecha en formato yyyy-mm-dd>

            Example of the expected format:
            Categoría: Gastos | Cuenta: gastos por ventas | Valor: 123 | Fecha: 2023-12-31
            Categoría: Gastos | Cuenta: depreciación y amortización | Valor: 123 | Fecha: 2023-12-31

            Rules:
            1. Include only accounts classified under "Gastos" with a non-zero value for the latest period.
            2. Validate that the sum of the values of all included accounts is equal to the "Total Gastos" for the latest period.
            3. Exclude aggregated accounts like "total" or "Total Gastos" or similar totals.
            4. The result must adhere strictly to the described format without additional text, comments, or explanations.
            5. Skip any accounts without a numeric value or where the value for the latest period is missing or zero.
            6. The output must only include accounts under the "Gastos" category.
            7. If you cannot identify valid accounts, respond with "SIN CUENTAS."

            """
            try:
                conversation_history = []
                #activos corrientes
                response_data_cuentas_activo_corriente = ollama_chat(lpa_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
                
                #print(NEON_GREEN + "response_data_cuentas activos corrientes: \n\n" + response_data_cuentas_activo_corriente + RESET_COLOR)
                      
                print(NEON_GREEN + "response_data_cuentas activos corrientes: \n\n" + response_data_cuentas_activo_corriente + RESET_COLOR)

                data_activo_corriente = (response_data_cuentas_activo_corriente)
                
                conversation_history = []


                #activos no corrientes
                response_data_cuentas_activo_no_corriente = ollama_chat(lpa_input_activo_no_corriente, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)

                data_activo_no_corriente = (response_data_cuentas_activo_no_corriente)
                
                print(NEON_GREEN + "response_data_cuentas_activo_no_corriente: \n\n" + data_activo_no_corriente + RESET_COLOR)

                conversation_history = []

                #pasivos corrientes
                response_data_cuentas_pasivo_corriente = ollama_chat(lpa_input_pasivo_corriente, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
               
                data_pasivo_corriente = (response_data_cuentas_pasivo_corriente)
                
                print(NEON_GREEN + "response_data_cuentas_pasivo_corriente: \n\n" + data_pasivo_corriente + RESET_COLOR)
                
                conversation_history = []
                
                #pasivos no corrientes
                response_data_cuentas_pasivo_no_corriente = ollama_chat(lpa_input_pasivo_no_corriente, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)   

                data_pasivo_no_corriente = (response_data_cuentas_pasivo_no_corriente)

                print(NEON_GREEN + "response_data_cuentas_pasivo_no_corriente: \n\n" + data_pasivo_no_corriente + RESET_COLOR)

                conversation_history = []
                
                #patrimonio
                response_data_cuentas_patrimonio = ollama_chat(lpa_input_patrimonio, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
                
                data_patrimonio = (response_data_cuentas_patrimonio)
                print(NEON_GREEN + "response_data_cuentas_patrimonio: \n\n" + response_data_cuentas_patrimonio + RESET_COLOR)

                conversation_history = []
            
                #Ingresos Estado de Resultados
                response_data_cuentas_ingresos = ollama_chat(lpa_input_ingresos, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
                print(NEON_GREEN + "response_data_cuentas_ingresos: \n\n" + response_data_cuentas_ingresos + RESET_COLOR)

                response_data_cuentas_ingresos_corrected = (response_data_cuentas_ingresos)

                data_ingresos = (response_data_cuentas_ingresos)

                conversation_history = []

                #Gastos Estado de Resultados
                response_data_cuentas_gastos = ollama_chat(lpa_input_gastos, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
                print(NEON_GREEN + "response_data_cuentas_gastos: \n\n" + response_data_cuentas_gastos + RESET_COLOR)

                response_data_cuentas_gastos_corrected = (response_data_cuentas_gastos)

                data_gastos = (response_data_cuentas_gastos)

                conversation_history = []

            except ValueError as e:
                print(f"Error: {e}")


            # Carga el string JSON como un objeto Python (en este caso, una lista de diccionarios)
            #data = json.loads(response)
            #print(response_data_cuentas_activo_no_corriente)
                #estado cero porque no se ha procesado en la aplicacion
            estado=0
            cuenta_reclasificada='' 
            resultados_acumulados = []
            resultados_acumulados_ER = []

            selected_columns_results = []
            orden_cuenta_global=1

            moneda='USD'

            #activos corrientes identificados
            resultados_acumulados, orden_cuenta_global = procesar_cuentas_simple(
            data_activo_corriente, 'activo_corriente', diccionario_cuentas_activos_corrientes, 'Activo Corriente', orden_cuenta_global,resultados_acumulados,nombre_cliente,cuit,id_cliente,periodo_archivo,moneda,sql_connector
            )

            
            #activos no corrientes identificados
            resultados_acumulados, orden_cuenta_global = procesar_cuentas_simple(
            data_activo_no_corriente, 'activo_no_corriente', diccionario_cuentas_activos_no_corrientes, 'Activo No Corriente', orden_cuenta_global,resultados_acumulados,nombre_cliente,cuit
            )

            #pasivos corrientes identificados
            resultados_acumulados, orden_cuenta_global = procesar_cuentas_simple(
            data_pasivo_corriente, 'pasivo_corriente', diccionario_cuentas_pasivos_corrientes, 'Pasivo Corriente', orden_cuenta_global,resultados_acumulados,nombre_cliente,cuit
            )

            #pasivos no corrientes identificados
            resultados_acumulados, orden_cuenta_global = procesar_cuentas_simple(
            data_pasivo_no_corriente, 'pasivo_no_corriente', diccionario_cuentas_pasivos_no_corrientes, 'Pasivo No Corriente', orden_cuenta_global,resultados_acumulados,nombre_cliente,cuit
            )

            #patrimonio identificados
            resultados_acumulados, orden_cuenta_global = procesar_cuentas_simple(
            data_patrimonio, 'patrimonio', diccionario_cuentas_patrimonio, 'Patrimonio', orden_cuenta_global,resultados_acumulados,nombre_cliente,cuit
            )

            #Ingresos Estado de resultados
            resultados_acumulados_ER, orden_cuenta_global = procesar_cuentas_simple(
            data_ingresos, 'Ingresos', diccionario_cuentas_estado_resultados, 'estado resultados', orden_cuenta_global,resultados_acumulados_ER,nombre_cliente,cuit
            )

            #Gastos Estado de resultados
            resultados_acumulados_ER, orden_cuenta_global = procesar_cuentas_simple(
            data_gastos, 'Gastos', diccionario_cuentas_estado_resultados, 'estado resultados', orden_cuenta_global,resultados_acumulados_ER,nombre_cliente,cuit
            )

            df_balances = pd.DataFrame(resultados_acumulados, columns=['cliente','cuit','categoria','orden_cuenta', 'cuenta', 'cuenta_reclasificada','categoria_reclasificada', 'valor', 'periodo','tipo_periodo','diferencia_periodos','estado'])
            df_financiero = pd.DataFrame(selected_columns_results, columns=['cliente','cuit','categoria','orden_cuenta','cuenta','valor', 'periodo','estado'])
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)

            #print(df_balances)
            print(df_balances.loc[:, ['orden_cuenta','categoria', 'cuenta', 'valor', 'periodo']])
            print(df_financiero)     
            print(f"-------------------")
            
            df_balances['mm'] = pd.to_datetime(df_balances['periodo']).dt.month
            diferencia_meses = df_balances['mm'].iloc[0]
            periodo_archivo = df_balances['periodo'].iloc[0]
            print(diferencia_meses)
            
            

            #obtiene el maximo id balance temporal
            id_balance_max=get_maximo_id_balance_temp(sql_connector)
            #Creacion de id_balance
            #####(Falta ajustar cuando hayan multiples periodos)
            insert_inicial(id_balance_max,id_cliente,nombre_cliente,periodo_archivo,id_lote,id_doc,cuit,nombre_doc,paginas_encontradas,sql_connector,moneda)
            print(df_balances)
            insertar_informacion(df_balances,df_financiero,id_balance_max,id_doc,sql_connector)
        else:
            print(f'No se generó texto para el document: "{nombre_doc} " Pasando al siguiente documento.') 
        
        #Mueve el archivo
        os.replace(pdf_path, pdf_path_out)
    else:
        print('No existe archivo')


def insertar_informacion(df_balances,df_financiero,id_balance_max,id_doc,sql_connector):
    categoria_a_diccionario = {
                'activo corriente': diccionario_var_activo_corriente,
                'activo no corriente': diccionario_var_activo_no_corriente,
                'pasivo corriente': diccionario_var_pasivo_corriente,
                'pasivo no corriente': diccionario_var_pasivo_no_corriente,
                'patrimonio': diccionario_var_patrimonio,
                'estado resultados': diccionario_var_estado_resultados
                
            }
            # Iterar sobre las filas del DataFrame
    for index, row in df_balances.iterrows():
        categoria = str(row['categoria_reclasificada']).lower()

        if categoria in categoria_a_diccionario:
            procesar_fila(row, index, id_doc, id_balance_max, categoria_a_diccionario, diccionario_resultantes,categoria,sql_connector)
        else:

            # Si la categoría no está mapeada
            print(f"Categoría desconocida: {categoria}")
            procesar_fila(row, index, id_doc, id_balance_max, categoria_a_diccionario, diccionario_resultantes,'None',sql_connector)
    #incrementa el id documento
    id_doc=id_doc+1
    #imprime las cuentas extraidas
    
    print(df_financiero)     
    print(f"-------------------")


# Función para manejar inserciones y actualizaciones según reglas
def procesar_fila(row, index, id_doc, id_balance_max, categoria_diccionario, diccionario_resultantes,categoria,sql_connector):
    cuenta = row['cuenta']
    valor = row['valor'] if row['valor'] != '' else 0
    cuenta_reclasificada = str(row['cuenta_reclasificada']) if str(row['cuenta_reclasificada']) != '' else 'None'

    # Determinar diccionario a usar
    diccionario = categoria_diccionario.get(categoria, {})
    variable = diccionario.get(cuenta_reclasificada, diccionario_resultantes.get(cuenta_reclasificada, None))

    if variable:
        # Insertar en TL_LPA_CUENTAS
        
        INSERT_TL_LPA_CUENTAS = f"""
            Insert into TL_LPA_CUENTAS 
            (id_documento, id_orden, cuenta_extraida, categoria_clasificada, cuenta_clasificada, cuenta_final, variable, nota, fecha_modificacion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (id_doc, index, cuenta, categoria, cuenta_reclasificada, cuenta_reclasificada, variable, 'NA', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sql_connector.insert_data(INSERT_TL_LPA_CUENTAS,params) 
        # Actualizar TB_LPA_BALANCE
        UPDATE_TB_LPA_BALANCE = """
            UPDATE TB_LPA_BALANCE 
            SET {} = ? 
            WHERE id_balance = ?;
        """
        query = UPDATE_TB_LPA_BALANCE.format(variable)
        params=(valor, id_balance_max)
        sql_connector.insert_data(query,params) 
        # Insertar en T_BALANCE
        
        INSERT_T_BALANCE = """
            Insert into T_BALANCE 
            (id_balance, id_cuenta, cuenta_inicial, variable, cuenta_final, valor_inicial, valor_final, id_documento)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params=(id_balance_max, index, cuenta, variable, cuenta_reclasificada, valor, valor, id_doc)
        sql_connector.insert_data(INSERT_T_BALANCE,params) 

        INSERT_TL_LPA_VALORES="""
        Insert into TL_LPA_VALORES (id_documento, id_periodo, id_cuenta,valor_leido,confianza, usuario_modificacion,fecha_modificacion)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """
        params=(id_doc,1,index,row['valor'],0.8,'LPA',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sql_connector.insert_data(INSERT_TL_LPA_VALORES,params) 
        
        
    else:
        # Insertar en TL_LPA_CUENTAS sin variable
        INSERT_TL_LPA_CUENTAS = f"""
            Insert into TL_LPA_CUENTAS 
            (id_documento, id_orden, cuenta_extraida, categoria_clasificada, nota, fecha_modificacion,cuenta_clasificada, cuenta_final)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (id_doc, index, cuenta, categoria, 'NA', datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'None','None')
        sql_connector.insert_data(INSERT_TL_LPA_CUENTAS,params) 
        # Insertar en T_BALANCE sin variable
        INSERT_T_BALANCE = f"""
            Insert into T_BALANCE 
            (id_balance, id_cuenta, cuenta_inicial, valor_inicial, valor_final, id_documento)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params =(id_balance_max, index, cuenta, valor, valor, id_doc)
        sql_connector.insert_data(INSERT_T_BALANCE,params) 

        INSERT_TL_LPA_VALORES="""
        Insert into TL_LPA_VALORES (id_documento, id_periodo, id_cuenta,valor_leido,confianza, usuario_modificacion,fecha_modificacion)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """
        params=(id_doc,1,index,row['valor'],0.8,'LPA',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sql_connector.insert_data(INSERT_TL_LPA_VALORES,params) 

    
def consultar_cliente(cuit,sql_cuit,sql_nombre,sql_connector):
    #Consultar cliente existe y extrae id_cliente
    nombre_cliente ='PRUEBA'
    select_cliente = f"Select id_cliente,{sql_nombre},{sql_cuit} from TB_cliente where {sql_cuit} ='{cuit}' or cod_cliente='{cuit}'"
    consulta_cliente=sql_connector.read_data(select_cliente)
    cuit='11111'
    id_cliente='11111'
    if len(consulta_cliente) == 0:
        print('Cliente no existe')
    else:        
        for row in consulta_cliente:
            cuit = getattr(row, sql_cuit, None).strip()
            #cuit = row.num_doc
            id_cliente = getattr(row, 'id_cliente', None)
            nombre_cliente = getattr(row, sql_nombre, None)
            #nombre_cliente = row.nombrecomercial
    return cuit,id_cliente,nombre_cliente

def get_maximo_id_balance_temp(sql_connector):
    select_id_balance = 'Select isnull(max(id_balance),0) as id_balance from TB_LPA_BALANCE'
    select_id_balance_2 = 'Select isnull(max(id_balance_temp),0) as id_balance_temp from TL_LPA_CREACION'
    consulta_max_id_balance=sql_connector.read_data(select_id_balance)
    consulta_max_id_balance_2=sql_connector.read_data(select_id_balance_2)

    
    for row_b in consulta_max_id_balance:
        id_balance = row_b.id_balance +1

    for row_c in consulta_max_id_balance_2:
        id_balance_2 = row_c.id_balance_temp +1

    return max(id_balance, id_balance_2)
def insert_inicial(id_balance_max,id_cliente,nombre_cliente,periodo_archivo,id_lote,id_doc,cuit,nombre_doc,paginas_encontradas,sql_connector,moneda):
    
    INSERT_TB_LPA_BALANCE="Insert into TB_LPA_BALANCE (id_balance,num_calificacion,id_cliente,nombre_cliente,id_rating,id_modelo_bal,estatus,moneda,unidad_monetaria,tasa_cambio,mes,estado_mes,estado,usuario_creacion,fecha_creacion,fecha_asignacion,norma_contable,auditor) "\
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    params =(id_balance_max,'1',id_cliente,nombre_cliente,id_balance_max,1,'Auditado',moneda,'Absolutas',1,'12','Individual','A','LPA',datetime.now().strftime('%Y-%m-%d %H:%M:%S'),periodo_archivo,'IFRS','NA')
    sql_connector.insert_data(INSERT_TB_LPA_BALANCE, params)

    INSERT_TL_LPA_CREACION = "Insert into TL_LPA_CREACION (id_ejecucion,id_documento,id_periodo,id_balance_temp,fecha_asignacion,estado) "\
        "VALUES (?, ?, ?, ?, ?, ?)"
    params = (id_lote, id_doc, 1, id_balance_max, periodo_archivo, 'A')
    sql_connector.insert_data(INSERT_TL_LPA_CREACION, params)
    
    ##to do (Check)(Revisar)
    #Cambiar el id_documento
    #####################Falta crear id doc
    INSERT_TL_BALANCE = "Insert into TL_BALANCE (id_ejecucion,id_documento, id_periodo, cuit, nombre_documento, fecha_asignacion,pag_procesadas, estado,proceso, observacion,fecha_creacion) "\
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    params=(id_lote,id_doc,1,cuit,nombre_doc,periodo_archivo,paginas_encontradas,'P','A','Se procesó correctamente',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    sql_connector.insert_data(INSERT_TL_BALANCE,params) 

def insert_error(id_lote,id_doc,cuit,nombre_doc,sql_connector):
    INSERT_TL_BALANCE = "Insert into TL_BALANCE (id_ejecucion,id_documento, id_periodo, cuit, nombre_documento, estado,proceso, observacion,fecha_creacion) "\
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    params=(id_lote,id_doc,1,cuit,nombre_doc,'F','F','Hubo un error al procesar el documento',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    sql_connector.insert_data(INSERT_TL_BALANCE,params) 

def cargar_diccionarios(sql_connector):
    #Consulta diccionario de cuentas
    select_statement_activos_corrientes = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Activo corriente' and tipo='Cuenta';"
    select_statement_activos_no_corrientes = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Activo No corriente' and tipo='Cuenta';"
    select_statement_pasivos_corrientes = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Pasivo Corriente' and tipo='Cuenta';"
    select_statement_pasivos_no_corrientes = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Pasivo No corriente' and tipo='Cuenta';"
    select_statement_patrimonio = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Patrimonio' and tipo='Cuenta';"
    select_statement_estado_resultados = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS where categoria ='Estado Resultados' and tipo='Cuenta';"
    result_activos_corrientes = sql_connector.read_data(select_statement_activos_corrientes)
    result_activos_no_corrientes = sql_connector.read_data(select_statement_activos_no_corrientes)
    result_pasivos_corrientes = sql_connector.read_data(select_statement_pasivos_corrientes)
    result_pasivos_no_corrientes = sql_connector.read_data(select_statement_pasivos_no_corrientes)
    result_patrimonio = sql_connector.read_data(select_statement_patrimonio)
    result_estado_resultados = sql_connector.read_data(select_statement_estado_resultados)
    for row in result_activos_corrientes:
        diccionario_cuentas_activos_corrientes.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))
    
    for row in result_activos_no_corrientes:
        diccionario_cuentas_activos_no_corrientes.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))

    for row in result_pasivos_corrientes:
        diccionario_cuentas_pasivos_corrientes.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))

    for row in result_pasivos_no_corrientes:
        diccionario_cuentas_pasivos_no_corrientes.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))

    for row in result_patrimonio:
        diccionario_cuentas_patrimonio.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))

    for row in result_estado_resultados:
        diccionario_cuentas_estado_resultados.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))
    #Llenado de duplas cuenta - var
    select_variables ="Select variable,categoria,cuenta,id_modelo from TL_LPA_MODELO"
    result_variables = sql_connector.read_data(select_variables)
    for registro in result_variables:
        if registro[1].lower() == 'activo corriente':
            diccionario_var_activo_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1].lower() == 'activo no corriente':
            diccionario_var_activo_no_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1].lower() == 'pasivo corriente':
            diccionario_var_pasivo_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1].lower() == 'pasivo no corriente':
            diccionario_var_pasivo_no_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1].lower() == 'patrimonio':
            diccionario_var_patrimonio[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        else:
            diccionario_var_estado_resultados[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]

def obtener_archivo_banesco(carpeta_entrada,id_onbase,cuit,periodo,ubicacion_actual):
    
    carpeta_padre = os.path.dirname(ubicacion_actual)
    carpeta_raiz =os.path.dirname(carpeta_padre)
    # Ruta a la carpeta que contiene los archivos PDF
    carpeta_pdf_in = os.path.join(carpeta_raiz, 'MLPA/BALANCES/IN',carpeta_entrada)
    carpeta_pdf_out = os.path.join(carpeta_raiz, 'MLPA/BALANCES/OUT')

    archivo_pdf = f'{id_onbase}_{cuit}_{periodo}.pdf'
    # Obtener la lista de archivos en la carpeta
    archivos_en_carpeta = os.listdir(carpeta_pdf_in)
    pdf_path = os.path.join(carpeta_pdf_in, archivo_pdf)
    pdf_path_out = os.path.join(carpeta_pdf_out, archivo_pdf)

    return pdf_path,pdf_path_out,archivo_pdf
def obtener_carpeta_petersen(ubicacion_actual):
    directorio_actual =os.path.abspath(ubicacion_actual)
    disco = os.path.splitdrive(directorio_actual)[0]
    # Ruta a la carpeta que contiene los archivos PDF
    carpeta_pdf = os.path.join(disco, '\MLPA\BALANCE\desarrollo')

    return carpeta_pdf

def consultar_balance(id_cliente,fecha_asignacion,moneda,sql_connector):
    SELET_STATEMENT = f"select id_balance from tb_balance where id_cliente={id_cliente} and fecha_asignacion ={fecha_asignacion} and moneda='{moneda}'"
    consulta= sql_connector.read_data(SELET_STATEMENT)
    id_balance=-1
    if len(consulta) < 2:
        for row in consulta:
            id_balance= row.id_balance
    return id_balance
def consultar_cuentas_previas(id_balance,sql_connector):
    cuentas=None
    SELET_STATEMENT = f"select max(id_documento) from tl_lpa_creacion where id_balance={id_balance}"
    consulta= sql_connector.read_data(SELET_STATEMENT)
    for row in consulta:
        id_documento= row.id_documento

    SELECT_CUENTAS=f"Select categoria_clasificada, cuenta_final from tl_lpa_cuentas where id_documento={id_documento}"
    consulta_cuentas= sql_connector.read_data(SELECT_CUENTAS)
    for cuenta in consulta_cuentas:
        cuentas.append(cuenta)
    return cuentas