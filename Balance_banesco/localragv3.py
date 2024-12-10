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
def procesar_pagina_con_ocr(image, palabras_clave_regex, alto_region_superior=500):
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

    print (texto_detectado)
    # Buscar cualquier coincidencia de las palabras clave usando regex
    for regex in palabras_clave_regex:
        # Modificar la expresión regular para permitir errores tipográficos
        regex_modificado = regex.replace("situacion", "(situacion|sltuacion|situación)")  # Ejemplo de error tipográfico permitido
        regex_modificado = regex_modificado.replace("resultado", "(resultado|resultados)")

        if re.search(regex_modificado, texto_detectado):
            return True, region_superior  # Retorna la región de la imagen además del resultado

    return False, region_superior

palabras_clave_regex = [
    r"estado de situación financiera",
    r"estado de situacion patrimonial",
    r"estado consolidado de situación financiera",
    r"estado consolidado de resultados y otras utilidades integrales",
    r"estado de situacion financiera",
    r"estado de resultados",
    r"estado de resultados integrales",
    r"estado de resultado",
    r"Balance Sheet",
    r"balance shcet "
]

diccionario_cuentas =[]


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
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
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
        temperature=0.2,
    )
    print(conversation_history)
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content


def obtener_reclasificacion(cuenta, categoria, diccionario_cuentas, umbral_similitud=0.8):
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
    model = SentenceTransformer('all-mpnet-base-v2')
    cuenta= preprocess_text(cuenta)
    # Genera embeddings para la cuenta y categoría proporcionadas
    cuenta = f"El significado de '{cuenta}' en el contexto financiero."
    cuenta_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cuenta)["embedding"])
    categoria_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=categoria)["embedding"])
    
    cuenta_emb_alterna = torch.tensor(model.encode(cuenta))

    
    max_similitud_cuenta = 0
    cuenta_reclasificada = None
    categoria_reclasificada = None

    # Itera sobre el diccionario para buscar una coincidencia
    for cuenta_diccionario, cat_diccionario, reclasificacion, categoria_reclasificacion in diccionario_cuentas:
            # Genera embeddings para la cuenta y categoría del diccionario
            cuenta_diccionario= preprocess_text(cuenta_diccionario)
            cuenta_diccionario = f"El significado de '{cuenta_diccionario}' en el contexto financiero."
            cuenta_dic_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cuenta_diccionario)["embedding"])
            cat_dic_emb = torch.tensor(ollama.embeddings(model='mxbai-embed-large', prompt=cat_diccionario)["embedding"])
            cuenta_dic_emb_alterna = torch.tensor(model.encode(cuenta_diccionario))
            # Calcula la similitud de coseno para cuenta y categoría
            similitud_cuenta = F.cosine_similarity(cuenta_emb, cuenta_dic_emb, dim=0)
            similitud_categoria = F.cosine_similarity(categoria_emb, cat_dic_emb, dim=0)

            ##    
            ##
            #Usar similitud del 80% en esta alternativa
            ##
            ##
            ##
            ##
            similitud_alterna = F.cosine_similarity(cuenta_emb_alterna, cuenta_dic_emb_alterna, dim=0)
            ##
            ##
            ##
            ##
            ##
            ##
            print (f'cuenta :{cuenta}')
            print (f'cuenta_diccionario :{cuenta_diccionario}')
            print (f'similitud_cuenta :{similitud_cuenta}')
            print (f'similitud_alterna :{similitud_alterna}')
        # Verifica si ambas similitudes superan el umbral
            if similitud_alterna >= umbral_similitud:
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
        "You are a financial assistant specialized in extracting financial accounts from statements.",
        "Your task is to extract **all** relevant accounts and organize them into JSON format. Without additional text or explanations.",
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
        3. Do not include any total account.
        """,
        """
        Special rules:
        - Do not use dot (.) or comma (,) as thousand separator in the field "valor". For example 100,000 becomes 100000.
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
                    "categoria": "<one of the valid categories>",
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for the most recent period in yyyy-mm-dd>", ,
                }
            ],
            "estado_resultado_integral": [
                {
                    "categoria": "<one of the valid categories>",
                    "cuenta": "<account name>",
                    "valor": "<value for each period>",
                    "fecha": "<date for the most recent period in yyyy-mm-dd>" ,
                }
            ]

        -The expected output is a JSON structured according to the described format. 
        -No additional text should be included before or after the JSON, nor labels like "Here is the JSON" or explanatory comments.
        Special rules:

        - Do not use dot (.) or comma (,) as thousand separator in the field "valor". For example 100,000 becomes 100000.
        - If an account like "Activo Fijo" is found, classify it as "Activo Corriente."
        - If an account like "Pasivo no circulante" is found, classify it as "Pasivo No Corriente."
        -If an account like "capital" or Capital is found, classify it as "Patrimonio" category.

        """
        data = ollama_chat(consulta_inicial, system_message_extraction, vault_embeddings, vault_content, ollama_model, conversation_history)
        
        if data.endswith("}"):
            cuentas_detectadas = data
        else:
            cuentas_detectadas = data+"}"

        print(f"Cuentas detectadas inicialmente: {cuentas_detectadas}")
        if  is_valid_json(cuentas_detectadas):
            cuentas_detectadas_json = json.loads(cuentas_detectadas)
            consulta_validacion= """
            de las cuentas del documento recibido y segun la pregunta previa de las cuentas obtenidas Responder: 
            -la suma de activos y de pasivos más patrimonio, deben ser iguales, de lo contrario reprocesar y generar el json con el mismo formato la respuesta anterior con las cuentas correctas: Activos = Pasivo + Patrimonio"  
            
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
        with os.scandir(carpeta_pdf) as carpetas_bancos:
            for fichero in carpetas_bancos:
                carpeta_banco = os.path.join(carpeta_pdf,fichero.name,'IN')
                carpeta_salida = os.path.join(carpeta_pdf,fichero.name,'OUT')
                archivos_en_carpeta= os.listdir(carpeta_banco)
                entidad=fichero.name
                for index,archivo_pdf in enumerate(archivos_en_carpeta):
                    if archivo_pdf.endswith('.PDF') or archivo_pdf.endswith('.pdf'):
                        cuit = archivo_pdf.split("_")[0]
                        pdf_path = os.path.join(carpeta_banco,archivo_pdf)
                        pdf_path_out =os.path.join(carpeta_salida,archivo_pdf)
                        nombre_doc = archivo_pdf.replace(".pdf","")
                        print(f"Procesando archivo: {pdf_path}")
                        id_ejecucion = id_ejecucion+1
                        procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_onbase)
    else:
        pdf_path,pdf_path_out,archivo_pdf = obtener_archivo_banesco(carpeta_entrada,id_onbase,cuit,periodo,ubicacion_actual)
        nombre_doc = archivo_pdf.replace(".pdf","")
        print(f"Procesando archivo: {pdf_path}")
        procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_onbase)

def procesar_llm(id_lote,id_doc,cuit,pdf_path,pdf_path_out,nombre_doc,ubicacion_actual,sql_connector,args,id_onbase):
    
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
                paginas_encontradas=paginas_encontradas+str(i)+'.'
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

                # Establece hay_espaciado_inicial en True al inicio
                hay_espaciado_inicial = True
                # Recorre la información para determinar los saltos de línea
                for info in info_extraida:
                        y = info["box"][0][1]  # Tomar el valor Y de uno de los puntos, ya que son aproximadamente iguales
                        x = info["box"][0][0]  # Tomar el valor X del punto izquierdo


                        if abs(y - last_y) > tolerancia_vertical:
                            lines.append("\n")
                            hay_espaciado_inicial = True

                        #if((x - last_x) > tolerancia_horizontal):
                        # Agregar espaciado entre palabras basado en las coordenadas X
                        #    espaciado = max(0, int((x - last_x) / 20))  # Ajusta el divisor según sea necesario
                            #lines.append(" " * espaciado)
                        
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
                carpeta_texto_reconstruido = os.path.join(ubicacion_actual, 'LLMpdf/pdfs/text_reconstruido')
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
        if ruta_texto_reconstruido is not None and os.path.exists(ruta_texto_reconstruido):
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
                        "orden_cuenta": "<sequential number>",
                        "cuenta": "<account name>",
                        "valor": "<value for each period>",
                        "fecha": "<date for each period in yyyy-mm-dd>" ,
                    }
                ],
                "estado_resultado_integral": [
                    {
                        "categoria": "<one of the valid categories>",
                        "orden_cuenta": "<sequential number>",
                        "cuenta": "<account name>",
                        "valor": "<value for each period>",
                        "fecha": "<date for each period in yyyy-mm-dd>",
                    }
                ]
            }
            3. Do not include totals like "Total Activo" or "Total Pasivo" or "total de pavisos no corrientes" or " Total del Patrimonio".
            4. Reset "orden_cuenta" for each section.
            """,
            """
            Special rules:
            - If an account like "Activo Fijo" is found, classify it as "Activo Corriente."
            - If an account like "Pasivo no circulante" is found, classify it as "Pasivo No Corriente."
            - If an account like "capital" is found, classify it as "Patrimonio" category.
            """,
            "Respond in Spanish, using the JSON format described, and ensure accuracy in field assignment."
            ]
            lpa_input = """
            Respond only based on the information extracted in Relevant Context. Respond in a structured JSON format and WITHOUT ADDITIONAL TEXT, using the expected fields in the "estado_de_situacion_financiera" and "estado_resultado_integral" structures.
            - Extracted accounts must not exceed 100 in "orden_cuenta".
            -Respond only with valid JSON. Do not include any additional text or escape characters like \n' or '\'

            The response must include two JSON blocks:
                - "estado_de_situacion_financiera": for accounts related to the balance sheet, within the categories {"Activo Corriente", "Activo No Corriente", "Pasivo Corriente", "Pasivo No Corriente", "Patrimonio"}.
                - "estado_resultado_integral": for accounts related to comprehensive income results.

            -Do not include accounts like totals like "Total Activo" or "Total Pasivo" or "total de pavisos no corrientes" or " Total del Patrimonio".
            response example:
            {
            "estado_de_situacion_financiera": [
                {"categoria": "Activo Corriente", "orden_cuenta": 1, "cuenta": "Caja y Bancos", "valor": "prueba", "fecha": "2021-12-31"},
                {"categoria": "Activo No Corriente", "orden_cuenta": 2, "cuenta": "Mobiliario y Equipo", "valor":  "41250.00", "fecha": "2021-12-31" },
                {"categoria": "Pasivo Corriente", "orden_cuenta": 3, "cuenta": "Cuentas por Pagar", "valor": "71000.00", "fecha": "2021-12-31" },
                {"categoria": "Patrimonio", "orden_cuenta": 4, "cuenta": "Capital en Acciones", "valor": "10000.00", "fecha": "2021-12-31" }
            ],
            "estado_resultado_integral": [
                {"categoria": "Ingresos", "orden_cuenta": 1, "cuenta": "Ventas", "valor": "500000.00", "fecha": "2020-06-30",
                {"categoria": "Gastos", "orden_cuenta": 2, "cuenta": "Costo de Ventas", "valor": "100000.00", "fecha": "2020-06-30"}
            ]
            }

            -The expected output is a JSON structured according to the described format. 
            -No additional text should be included before or after the JSON, nor labels like "Here is the JSON" or explanatory comments.
            Special rules:
            - If an account like "Activo Fijo" is found, classify it as "Activo Corriente."
            - If an account like "Pasivo no circulante" is found, classify it as "Pasivo No Corriente."
            - If an account like "capital" is found, classify it as "Patrimonio" category.
            """
            try:    
                data = ollama_chat_with_validation(lpa_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
                #print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
            #print(NEON_GREEN + "Response: \n\n" + response["estado_de_situacion_financiera"] + RESET_COLOR)
            #print (response[0])
                periodo_procesado = data["estado_de_situacion_financiera"][0]["fecha"]
                print(f"Periodo procesado: {periodo_procesado}")
            except ValueError as e:
                print(f"Error: {e}")
            nombre_cliente='prueba'

            # Carga el string JSON como un objeto Python (en este caso, una lista de diccionarios)
            #if response.endswith("}"):
            #    data = json.loads(response)
            #else:
             #   data = json.loads(response+"}")
            #print(data.items())
           # periodo_procesado=data["estado_de_situacion_financiera"][0]["fecha"]
                #estado cero porque no se ha procesado en la aplicacion
            estado=0
            cuenta_reclasificada='' 
            selected_columns = []
            selected_columns_results = []

            for tipo_estado, items in data.items():   
                print(f"Tipo cuenta: {tipo_estado}")
                if(tipo_estado=='estado_de_situacion_financiera'):
                    for item in items:
                        cuenta      =   item["cuenta"]
                        categoria   =   item["categoria"]
                        valor       =   item["valor"] 
                        periodo     =   item["fecha"]
                        cuenta_reclasificada=''
                        categoria_reclasificada=''
                        diferencia_periodos='pendiente'
                        tipo_periodo='pendiente'
                        
                        cuenta_reclasificada,categoria_reclasificada = obtener_reclasificacion(cuenta,categoria,diccionario_cuentas)

                        selected_columns.append((nombre_cliente,
                                                cuit,
                                                categoria,
                                                cuenta,
                                                cuenta_reclasificada,
                                                categoria_reclasificada,
                                                item["valor"], 
                                                item["fecha"],
                                                tipo_periodo,
                                                diferencia_periodos,
                                                estado
                                                ))
                        
                elif(tipo_estado=='estado_resultado_integral'):
                    for item in items:
                        print(item)
                        categoria   =   item["categoria"]
                        cuenta      =   item["cuenta"] 
                        valor       =   item["valor"] 
                        periodo     =   item["fecha"]
                        #print(f"{categoria}, Valor: {valor}, Periodo: {periodo}")
                        selected_columns_results.append((nombre_cliente,
                                                cuit,
                                                categoria,
                                                cuenta,
                                                item["valor"], 
                                                item["fecha"],
                                                estado
                                                ))             
            df_balances = pd.DataFrame(selected_columns, columns=['cliente','cuit','categoria', 'cuenta', 'cuenta_reclasificada','categoria_reclasificada', 'valor', 'periodo','tipo_periodo','diferencia_periodos','estado'])
            df_financiero = pd.DataFrame(selected_columns_results, columns=['cliente','cuit','categoria','cuenta','valor', 'periodo','estado'])
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            
            df_balances['mm'] = pd.to_datetime(df_balances['periodo']).dt.month
            diferencia_meses = df_balances['mm'].iloc[0]
            periodo_archivo = df_balances['periodo'].iloc[0]
            print(diferencia_meses)
            
            #Consulta cliente 
            cuit,id_cliente,nombre_cliente = consultar_cliente(cuit,sql_connector,id_onbase)

            #obtiene el maximo id balance temporal
            id_balance_max=get_maximo_id_balance_temp(sql_connector)
            #Creacion de id_balance
            #####(Falta ajustar cuando hayan multiples periodos)
            insert_inicial(id_balance_max,id_cliente,nombre_cliente,periodo_archivo,id_lote,id_doc,cuit,nombre_doc,paginas_encontradas,sql_connector,id_onbase)
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
                'patrimonio': diccionario_var_patrimonio
            }
            # Iterar sobre las filas del DataFrame
    for index, row in df_balances.iterrows():
        categoria = str(row['categoria_reclasificada']).lower()

        if categoria in categoria_a_diccionario:
            insert_tl_lpa_cuentas, update_tb_lpa_balance, insert_t_balance = procesar_fila(
                row, index, id_doc, id_balance_max, categoria_a_diccionario, diccionario_resultantes,categoria
            )
            print(f'insert_tl_lpa_cuentas: {insert_tl_lpa_cuentas}')
            print(f'update_tb_lpa_balance: {update_tb_lpa_balance}')
            print(f'insert_t_balance: {insert_t_balance}')
            sql_connector.insert_data(insert_tl_lpa_cuentas)
            sql_connector.insert_data(update_tb_lpa_balance)
            sql_connector.insert_data(insert_t_balance)
        else:

            # Si la categoría no está mapeada
            print(f"Categoría desconocida: {categoria}")
            insert_tl_lpa_cuentas, insert_t_balance = procesar_fila(
                row, index, id_doc, id_balance_max, categoria_a_diccionario, diccionario_resultantes,'None'
            )
            print(f'insert_tl_lpa_cuentas: {insert_tl_lpa_cuentas}')
            print(f'insert_t_balance: {insert_t_balance}')
            sql_connector.insert_data(insert_tl_lpa_cuentas)
            sql_connector.insert_data(insert_t_balance)
    #incrementa el id documento
    id_doc=id_doc+1
    #imprime las cuentas extraidas
    
    print(df_financiero)     
    print(f"-------------------")


# Función para manejar inserciones y actualizaciones según reglas
def procesar_fila(row, index, id_doc, id_balance_max, categoria_diccionario, diccionario_resultantes,categoria):
    cuenta = row['cuenta']
    valor = row['valor'] if row['valor'] != '' else 0
    cuenta_reclasificada = str(row['cuenta_reclasificada']) if str(row['cuenta_reclasificada']) != '' else 'None'

    # Determinar diccionario a usar
    diccionario = categoria_diccionario.get(categoria, {})
    variable = diccionario.get(cuenta_reclasificada, diccionario_resultantes.get(cuenta_reclasificada, None))

    if variable:
        # Insertar en TL_LPA_CUENTAS
        insert_tl_lpa_cuentas = f"""
            Insert into TL_LPA_CUENTAS 
            (id_documento, id_orden, cuenta_extraida, categoria_clasificada, cuenta_clasificada, cuenta_final, variable, nota, fecha_modificacion)
            values({id_doc}, {index}, '{cuenta}', '{categoria}', '{cuenta_reclasificada}', '{cuenta_reclasificada}', '{variable}', 'NA', '{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}');
        """
        # Actualizar TB_LPA_BALANCE
        update_tb_lpa_balance = f"""
            Update TB_LPA_BALANCE 
            set {variable} = {valor}
            where id_balance = {id_balance_max};
        """
        # Insertar en T_BALANCE
        insert_t_balance = f"""
            Insert into T_BALANCE 
            (id_balance, id_cuenta, cuenta_inicial, variable, cuenta_final, valor_inicial, valor_final, id_documento)
            values({id_balance_max}, {index}, '{cuenta}', '{variable}', '{cuenta_reclasificada}', {valor}, {valor}, {id_doc});
        """
        return insert_tl_lpa_cuentas, update_tb_lpa_balance, insert_t_balance
    else:
        # Insertar en TL_LPA_CUENTAS sin variable
        insert_tl_lpa_cuentas = f"""
            Insert into TL_LPA_CUENTAS 
            (id_documento, id_orden, cuenta_extraida, categoria_clasificada, nota, fecha_modificacion,cuenta_clasificada, cuenta_final)
            values({id_doc}, {index}, '{cuenta}', '{categoria}', 'NA', '{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}','None','None');
        """
        # Insertar en T_BALANCE sin variable
        insert_t_balance = f"""
            Insert into T_BALANCE 
            (id_balance, id_cuenta, cuenta_inicial, valor_inicial, valor_final, id_documento)
            values({id_balance_max}, {index}, '{cuenta}', {valor}, {valor}, {id_doc});
        """
        return insert_tl_lpa_cuentas, insert_t_balance

#Consultar cliente existe y extrae id_cliente
def consultar_cliente(cuit,sql_connector,id_onbase):
    #Sgina nombre campos según banesco o petersen
    if id_onbase is None:
        sql_cuit='codigociiu'
        sql_nombre='nombrecomercial'
    else:
        sql_cuit='num_doc'
        sql_nombre='nombre'
    nombre_cliente ='PRUEBA'
    select_cliente = f"Select id_cliente,{sql_nombre},{sql_cuit} from TB_cliente where {sql_cuit} ='{cuit}'"
    print(select_cliente)
    consulta_cliente=sql_connector.read_data(select_cliente)
    cuit='11111'
    id_cliente='11111'
    if len(consulta_cliente) == 0:
        print('Cliente no existe')
    else:        
        for row in consulta_cliente:
            if id_onbase is None:
                cuit = row.codigociiu
            #cuit = row.num_doc
                id_cliente = row.id_cliente
            #nombre_cliente = row.nombre
                nombre_cliente = row.nombrecomercial
            else:
                cuit = row.num_doc
                id_cliente = row.id_cliente
                nombre_cliente = row.nombre
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
def insert_inicial(id_balance_max,id_cliente,nombre_cliente,periodo_archivo,id_lote,id_doc,cuit,nombre_doc,paginas_encontradas,sql_connector,id_onbase):
    
    if id_onbase is None:
        print('petersen')
        """num_calificacion
        nombre_cliente
        id_modelo_bal
        estatus
        moneda
        unidad_monetaria
        tasa_cambio
        norma_contable
        auditor"""
    else:
        print ("banesco")


    insert_tb_lpa_balance = "Insert into TB_LPA_BALANCE (id_balance,num_calificacion,id_cliente,nombre_cliente,id_rating,id_modelo_bal,estatus,moneda,unidad_monetaria,tasa_cambio,mes,estado_mes,estado,usuario_creacion,fecha_creacion,fecha_asignacion,norma_contable,auditor) "\
        f"values ({id_balance_max},'1','{id_cliente}','{nombre_cliente}',{id_balance_max},1,'Auditado','USD','Absolutas',1,'12','Individual','A','LPA','{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}','{periodo_archivo}','IFRS','NA')"
    print(insert_tb_lpa_balance)
    sql_connector.insert_data(insert_tb_lpa_balance)

    insert_tl_lpa_creacion = "Insert into TL_LPA_CREACION (id_ejecucion,id_documento,id_periodo,id_balance_temp,fecha_asignacion,estado) "\
        f"values ({id_lote},{id_doc},1,{id_balance_max},'{periodo_archivo}','A')"
    print(insert_tl_lpa_creacion)
    sql_connector.insert_data(insert_tl_lpa_creacion)
    
    ##to do (Check)(Revisar)
    #Cambiar el id_documento
    #####################Falta crear id doc
    insert_TL_BALANCE = "Insert into TL_BALANCE (id_ejecucion,id_documento, id_periodo, cuit, nombre_documento, fecha_asignacion,pag_procesadas, estado,proceso, observacion,fecha_creacion) "\
    f"values({id_lote},{id_doc},1,'{cuit}','{nombre_doc}','{periodo_archivo}','{paginas_encontradas}','P','A','Se procesó correctamente','{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')"
    print(insert_TL_BALANCE) 
    sql_connector.insert_data(insert_TL_BALANCE) 

def cargar_diccionarios(sql_connector):
    #Consulta diccionario de cuentas
    select_statement = "Select cuenta,categoria,cuenta_reclasificada,categoria_reclasificada from TB_DICCIONARIO_CUENTAS;"
    result = sql_connector.read_data(select_statement)
    for row in result:
        diccionario_cuentas.append((str(row[0]),str(row[1]),str(row[2]),str(row[3])))
    #Llenado de duplas cuenta - var
    select_variables ="Select variable,categoria,cuenta,id_modelo from TL_LPA_MODELO"
    result_varianles = sql_connector.read_data(select_variables)
    for registro in result_varianles:
        if registro[1] == 'activo corriente':
            diccionario_var_activo_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1] == 'activo no corriente':
            diccionario_var_activo_no_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1] == 'pasivo corriente':
            diccionario_var_pasivo_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1] == 'pasivo no corriente':
            diccionario_var_pasivo_no_corriente[registro[2]] = registro[0]
            diccionario_resultantes[registro[2]] = registro[0]
        elif registro[1] == 'patrimonio':
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
    
    # Ruta a la carpeta que contiene los archivos PDF
    carpeta_pdf = os.path.join(ubicacion_actual, 'MLPA/BALANCES')

    return carpeta_pdf