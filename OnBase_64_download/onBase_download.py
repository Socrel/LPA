import requests
import base64

GRANT_TYPE = 'client_credentials'
VALIDITY_PERIOD ='3600'
CONTENT_TYPE='application/x-www-form-urlencoded'
URL_TOKEN = "https://qa-auth-ob.banesco.com.pa/oauth2/token?grant_type="+GRANT_TYPE+"&validity&validity_period="+VALIDITY_PERIOD
URL_BASE64 ="https://qa.api.ob.banesco.com.pa/APIUtil/v1/documents/ID"
username = "6cbk7drlvbc88vdgvgo1tg1r3o"
password ="b9vti5n8pbn1cn2gr35vlgq8s3uq7mm1j2d79ih0u1jjb4hkd3m"

DATA = {
    "title": "Example Title",
    "body": "Content of a new post",
    "userId": 1
}

def get_token():
    headers_token = {"Content-Type": CONTENT_TYPE}
    response = requests.post(URL_TOKEN,headers=headers_token, auth=(username, password))

    if response.status_code == 200:
        print("token obtenido")
        data = response.json()
        token = data["access_token"]
        return token

    else:
        print('Error in the request, details:', response.text)

def get_base64(auth_token):
    headers_base64 ={'Authorization': f'Bearer {auth_token}'}
    json_data ={
        "DocumentHandle":"73075079",
        "ReturnBase64":"true"
        }
    response = requests.post(URL_BASE64,headers=headers_base64,json=json_data)

    if response.status_code == 200:
        data = response.json()
        base64 = data["Documents"][0]["Base64"]
        print ("base 64 obtenido")
        return base64

    else:
        print('Error in the request, details:', response.text)

def convert_base64_to_pdf(base64_data):

    # Decodifica el contenido base64
    pdf_data = base64.b64decode(base64_data)

    # Guarda el archivo PDF
    with open("archivo_salida.pdf", "wb") as pdf_file:
        pdf_file.write(pdf_data)

basearchivo = get_base64(get_token())
convert_base64_to_pdf(basearchivo)