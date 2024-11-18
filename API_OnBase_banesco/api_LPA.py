"""
Descarga documentos de onbsae y los transforma de base 64 a pdf
"""
from flask import Flask, jsonify, request,  abort ,send_from_directory
from flask_oauthlib.provider import OAuth2Provider
from werkzeug.security import gen_salt
from datetime import datetime
from functools import wraps
import os
from connection_IVA_BBDD import SQLConnector
import logging
import onBase_download as onBase

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
oauth = OAuth2Provider(app)

now = datetime.now()

server = 'ec2-18-191-95-248.us-east-2.compute.amazonaws.com'
database = 'BERAPPRAT'
user = 'sa'
password = r'#MSApprating#'

# Datos de ejemplo
lista_urls = []
tokens = {}

class OAuth2Token:
    def __init__(self, access_token, token_type='Bearer', scope=''):
        self.access_token = access_token
        self.token_type = token_type
        self.scope = scope
        
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            response_body = {
            'Succes':False,
            'Message':'Token faltante o inválido',
            'Status':401,
            'Data':None,
            }
            return jsonify(response_body), 401
        token = token[len('Bearer '):]
        if token not in tokens:
            response_body = {
            'Succes':False,
            'Message':'Token no autorizado',
            'Status':401,
            'Data':None,
        }
            return jsonify(response_body), 401
        return f(*args, **kwargs)
    return decorated_function
# Manejo de errores generales
@app.errorhandler(Exception)
def handle_exception(e):
    response_body = {
            'Succes':False,
            'Message':'Cuerpo de solicitud incorrecto',
            'Status':500,
            'Data':f'Detalle del error :{str(e)}',
        }
    return jsonify(response_body), 500

# Manejo de errores específicos
@app.errorhandler(404)
def handle_404_error(e):
    return jsonify({"error": "Recurso no encontrado", "mensaje": str(e)}), 404

@app.errorhandler(400)
def handle_400_error(e):
    response_body = {
            'Succes':False,
            'Message':'Cuerpo de solicitud incorrecto',
            'Status':400,
            'Data':str(e)
        }
    return jsonify(response_body), 400


@oauth.token_handler
def token_handler():
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token[len('Bearer '):]
        if token in tokens:
            return OAuth2Token(access_token=token)
    abort(401)


@app.route('/lpa/onbase', methods=['POST'])
@token_required
def get_links():
    print("entra al endpoint")
    try:
        if not request.json or 'User' not in request.json or 'Documents' not in request.json or 'OnDemand' not in request.json:
            raise ValueError("Faltan datos necesarios en la solicitud")
        
        body = {
            'User':request.json.get('User', ''),
            'Documents':request.json.get('Documents', ''),
            'OnDemand':request.json.get('OnDemand', ''),
        }
        inserta_encolado(body)
        response_body = {
            'Succes':True,
            'Message':'Datos enviados correctamente',
            'Status':201,
            'Data':body['Documents'],
        }
        return jsonify(response_body), 201
    except ValueError as e:
        return handle_400_error(e)
    except Exception as e:
        return handle_exception(e)

@app.route('/lpa/token', methods=['POST'])
def issue_token():
    if not request.json or 'username' not in request.json:
        return jsonify({'error': 'Bad request'}), 400
    
    username = request.json['username']
    token = gen_salt(32)
    tokens[token] = username
    return jsonify({'access_token': token, 'token_type': 'Bearer'})

@app.route("/")
def index():
    return "<h1>Hello</h1>"

@app.route('/lpa/onbase/existe', methods=['GET'])
@token_required
def document_exist():
    print("entra al endpoint")
    try:
        if not request.json or 'DocumentId' not in request.json :
            raise ValueError("Faltan datos necesarios en la solicitud")
        
        body = {
            'DocumentId':request.json.get('DocumentId', '')
        }
        
        if onBase.consultar_archivo(body['DocumentId']):
            response_body = {
                'Succes':True,
                'Message':'Documento existe en OnBase',
                'Status':200,
                'Data':body['DocumentId'],
            }
        else:
            response_body = {
                'Succes':False,
                'Message':'Documento No eciste',
                'Status':200,
                'Data':None,
            }
        return jsonify(response_body), 200
    except ValueError as e:
        return handle_400_error(e)
    except Exception as e:
        return handle_exception(e)

def inserta_encolado(body):
    select_id_max = 'Select isnull(max(id_onbase),0) as id_onbase from TB_DOCUMENTOS_ENCOLADOS'
    consulta_max_id=sql_connector.read_data(select_id_max)
    for row in consulta_max_id:
        id_onbase = row.id_onbase +1
    for document in body['Documents']:
        insert_statement_inic = "Insert into TB_DOCUMENTOS_ENCOLADOS (id_onbase, user_onbase, client_number, type_id,identification,type_person,code_country,client_name,code_industry,id_doc,periodo,on_demand,fecha_creacion) "\
            f"values({id_onbase},'{body['User']}','{document['ClientNumber']}','{document['TypeId']}','{document['Identification']}','{document['TypePerson']}','{document['CodeCountry']}','{document['Name']}','{document['CodeIndustry']}',{document['IdDoc']},{document['Periodo']},{body['OnDemand']},'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')"
        print(body['OnDemand'])
        if body['OnDemand']=='1':
            onBase.descargar_archivo(id_onbase,document['IdDoc'],document['Identification'],document['Periodo'],'onDemand')
        else:
            onBase.descargar_archivo(id_onbase,document['IdDoc'],document['Identification'],document['Periodo'],'batch')
        sql_connector.insert_data(insert_statement_inic)
    return insert_statement_inic



if __name__ == '__main__':
    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()
    app.run(debug=True,ssl_context='adhoc',host='0.0.0.0')
    """
    server_port = os.environ.get('PORT', '5000')
    app.run(debug=False, port=server_port, host='0.0.0.0')"""