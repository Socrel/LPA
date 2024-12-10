"""
Descarga documentos de onbsae y los transforma de base 64 a pdf
"""
from flask import Flask, jsonify, request,  abort ,send_from_directory
from flask_oauthlib.provider import OAuth2Provider
from werkzeug.security import gen_salt,generate_password_hash, check_password_hash
from datetime import datetime,timezone,timedelta
from functools import wraps
import os
from connection_IVA_BBDD import SQLConnector
import logging
import onBase_download as onBase
import jwt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'NNIlOQLozn2kSO1RGWPTXDtjUI7asPc3iMCQcSSZVSRU0EcPJnZpTL4vtx3KpnJ'
oauth = OAuth2Provider(app)

users = {
    "user1": generate_password_hash("password"),
    "admin": generate_password_hash("XE7BtJVpLDWHSc6"),
}

now = datetime.now()
"""
server = '10.40.192.77'
database = 'BBDD_MSRating_Banesco'
user = 'usr_safi'
password = r'R4t3aws2o2$'
"""
server = 'ec2-18-191-95-248.us-east-2.compute.amazonaws.com'
database = 'BERAPPRAT'
user = 'sa'
password = r'#MSApprating#'

# Datos de ejemplo
lista_urls = []
tokens = {}

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'Success': False, 'Message': 'Token faltante o inválido', 'Status': 401}), 401
        
        token = token.split(" ")[1]  # Extraer el JWT del encabezado
        try:
            decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS512"])
            request.user = decoded['user']  # Guardar el usuario decodificado
        except jwt.ExpiredSignatureError:
            return jsonify({'Success': False, 'Message': 'Token expirado', 'Status': 401}), 401
        except jwt.InvalidTokenError:
            return jsonify({'Success': False, 'Message': 'Token inválido', 'Status': 401}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Manejo de errores generales
@app.errorhandler(Exception)
def handle_exception(e):
    print(str(e))
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
    response_body = {
            'Succes':False,
            'Message':'Pagina no encontrada',
            'Status':404,
            'Data':str(e)
        }
    return jsonify(response_body), 404

@app.errorhandler(400)
def handle_400_error(e):
    response_body = {
            'Succes':False,
            'Message':'Cuerpo de solicitud incorrecto',
            'Status':400,
            'Data':str(e)
        }
    return jsonify(response_body), 400

@app.route('/lpa/onbase', methods=['POST'])
@token_required
def get_links():
    print("entra al endpoint")
    try:
        if not request.json :
            raise ValueError("Cuerpo de solicitud vacío")

        data = {key.lower(): value for key, value in request.json.items()}
        
        if 'user' not in data or 'documents' not in data or 'ondemand' not in data:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        body = {
            'user':data.get('user', ''),
            'documents':data.get('documents', ''),
            'ondemand':data.get('ondemand', ''),
        }
        inserta_encolado(body)
        response_body = {
            'Succes':True,
            'Message':'Datos enviados correctamente',
            'Status':201,
            'Data':body['documents'],
        }
        return jsonify(response_body), 201
    except ValueError as e:
        return handle_400_error(e)
    except Exception as e:
        return handle_exception(e)

@app.route('/lpa/v1/token', methods=['POST'])
def issue_token_deprecate():
    if not request.json or 'username' not in request.json:
        return jsonify({'error': 'Bad request'}), 400
    
    username = request.json['username']
    token = gen_salt(32)
    tokens[token] = username
    return jsonify({'access_token': token, 'token_type': 'Bearer'})

@app.route('/lpa/token', methods=['POST'])
def issue_token():
    if not request.json or 'username' not in request.json or 'password' not in request.json:
        return jsonify({'Success': False, 'Message': 'Faltan datos en la solicitud', 'Status': 400}), 400
    
    username = request.json['username']
    password = request.json['password']
    
    # Validar credenciales
    if username not in users or not check_password_hash(users[username], password):
        return jsonify({'Success': False, 'Message': 'Credenciales inválidas', 'Status': 401}), 401
    
    expiration = datetime.now(timezone.utc) + timedelta(hours=1)  # Token válido por 1 hora
    token = jwt.encode({'user': username, 'exp': expiration}, app.config['SECRET_KEY'], algorithm="HS512")
    
    return jsonify({'access_token': token, 'token_type': 'Bearer', 'expires_in': 3600})


@app.route("/")
def index():
    return "<h1>Hello</h1>"

@app.route('/lpa/onbase/exist', methods=['POST'])
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
    for document in body['documents']:
        data = {key.lower(): value for key, value in document.items()}
        insert_statement_inic = "Insert into TB_DOCUMENTOS_ENCOLADOS (id_onbase, user_onbase, client_number, type_id,identification,type_person,code_country,client_name,code_industry,id_doc,periodo,on_demand,fecha_creacion,procesado) "\
            f"values({id_onbase},'{body['user']}','{data['clientnumber']}','{data['typeid']}','{data['identification']}','{data['typeperson']}','{data['codecountry']}','{data['name']}','{data['codeindustry']}',{data['iddoc']},{data['periodo']},{body['ondemand']},'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',0)"
        if body['ondemand']=='1':
            #onBase.descargar_archivo(id_onbase,data['iddoc'],data['identification'],data['periodo'],'onDemand')
            print('OnDemand')
        else:
            #onBase.descargar_archivo(id_onbase,data['iddoc'],data['identification'],data['periodo'],'batch')
            print('batch')
        sql_connector.insert_data(insert_statement_inic)
    return insert_statement_inic



if __name__ == '__main__':
    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()
    app.run(debug=True,host='0.0.0.0')
    """
    app.run(debug=True,ssl_context='adhoc',host='0.0.0.0')
    """
    """
    server_port = os.environ.get('PORT', '5000')
    app.run(debug=False, port=server_port, host='0.0.0.0')"""