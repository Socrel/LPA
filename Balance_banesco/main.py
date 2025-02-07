import schedule
import time as tm
from datetime import time,timedelta,datetime
import localragv3 
from connection_IVA_BBDD import SQLConnector
from subprocess import Popen
import os
from dotenv import load_dotenv

dotenv_path = "/home/usr_ocr_dev/entorno_api/.env"
# Carga las variables de entorno desde el archivo .env
load_dotenv(dotenv_path=dotenv_path)

server = os.getenv("SERVER_HOST")
database = os.getenv("DATABASE_NAME")
user = os.getenv("USER_DATABASE")
password = os.getenv("PASSWOR_DATABASE")
HORA_EJECUCION = os.getenv("HORA_EJECUCION")
EJECUCION_ON_DEMAND = os.getenv("EJECUCION_ON_DEMAND")
"""
server = 'ec2-18-191-95-248.us-east-2.compute.amazonaws.com'
database = 'BERAPPRAT'
user = 'sa'
password = r'#MSApprating#'
"""
def job():
    print("Empieza el proceso", flush=True)
    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()

    select_verification ='Select count(1) as cuenta from TH_EJECUCION where estado=1'
    select_statement = "Select max(id_ejecucion) as id_ejecucion from TH_EJECUCION;" 
    verification = sql_connector.read_data(select_verification)
    for row in verification:
        if row.cuenta == 0:
            result = sql_connector.read_data(select_statement)
            for row in result:
                if row.id_ejecucion is None:
                    id_ejecucion = 1
                else:
                    id_ejecucion =row.id_ejecucion +1
            print (id_ejecucion)
            insert_statement_inic = "Insert into TH_EJECUCION (id_ejecucion, fecha_inicio, estado) "\
                f"values({id_ejecucion},'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',1)"
            print(insert_statement_inic)
            sql_connector.insert_data(insert_statement_inic)
            select_id_documento = 'Select isnull(max(id_documento),0) as id_documento from TL_BALANCE'
            
            consulta_max_id_doc=sql_connector.read_data(select_id_documento)
            
            for row in consulta_max_id_doc:
                select_on_demand = 'SELECT id_onbase, user_onbase, client_number, type_id,identification,type_person,code_country,client_name,code_industry,id_doc,periodo,on_demand FROM TB_DOCUMENTOS_ENCOLADOS WHERE procesado = 0'
                result_TB_DOCUMENTOS_ENCOLADOS = sql_connector.read_data(select_on_demand)
                for row_2 in result_TB_DOCUMENTOS_ENCOLADOS: 
                    id_doc = row.id_documento +1
                    localragv3.run(id_ejecucion,id_doc,row_2.id_onbase,row_2.identification,sql_connector,'batch',row_2.periodo)
                    update_statement = f'UPDATE TB_DOCUMENTOS_ENCOLADOS set on_demand=0, procesado=1 where id_onbase={row_2.id_onbase}'
                    sql_connector.insert_data(update_statement)
                    insert_statement_fin = "Update TH_EJECUCION "\
                        f"Set fecha_fin='{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',estado= 0"\
                        f"where id_ejecucion={id_ejecucion}"
                    sql_connector.insert_data(insert_statement_fin)
                
        else:
            print ("Hay otro proceso en ejecución")

    sql_connector.close()
    
def check_on_demand():
    print("Revisando documentos encolados para ejecución on-demand...", flush=True)

    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()
    # Consulta para verificar si hay registros en TB_DOCUMENTOS_ENCOLADOS con on_demand = 1
    select_on_demand = 'SELECT id_onbase, user_onbase, client_number, type_id,identification,type_person,code_country,client_name,code_industry,id_doc,periodo,on_demand FROM TB_DOCUMENTOS_ENCOLADOS WHERE procesado = 0 and on_demand = 1'
    result_TB_DOCUMENTOS_ENCOLADOS = sql_connector.read_data(select_on_demand)
    if len(result_TB_DOCUMENTOS_ENCOLADOS)> 0:
        select_verification ='Select count(1) as cuenta from TH_EJECUCION where estado=1'
        select_statement = "Select max(id_ejecucion) as id_ejecucion from TH_EJECUCION;" 
        verification = sql_connector.read_data(select_verification)
        for row in verification:
            if row.cuenta == 0:
                result = sql_connector.read_data(select_statement)
                for row in result:
                    if row.id_ejecucion is None:
                        id_ejecucion = 1
                    else:
                        id_ejecucion =row.id_ejecucion +1
                print (id_ejecucion)
                insert_statement_inic = "Insert into TH_EJECUCION (id_ejecucion, fecha_inicio, estado) "\
                    f"values({id_ejecucion},'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',1)"
                print(insert_statement_inic)
                sql_connector.insert_data(insert_statement_inic)
                select_id_documento = 'Select isnull(max(id_documento),0) as id_documento from TL_BALANCE'
                
                consulta_max_id_doc=sql_connector.read_data(select_id_documento)
                
                for row_2 in consulta_max_id_doc:
                    id_doc = row_2.id_documento +1
                    
                    
                    for row_3 in result_TB_DOCUMENTOS_ENCOLADOS:
                        print("Se encontró un proceso on-demand. Ejecutando job...", flush=True)
                        validar_crear_cliente(row_3,sql_connector)
                        localragv3.run(id_ejecucion,id_doc,row_3.id_onbase,row_3.identification,sql_connector,'onDemand',row_3.periodo)
                        update_statement = f'UPDATE TB_DOCUMENTOS_ENCOLADOS set on_demand=0, procesado=1 where id_onbase={row_3.id_onbase}'
                        sql_connector.insert_data(update_statement) 
                        insert_statement_fin = "Update TH_EJECUCION "\
                            f"Set fecha_fin='{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',estado= 0"\
                            f"where id_ejecucion={id_ejecucion}"
                        sql_connector.insert_data(insert_statement_fin)
                                    
            else:
                print ("Hay otro proceso en ejecución")
    else:
        print("No hay procesos de prioridad alta encolados.")
    sql_connector.commit()
    sql_connector.close()

schedule.every().day.at(f"{HORA_EJECUCION}","America/Bogota").do(job)
schedule.every(EJECUCION_ON_DEMAND).seconds.do(check_on_demand)

def validar_crear_cliente(cliente,sql_connector):
    validacion_statement = f"Select count(1) as contador from TB_CLIENTE where num_doc='{cliente.identification}' or cod_cliente='{cliente.identification}'"
    verification = sql_connector.read_data(validacion_statement)
    for row in verification:
        if row.contador == 0:
            print('Creando cliente')
            select_id_cliente='Select isnull(max(id_cliente),0) as id_cliente from TB_CLIENTE'
            result=sql_connector.read_data(select_id_cliente)
            for id_cl in result:
                    if id_cl.id_cliente is None:
                        id_cliente = 1
                    else:
                        id_cliente = id_cl.id_cliente+1
            
            select_pais = f"Select Descricpcion from TP_CATALOGOS WHERE TipoCatalogo='PAIS' and Codigo= '{cliente.code_country}'"
            print(select_pais)
            pais_consulta = sql_connector.read_data(select_pais)
            pais_descripcion=pais_consulta[0].Descricpcion

            select_industria = f"Select Descricpcion from TP_CATALOGOS WHERE TipoCatalogo='TIPOINDUSTRIA' and Codigo= '{cliente.code_industry}'"
            print(select_industria)
            industria_consulta = sql_connector.read_data(select_industria)
            industria_descripcion=industria_consulta[0].Descricpcion
            insert_cliente = f"Insert into TB_CLIENTE (id_cliente,cod_cliente,tipo_doc,num_doc,cod_act_economica,act_economica,cod_pais_residencia,pais_residencia,tip_persona,nombre,estado,fecha_creacion,total_ventas,producto_activo) "\
                                f"values ({id_cliente},'{cliente.client_number}','{cliente.type_id}','{cliente.identification}',{cliente.code_industry},'{industria_descripcion}','{cliente.code_country}','{pais_descripcion}','{cliente.type_person}','{cliente.client_name}','A','{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',0,0)"
            print(insert_cliente)
            sql_connector.insert_data(insert_cliente)
        else:
            print('Cliente existe')

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    schedule.run_pending()
    tm.sleep(1)
