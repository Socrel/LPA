import schedule
import time as tm
from datetime import time,timedelta,datetime
import localragv3 
from connection_IVA_BBDD import SQLConnector
from subprocess import Popen
import os



def job():
    print("Empieza el proceso", flush=True)
    
    server = 'ec2-18-191-95-248.us-east-2.compute.amazonaws.com'
    database = 'BERAPPRAT'
    user = 'sa'
    password = r'#MSApprating#'

    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()

#########################Definir el proceso ejecución on batch y online
    id_onbase =4
#######################################################################

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
                
                    id_doc = row.id_documento +1
                    localragv3.run(id_ejecucion,id_doc,id_onbase,sql_connector,'batch')
                    insert_statement_fin = "Update TH_EJECUCION "\
                        f"Set fecha_fin='{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',estado= 0"\
                        f"where id_ejecucion={id_ejecucion}"
                    sql_connector.insert_data(insert_statement_fin)
                
        else:
            print ("Hay otro proceso en ejecución")

    sql_connector.close()
    
def check_on_demand():
    print("Revisando documentos encolados para ejecución on-demand...", flush=True)

    server = 'ec2-18-191-95-248.us-east-2.compute.amazonaws.com'
    database = 'BERAPPRAT'
    user = 'sa'
    password = r'#MSApprating#'

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
            
            for row_2 in consulta_max_id_doc:
                id_doc = row_2.id_documento +1
                # Consulta para verificar si hay registros en TB_DOCUMENTOS_ENCOLADOS con on_demand = 1
                select_on_demand = 'SELECT id_onbase, user_onbase, client_number, type_id,identification,type_person,code_country,client_name,code_industry,id_doc,periodo,on_demand FROM TB_DOCUMENTOS_ENCOLADOS WHERE on_demand = 1'
                result = sql_connector.read_data(select_on_demand)
                if len(result)> 0:
                    for row_3 in result:
                        print("Se encontró un proceso on-demand. Ejecutando job...", flush=True)
                        localragv3.run(id_ejecucion,id_doc,row_3.id_onbase,row_3.identification,sql_connector,'batch')
                        insert_statement_fin = "Update TH_EJECUCION "\
                        f"Set fecha_fin='{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',estado= 0"\
                        f"where id_ejecucion={id_ejecucion}"
                        sql_connector.insert_data(insert_statement_fin)
                        update_statement = f'UPDATE TB_DOCUMENTOS_ENCOLADOS set on_demand=0 where id_onbase={row_3.id_onbase}'
                        sql_connector.read_data(update_statement)
                        
                        
                else:
                    print("No hay procesos de prioridad alta encolados.")

                    
                
        else:
            print ("Hay otro proceso en ejecución")

    sql_connector.close()

    sql_connector.close()

schedule.every().day.at("17:29","America/Bogota").do(job)

#schedule.every(1).minutes.do(check_on_demand)

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    schedule.run_pending()
    tm.sleep(1)
