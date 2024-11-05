import schedule
import time as tm
from datetime import time,timedelta,datetime
import ocr_IVA 
from connection_IVA_BBDD import SQLConnector
from subprocess import Popen
import os



def job():
    print("Empieza el proceso", flush=True)
    
    server = 'ec2-3-144-70-242.us-east-2.compute.amazonaws.com'
    database = 'BBDD_MLAP'
    user = 'sa'
    password = r'#MSApprating#'

    sql_connector = SQLConnector(server, database, user, password)
    sql_connector.connect()

    select_verification ='Select count(1) as cuenta from TL_PROCESOS_EJECUTADOS where estado=1'
    select_statement = "Select max(id_ejecucion) as id_ejecucion from TL_PROCESOS_EJECUTADOS;" 
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
            insert_statement_inic = "Insert into TL_PROCESOS_EJECUTADOS (id_ejecucion, fecha_inicio, estado) "\
                f"values({id_ejecucion},'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',1)"
            print(insert_statement_inic)
            sql_connector.insert_data(insert_statement_inic)
            ocr_IVA.run(id_ejecucion,sql_connector)
            insert_statement_fin = "Update TL_PROCESOS_EJECUTADOS "\
                f"Set fecha_fin='{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',estado= 0"\
                f"where id_ejecucion={id_ejecucion}"
            sql_connector.insert_data(insert_statement_fin)
                
        else:
            print ("Hay otro proceso en ejecución")


    #Llamado transformaciones
    #param1=r'/file:C:\Users\David\Downloads\data-integration\archivos\Prueba_1.ktr'
    #param2=r'/level:Basic'
    #p = Popen([r'C:\Users\David\Downloads\data-integration\Pan.bat', param1, param2])
    #output, errors = p.communicate()
    #p.wait()
            
    #---------------------------------------------------------------------------------------------------
    #Llamado jobs
    param1=r'/file:C:\Users\svbmgms\Documents\data-integration\archivos\Job_MLAP.kjb'
    param2=r'/level:Basic'
    p = Popen([r'C:\Users\svbmgms\Documents\data-integration\kitchen.bat', param1, param2])
    output, errors = p.communicate() 
    p.wait()
    
    sql_connector.close()
    
schedule.every().day.at("16:56","America/Bogota").do(job)

while True:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    schedule.run_pending()
    tm.sleep(1)
