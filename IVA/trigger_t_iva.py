import pyodbc
from datetime import date
# Configuración de la conexión a la base de datos
server = 'BSFSBALSQL01T.petersen.corp'
database = 'BSFAPPRAT'
user = 'BsfBalance01T'
password = r'BsfB41ance2022'


conn = pyodbc.connect('DRIVER={SQL Server};'
                      f'SERVER={server};'
                      f'DATABASE={database};'
                      f'UID={user};'
                      f'PWD={password}')

cursor = conn.cursor()

def trigger_logic(id_ejecucion,sub_estado_proceso,estado,cuit,periodo,valor_IVA_leido,confianza,
                  fecha_creacion,banco,nombre_documento,formato,observacion,fecha_iva_completa):
    try:
        # Iniciar transacción
        cursor.execute("BEGIN TRANSACTION")
        
        # Truncar la tabla T_IVA_TRIGGER_P
        cursor.execute("TRUNCATE TABLE T_IVA_TRIGGER_P")

        # Condición para sub_estado_proceso = 'CUIT_PERIODO'
        if sub_estado_proceso == 'CUIT_PERIODO':
            cursor.execute("SELECT 1 FROM TB_CLIENTE WHERE CodigoCiiu = ?", cuit)
            if cursor.fetchone():
                # Insertar en T_IVA_TRIGGER_P
                cursor.execute("""
                    INSERT INTO T_IVA_TRIGGER_P 
                        (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion))

                # Comprobación en TB_CLIENTE_CIERRE
                cursor.execute("SELECT 1 FROM TB_CLIENTE_CIERRE WHERE cuit = ?", cuit)
                if cursor.fetchone():
                    # Obtener max_id_in y n_periodo
                    cursor.execute("SELECT ISNULL(MAX(id), 0) + 1 FROM tb_lpa_iva")
                    max_id_in = cursor.fetchone()[0]

                    cursor.execute("""SELECT cuit, MAX(fecha_asignacion) AS fecha_asignacion
                                    FROM TB_CLIENTE_CIERRE 
                                    WHERE cuit = ? AND fecha_asignacion <= EOMONTH(DATEFROMPARTS(LEFT(?, 4), RIGHT(?, 2), '01'))
                                    GROUP BY cuit""",(cuit, periodo, periodo))
                    print(cursor.fetchone().fecha_asignacion)
                    fecha_cierre=date(cursor.fetchone().fecha_asignacion)

                    if fecha_iva_completa <= fecha_cierre:
                        n_periodo = ((fecha_cierre.year - fecha_iva_completa.year) * 12 + fecha_cierre.month - fecha_iva_completa.month) + 12
                    else:
                        n_periodo = (fecha_cierre.year - fecha_iva_completa.year) * 12 + fecha_cierre.month - fecha_iva_completa.month

                    # Lógica con la confianza
                    if confianza > 0.7:
                        cursor.execute("EXEC Update_TB_lpa_iva @num_periodo = ?, @estado = 'A'", n_periodo)
                        cursor.execute("EXEC Insert_TB_lpa_iva @num_periodo = ?, @id_entrada = ?, @estado = 'A'", n_periodo, max_id_in)
                        cursor.execute("EXEC Insert_TB_VENTA @num_periodo = ?, @id_venta = ?", n_periodo, max_id_in)
                        cursor.execute("EXEC Update_TB_VENTA @num_periodo = ?, @estado = 'A'", n_periodo)
                        estado_interno='A'
                        proceso='A'
                        observacion='Proceso exitoso'
                        cursor.execute("""
                            INSERT INTO TL_IVA
                                (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion, proceso)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado_interno,observacion , fecha_creacion, proceso))


                    else:
                        # Si la confianza es baja
                        cursor.execute("EXEC Update_TB_lpa_iva @num_periodo = ?, @estado = 'C'", n_periodo)
                        cursor.execute("EXEC Insert_TB_lpa_iva @num_periodo = ?, @id_entrada = ?, @estado = 'C'", n_periodo, max_id_in)
                        cursor.execute("EXEC Insert_TB_VENTA @num_periodo = ?, @id_venta = ?", n_periodo, max_id_in)
                        cursor.execute("EXEC Update_TB_VENTA @num_periodo = ?, @estado = 'C'", n_periodo)

                        cursor.execute("""
                            INSERT INTO TL_IVA
                                (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion, proceso)
                            SELECT id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, 'A', 'Proceso exitoso (Indice de confianza menor al esperado)', fecha_creacion, 'A' 
                            FROM inserted
                        """)

                else:
                    cursor.execute("EXEC Insert_T_IVA_ERROR")
                    cursor.execute("""
                        INSERT INTO TL_IVA
                            (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion, proceso)
                        SELECT id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, 'M', 'Es necesario insertar la fecha cierre del cliente', fecha_creacion, 'C' 
                        FROM inserted
                    """)
                    cursor.execute("EXEC Insert_TB_CLIENTE_CIERRE")
            
            else:
                cursor.execute("""
                    INSERT INTO TL_IVA
                        (id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, estado, observacion, fecha_creacion, proceso)
                    SELECT id_ejecucion, banco, cuit, nombre_documento, periodo, valor_IVA_leido, confianza, formato, 'M', 'El cliente no existente en la herramienta', fecha_creacion, 'X' 
                    FROM inserted
                """)
        
        # Confirmar transacción
        cursor.execute("COMMIT")
        
    except Exception as e:
        # En caso de error, revertir la transacción
        cursor.execute("ROLLBACK")
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()