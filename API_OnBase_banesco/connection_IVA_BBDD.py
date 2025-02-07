import pyodbc

class SQLConnector:
    def __init__(self, server, database, user, password):
        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        print(pyodbc.drivers()) 
        try:
            connection_string = (
                f'DRIVER={{ODBC Driver 18 for SQL Server}}; SERVER={self.server}; '
                f'DATABASE={self.database}; UID={self.user}; PWD={self.password};'
		f'Encrypt=yes; TrustServerCertificate=yes;'
            )
            self.connection = pyodbc.connect(connection_string)
            print("Conexión exitosa")
        except Exception as e:
            print(f"Error al realizar la conexión: {str(e)}")

    def insert_data(self, query):
        if self.connection is None:
            print("No se ha establecido una conexión.")

        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            cursor.commit()
            print("Datos insertados correctamente")
        except Exception as e:
            print(f"Error al insertar datos: {str(e)}")
        finally:
            cursor.close()

    def read_data(self, query):
        if self.connection is None:
            print("No se ha establecido una conexión.")
            
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error al ejecutar la consulta: {str(e)}")
            return []
        finally:
            cursor.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()