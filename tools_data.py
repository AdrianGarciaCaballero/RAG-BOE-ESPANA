import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_VACATIONS = os.path.join(BASE_DIR, "data", "Tabla Dinámica de VACACIONES.csv")
FILE_SICK_LEAVE = os.path.join(BASE_DIR, "data", "Tabla de Bajas Médicas.csv")
FILE_EMPLOYEES = os.path.join(BASE_DIR, "data", "employees.csv")

def query_employee_data(employee_name: str, query_type: str) -> str:
    results = []
    found_any = False
    
    # 0. Consultar Ficha General (employees.csv)
    try:
        if os.path.exists(FILE_EMPLOYEES):
            df_emp = pd.read_csv(FILE_EMPLOYEES)
            # Normalizar a string para comparar
            df_emp['id'] = df_emp['id'].astype(str)
            df_emp['name'] = df_emp['name'].astype(str)
            
            # Buscar por nombre O por ID
            match_emp = df_emp[
                df_emp['name'].str.contains(employee_name, case=False, na=False) | 
                df_emp['id'].str.contains(employee_name, case=False, na=False)
            ]
            
            if not match_emp.empty:
                found_any = True
                row = match_emp.iloc[0]
                results.append(f"👤 FICHA EMPLEADO ({row['name']}):")
                results.append(f"   - ID: {row['id']}")
                results.append(f"   - Puesto: {row['role']}")
                results.append(f"   - Vacaciones Restantes: {row['vacation_days_left']} días")
                results.append(f"   - Última subida salarial: {row['last_pay_raise']}")
                results.append("") # Separador
    except Exception as e:
        results.append(f"⚠️ Error leyendo employees.csv: {e}")

    # 1. Consultar Vacaciones (Histórico)
    try:
        if os.path.exists(FILE_VACATIONS):
            df_vac = pd.read_csv(FILE_VACATIONS)
            df_vac['ID_Empleado'] = df_vac['ID_Empleado'].astype(str)
            
            match_vac = df_vac[
                df_vac['Nombre_Empleado'].str.contains(employee_name, case=False, na=False) |
                df_vac['ID_Empleado'].str.contains(employee_name, case=False, na=False)
            ]
            
            if not match_vac.empty:
                found_any = True
                if query_type in ["vacation", "general"]:
                    total_days = match_vac['Días_Solicitados'].sum()
                    records = len(match_vac)
                    dept = match_vac.iloc[0]['Departamento']
                    results.append(f"🏖️ HISTÓRICO VACACIONES ({dept}): {records} solicitudes. Total días: {total_days}.")
                    for _, row in match_vac.iterrows():
                        results.append(f"   - {row['Fecha_Inicio']} a {row['Fecha_Fin']}: {row['Días_Solicitados']} días ({row['Estado']})")
    except Exception as e:
        results.append(f"⚠️ Error leyendo archivo de vacaciones (CSV): {e}")

    # 2. Consultar Bajas
    try:
        if os.path.exists(FILE_SICK_LEAVE):
            df_sick = pd.read_csv(FILE_SICK_LEAVE)
            df_sick['ID_Empleado'] = df_sick['ID_Empleado'].astype(str)
            
            match_sick = df_sick[
                df_sick['Nombre_Empleado'].str.contains(employee_name, case=False, na=False) |
                df_sick['ID_Empleado'].str.contains(employee_name, case=False, na=False)
            ]
            
            if not match_sick.empty:
                found_any = True
                if query_type in ["sick_leave", "general"]:
                    total_days = match_sick['Dias_Totales'].sum()
                    records = len(match_sick)
                    results.append(f"🤒 HISTÓRICO BAJAS: {records} bajas. Total días: {total_days}.")
                    for _, row in match_sick.iterrows():
                        results.append(f"   - {row['Fecha_Inicio']} ({row['Tipo_Baja']}): {row['Dias_Totales']} días. Motivo: {row['Motivo_Detallado']}")
    except Exception as e:
        results.append(f"⚠️ Error leyendo archivo de bajas (CSV): {e}")

    if not found_any:
        return f"No encontré información para el empleado '{employee_name}' en los archivos Excel de RRHH."
    
    return "\n".join(results)

if __name__ == "__main__":
    # Test local
    print(query_employee_data("Ana", "general"))
