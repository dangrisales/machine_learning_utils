# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:53:11 2023

@author: Usach
"""

# import pandas as pd
# from scipy.stats import f_oneway
# from scipy.stats import chi2_contingency

# df = pd.read_csv("demograficas//demografia_pd.csv")
# df = df[df['GROUP'].notna()]
# df = df[df['Age'].notna()]
# df = df[df['Education'].notna()]
# df = df[df['Sex'].notna()]

# df = df[df["GROUP"] != "CTR"]
# # df_dcl = df[df["GROUP"] == "dcl"]
# # df_no_dcl = df[df["GROUP"] == "no_dcl"]
# # df_ctr = df[df["GROUP"] == "CTR"]


#     # df_emparejado = pd.concat([df_emparejado,row.to_frame().T], ignore_index = True)

# grupos = df['GROUP']
# age = df['Age']
# education = df['Education']
# sex = df['Sex']

# p_value_age = f_oneway(*[age[grupos == g] for g in grupos.unique()])[1]
# p_value_education = f_oneway(*[education[grupos == g] for g in grupos.unique()])[1]

# tabla_contingencia = pd.crosstab(df['GROUP'], df['Sex'])
# _, p_value_sex, _, _ = chi2_contingency(tabla_contingencia)



# iteracion = 0
# # df_emparejado = pd.DataFrame()
# while (p_value_age < 0.1) or (p_value_education < 0.1) or (p_value_sex < 0.1) :
#     # p_value_age = f_oneway(df["Age"].values)[1]
#     # p_value_education = f_oneway(df["Education"].values)[1]
#     # _, p_value_sex, _, _ = chi2_contingency(tabla_contingencia)
#     print(str(iteracion) + ")")

#     grupo_mayor = df['GROUP'].value_counts().idxmax()
#     #df_min = df[df["GROUP"] == valor_mas_frecuente]
    
    
#     puntaje_p = 0
#     puntaje_p_mayor = 0
#     participante_mayor = 0
#     # Crear un bucle for para iterar sobre cada participante del grupo mayor
#     for participante in df[df['GROUP'] == grupo_mayor].index:
#         # Eliminar el participante del DataFrame copia
#         df_copia = df.drop(participante)
        
        
#         grupos = df_copia['GROUP']
#         age = df_copia['Age']
#         education = df_copia['Education']
#         sex = df_copia['Sex']
#         p_value_age = f_oneway(*[age[grupos == g] for g in grupos.unique()])[1]
#         p_value_education = f_oneway(*[education[grupos == g] for g in grupos.unique()])[1]

#         tabla_contingencia = pd.crosstab(df_copia['GROUP'], df_copia['Sex'])
#         _, p_value_sex, _, _ = chi2_contingency(tabla_contingencia)
        
#         puntaje_p = (3*(p_value_age*p_value_education*p_value_sex))/(p_value_age+p_value_education+p_value_sex)
#         if puntaje_p > puntaje_p_mayor:
#             puntaje_p_mayor = puntaje_p
#             participante_mayor = participante
           
    
#     df = df.drop(participante_mayor)
#     iteracion += 1

#     grupos = df['GROUP']
#     age = df['Age']
#     education = df['Education']
#     sex = df['Sex']
#     p_value_age = f_oneway(*[age[grupos == g] for g in grupos.unique()])[1]
#     p_value_education = f_oneway(*[education[grupos == g] for g in grupos.unique()])[1]

#     tabla_contingencia = pd.crosstab(df['GROUP'], df['Sex'])
#     _, p_value_sex, _, _ = chi2_contingency(tabla_contingencia)
        
#     print("p-value age: " + str(p_value_age))
#     print("p-value education: " + str(p_value_education))
#     print("p-value sex: " + str(p_value_sex))

# df.to_csv("emparejamiento//emparejamiento_4_p_0-10.csv")
    
    
    
    
    
    
    
    
    
import pandas as pd
import ManejoDatabases as md
import numpy as np

df = pd.read_csv("demografia_pd_ctr.csv", sep=";")
columnas_a_reemplazar = ['OFF', 'ON']
df[columnas_a_reemplazar] = df[columnas_a_reemplazar].replace('no', np.nan)
df[columnas_a_reemplazar] = df[columnas_a_reemplazar].replace('ne', np.nan)
df = df.fillna(float('nan'))
df["OFF"] = df["OFF"].astype(float)
df["ON"] = df["ON"].astype(float)

column_group = 'GROUP'
variables = ["Age","Education","Sex","Years PD", "L DOPA", "OFF", "ON"]
tipos = ["continua","continua","categorica","continua","continua","continua","continua","continua"]
min_p_value = 0.1

df_emparejado, p_values = md.emparejamiento_estadistico_f1_parejo(df,column_group,variables,tipos,min_p_value)

str_p_value = str(min_p_value).replace(".","-")
df_emparejado.to_csv("Results/emparejamiento_prueba.csv")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    