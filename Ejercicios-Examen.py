# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 08:59:07 2019

@author: dr
"""

import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests
from scipy import misc




print('****** EXAMEN JACINTO PALMA ******')

print('2) Crear un vector de ceros de tamaño 10')
vector_ceros_10= np.zeros(10)
vector_ceros_10

print('3) Crear un vector de ceros de tamaño 10 y el de la posicion 5 sea igual a 1')
vector_ceros_10_5_1= np.zeros(10)
vector_ceros_10_5_1[5] =1
vector_ceros_10_5_1

print('4) Cambiar el orden de un vector de 50 elementos, el de la posicion1 es el de la 50 etc.')
vector_50_cambiar_orden = np.arange(50,0,-1)
vector_50_cambiar_orden

print('5) Crear una matriz de 3x3 con valores del cero al 8')
matriz_0_8 = np.arange(0,9).reshape(3,3)
matriz_0_8

print('6) Encontrar los indices que no sean cero en un arreglo')
arreglo = [1,2,0,0,4,0]
arreglo_np = np.nonzero(arreglo)
print(arreglo_np[0])

print('7) Crear una matriz de identidad 3 x 3')
matriz_identidad_33 = np.identity(3)
matriz_identidad_33

print('8) Crear una matriz 3 x 3 x 3 con valores randomicos')
matriz_random_333 = np.random.rand(3,3,3)
matriz_random_333

print('9) Crear una matriz 10 x 10 y encontrar el mayor y el menor')
matriz_10_10 = np.random.rand(10,10)
matriz_10_10
print(f'Mayor valor: {matriz_10_10.max()}')
print(f'Menor valor: {matriz_10_10.min()}')

print('10) Sacar los colores RGB unicos en una imagen (cuales rgb existen ej: 0, 0, 0 - 255,255,255 -> 2 colores)')

camara = data.camera()
plt.imshow(camara)
plt.show(block=True)
plt.interactive(False)

print('11) ¿Como crear una serie de una lista, diccionario o arreglo?')
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

print('SERIE - LISTA')
serie_lista = pd.Series(mylist)
serie_lista
print('SERIE - DICCIONARIO')
serie_dic = pd.Series(mydict)
serie_dic
print('SERIE - ARREGLO')
serie_arr = pd.Series(myarr)
serie_arr

print('12) ¿Como convertir el indice de una serie en una columna de un DataFrame?')
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
ser = pd.Series(mydict) 

print('SERIE')
ser
# Transformar la serie en dataframe y hacer una columna indice
print('DATAFRAME - COLUMNA INDICE')
df = pd.DataFrame(ser.index)
df

print('13) ¿Como combinar varias series para hacer un DataFrame?')
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

combinar_series = pd.DataFrame([ser1,ser2])
combinar_series

print('14) ¿Como obtener los items que esten en una serie A y no en una serie B?')
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
###############################


print('15) ¿Como obtener los items que no son comunes en una serie A y serie B?')
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
##################################

print('16) ¿Como obtener el numero de veces que se repite un valor en una serie?')
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
ser_val_repetido = pd.Series.value_counts(ser)
ser_val_repetido

print('17) ¿Como mantener los 2 valores mas repetidos de una serie, y a los demas valores cambiarles por 0 ?')
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))


print('18) ¿Como transformar una serie de un arreglo de numpy a un DataFrame con un shape definido?')
ser = pd.Series(np.random.randint(1, 10, 35))
datos = pd.DataFrame()
datos.append(ser,ignore_index=True)

print('19) ¿Obtener los valores de una serie conociendo la posicion por indice?')
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]
# a e i o u
ser.iloc[pos]

print('20) ¿Como anadir series vertical u horizontalmente a un DataFrame?')
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))

# AÑADIR HORIZONTALMENTE
datos_horizontal = pd.DataFrame()
datos_horizontal.append(ser1,ignore_index=True)

#AÑADIR VERTICALMENTE
datos_vertical = pd.DataFrame({'index': ser1,'letter': ser2})
datos_vertical


print('21)¿Obtener la media de una serie agrupada por otra serie?')
#groupby tambien esta disponible en series.

frutas = pd.Series(np.random.choice(['manzana', 'banana', 'zanahoria'], 10))
pesos = pd.Series(np.linspace(1, 10, 10))
print(pesos.tolist())
print(frutas.tolist())
type(pesos.tolist())

df = pd.DataFrame({'FRUTAS':frutas.tolist(), 'PESOS':pesos.tolist()})
grupo =df.groupby('FRUTAS')
grupo['PESOS'].mean()
#> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#> ['banana', 'carrot', 'apple', 'carrot', 'carrot', 'apple', 'banana', 'carrot', 'apple', 'carrot']

# Los valores van a cambiar por ser random
# apple     6.0
# banana    4.0
# carrot    5.8
# dtype: float64



print('22)¿Como importar solo columnas especificas de un archivo csv?')
#https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv.
url="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
df = pd.DataFrame(c)
df = df[["age","tax"]]
print(df)