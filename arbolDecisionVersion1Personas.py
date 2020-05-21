#!/usr/bin/env python

"""
Fuentes: 
- https://ernestocrespo13.wordpress.com/2017/11/19/arbol-de-decision-hecho-en-python/

- https://www.youtube.com/watch?v=T5pRlIbr6gg

- https://github.com/llSourcell/gender_classification_challenge/blob/master/demo.py
"""

try:
     from sklearn import tree
except ImportError:
	import pip
	pip.main(['install', 'sklearn'])
finally:
	from sklearn import tree

#Se crea la instancia del arbol de decision.
arbol = tree.DecisionTreeClassifier()

# Datos de entrada: Personas con [altura, peso, talla_de_zapato]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]


# Variable objetivo: Predecir si es hombre o mujer.
Y = ['hombre', 'hombre', 'mujer', 'mujer', 'hombre', 'hombre', 'mujer', 'mujer',
     'mujer', 'hombre', 'hombre']

# Se le pasan los datos X y Y (predictores y objetivo)
arbol = arbol.fit(X, Y)

#Se definen dos datos nuevos

persona1 = [190, 70, 43]
persona2 = [185, 62, 37]
persona3 = [190, 72, 40]

prediction = arbol.predict([persona1])

#Se muestra el resultado de la prediccion de dato1
print("El primer dato se predijo como " , str(prediction))

prediction = arbol.predict([persona2])
print("El segundo dato se predijo como " , str(prediction))

prediction = arbol.predict([persona3])
print("El tercer dato se predijo como " , str(prediction))