"""
Fuente: 
- https://www.iartificial.net/arboles-de-decision-con-ejemplos-en-python/#Datos_para_regresion
"""

try:
     from sklearn import tree
except ImportError:
	import pip
	pip.main(['install', 'sklearn'])
finally:
    # datos de iris, estan dentro de la libreria sklearn
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.tree import export_text 

iris = load_iris()

# print(iris.DESCR) # informacion sobre del conjunto de datos iris

# Lo mas relevante es:
#    :Number of Instances: 150 (50 in each of three classes)
#    :Number of Attributes: 4 numeric, predictive attributes and the class
#    :Attribute Information:
#        - sepal length in cm
#        - sepal width in cm
#        - petal length in cm
#        - petal width in cm
#        - class:
#                - Iris-Setosa
#                - Iris-Versicolour
#                - Iris-Virginica

print("Primeras 20 filas del conjunto")
print(iris.data[0:10,:])

# Clase 0: Iris-Setosa, 
#       1: Iris-Versicolor
#       2: Iris-Virginica
print("Primeras 10 variables objetivo")
print(iris.target[0:150])

# Crear arbol
arbol = DecisionTreeClassifier(max_depth=5, random_state=100) 
     # Cantidad de niveles: 5
arbol.fit(iris.data, iris.target) # entrenamiento del arbol

# Obtener predicciones
print("PREDICCIONES ----------------------------")
print( arbol.predict(iris.data[100:140]) )
# tree.plot_tree(arbol) 

r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)


# Si queremos saber las probabilidades podemos usar el metodo predict_proba
print( tree.predict_proba(iris.data[47:53]) )

# la primera clase (Setosa) es la primera columna, la segunda clase en la segunda, etc.
# este es el resultado:
# [[1.         0.         0.        ]
#  [1.         0.         0.        ]
#  [1.         0.         0.        ]
#  [0.         0.90740741 0.09259259]
#  [0.         0.90740741 0.09259259]
#  [0.         0.90740741 0.09259259]]
