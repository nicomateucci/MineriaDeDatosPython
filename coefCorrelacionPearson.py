# Fuente: https://medium.com/@hdezfloresmiguelangel/el-coeficiente-de-correlaci%C3%B3n-de-pearson-con-ejemplo-en-python-6e8588f67e35

import matplotlib.pyplot as plt

try:
	import pandas
except ImportError:
	import pip
	pip.main(['install', 'pandas'])
finally:
	import pandas as pd

df = pd.read_csv("Advertising.csv")

#Archivo con cantidad de dinero invertido en publicidad en TV, Radio o Diarios con la cantidad de ventas que trajo dicha inversion.

df.head() # Mostrar un poco de informacion del archivo.

del df['Unnamed: 0'] # Borrar columna "Unnamed: 0"

df.head()
df.describe() # Mostrar estadisticas de los datos

df.corr(method="pearson")
# La columna "Sales" indica la correlacion es:
#	Ventas
# TV	0.78
# Radio 0.58
#Diario 0.22

# Grafico de ventas contra publicidad en TV
plt.plot(df["TV"], df["Sales"], "ro")
plt.title("Coeficiente de Correlacion de Pearson es 0.78")
plt.ylabel("Ventas")
plt.xlabel("TV")
plt.show()

# Grafico de ventas contra publicidad en diarios
plt.plot(df["TV"], df["Newspaper"], "ro")
plt.title("Coeficiente de Correlacion de Pearson es 0.58")
plt.ylabel("Ventas")
plt.xlabel("Diarios")
plt.show()


# Grafico de ventas contra publicidad en TV
plt.plot(df["TV"], df["Radio"], "ro")
plt.title("Coeficiente de Correlacion de Pearson es 0.22")
plt.ylabel("Ventas")
plt.xlabel("Radio")
plt.show()



