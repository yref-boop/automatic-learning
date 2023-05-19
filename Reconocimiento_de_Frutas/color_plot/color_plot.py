import matplotlib.pyplot as plt

# Abrir el archivo
with open('C:\\Users\\rnara\\OneDrive\\Documents\\UDC\\Q6\\AA\\practica\\jl\\color_plot\\naranja_plot.txt', 'r') as archivo:

    # Leer todas las líneas en una lista
    lineas = archivo.readlines()

    # Crear una figura y un eje
    fig, ax = plt.subplots()
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Iterar sobre cada línea y agregar una barra con el valor RGB correspondiente
    for i, linea in enumerate(lineas):
        valores = [float(valor) for valor in linea.split(',')]
        print(valores)
        color = tuple(valores)
        ax.bar(i, 1, color=color)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Mostrar el gráfico de barras
    plt.show()
