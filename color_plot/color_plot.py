import matplotlib.pyplot as plt

# Abrir el archivo
with open('C:\\Users\\rnara\\OneDrive\\Documents\\UDC\\Q6\\AA\\practica\\jl\\color_plot\\banana_plot.txt', 'r') as archivo:

    # Leer todas las líneas en una lista
    lineas = archivo.readlines()

    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Iterar sobre cada línea y agregar una barra con el valor RGB correspondiente
    for i, linea in enumerate(lineas):
        valores = [float(valor) for valor in linea.split(',')]
        print(valores)
        color = tuple(valores)
        ax.bar(i, 1, color=color)

    # Configurar el eje
    ax.set_xticks(range(len(lineas)))
    ax.set_xticklabels(range(1, len(lineas)+1))
    ax.set_xlim([-0.5, len(lineas)-0.5])
    ax.tick_params(axis='both', which='both', length=0)

    # Mostrar el gráfico de barras
    plt.show()
