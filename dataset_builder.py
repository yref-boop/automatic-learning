nombre_base = "manzanas"
archivo_salida = "manzanas.data"
cadena = ",manzana"
num_archivos = 11

with open(archivo_salida, "w+") as f_out:
    # Recorremos cada archivo de entrada
    for i in range(1, num_archivos + 1):
        nombre_archivo = nombre_base + str(i) + ".data"
        with open(nombre_archivo, "r") as f_in:
            # Recorremos cada línea del archivo de entrada
            for linea in f_in:
                # Añadimos la cadena al final de la línea y escribimos en el archivo de salida
                nueva_linea = linea.strip() + "\n"
                f_out.write(nueva_linea)

print("Se ha creado el archivo de salida", archivo_salida)
