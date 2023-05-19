fruta = "naranjas"
filename = fruta + ".data"
files = ["./datasets/" + fruta + "/glcm/" + fruta + "-glcm.data",
         "./datasets/" + fruta + "/edge/" + fruta + "-edge.data",
         "./datasets/" + fruta + "/rgb/" + fruta + "-rgb.data",
         "./datasets/" + fruta + "/sym/" + fruta + "-sym.data"]

lines = []

for file in files:
    with open(file, 'r+') as file:
        f_lines = file.readlines()
        for i in range(len(f_lines)):
            linea = f_lines[i].rstrip('\n')
            if len(lines) <= i:
                lines.append(linea)
            else:
                lines[i] += ',' + linea

with open(filename, 'w+') as f_out:
    for line in lines:
        f_out.write(line + "," + fruta + "\n")

# Los datasets resultantes de cada fruta se juntan posteriormente en el dataset "fruits.data", que se usa después como entrada para las aproximaciones.