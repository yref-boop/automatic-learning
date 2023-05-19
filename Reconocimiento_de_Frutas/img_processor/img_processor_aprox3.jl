using Images, ImageFiltering, StatsBase, ImageFeatures, ImageTransformations, FileIO

function image_processing(carpeta::String, nombre_archivo::String)
    io = open(nombre_archivo, "w+")
    archivos = readdir(carpeta, join = true)
    for (i, archivo) in enumerate(archivos)
        println(i, "/", length(archivos)," - Processing image ", archivo)
        img = RGB.(load(archivo))
        img_gray = Gray.(img)

        laplace = [-1 -1 -1;-1 8 -1;-1 -1 -1]
        img_edges = imfilter(img_gray, laplace)

        mean_edges = mean(img_edges .> 0)
        std_edges = std(img_edgebananas)

        linea = "$mean_edges"*","*"$std_edges"
        println(io, linea)
        println(linea, "\n")
    end
    close(io)
end

println("Processing folder data/manzanas/")
image_processing("data/manzanas/", "manzanas-edge.data")
println("--~~~====::> dataset manzanas-edge obtained <::====~~~--")

println("Processing folder data/bananas/")
image_processing("data/bananas/", "bananas-edge.data")
println("--~~~====::> dataset bananas-edge obtained <::====~~~--")

println("Processing folder data/naranjas/")
image_processing("data/naranjas/", "naranjas-edge.data")
println("--~~~====::> dataset naranjas-edge obtained <::====~~~--")

# Los resultados de este dataset se añaden a los demás por medio de el script de python "dataset-builder-v2.py" para tener más flexibilidad a la hora de añadir o eliminar características.