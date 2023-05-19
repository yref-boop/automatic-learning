using Images, ImageFiltering, StatsBase, ImageFeatures, ImageTransformations, FileIO

function image_processing(carpeta::String, nombre_archivo::String)
    io = open(nombre_archivo, "w+")
    archivos = readdir(carpeta, join = true)
    for (i, archivo) in enumerate(archivos)
        println(i, "/", length(archivos)," - Processing image ", archivo)
        img = RGB.(load(archivo))
        img_CHW = permutedims(channelview(img), (1,3,2))
        testmat = reshape(img_CHW, (3, size(img)[1]*size(img)[2]))
        n_cols = size(testmat, 2)
        cols_to_remove = []

        for j in 1:n_cols
            if all(testmat[:, j] .> 0.7) || all(testmat[:, j] .< 0.5)
                push!(cols_to_remove, j)
            end
        end
        filtered_array = hcat([testmat[:, j] for j in 1:n_cols if !(j in cols_to_remove)]...)
        
        means = mean(filtered_array, dims = 2)

        linea = "$means"
        println(io, linea)
        println(linea, "\n")
    end
    close(io)
end

println("Processing folder data/manzanas/")
image_processing("data/manzanas/", "manzanas-rgb.data")
println("--~~~====::> dataset manzanas-rgb obtained <::====~~~--")

println("Processing folder data/bananas/")
image_processing("data/bananas/", "bananas-rgb.data")
println("--~~~====::> dataset bananas-rgb obtained <::====~~~--")

# Los resultados de este dataset se añaden a los demás por medio de un script de python "dataset-builder-v2.py" para tener más flexibilidad a la hora de añadir o eliminar características.