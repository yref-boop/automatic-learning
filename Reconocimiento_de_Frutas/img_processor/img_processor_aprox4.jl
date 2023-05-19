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

        bool_matrix = img_edges .> 0.15
        bounding_box = component_boxes(bool_matrix*1)[2:end]
        x1, y1 = bounding_box[1][1]
        x2, y2 = bounding_box[1][2]
        img_crop = img[x1 : x2 , y1 : y2]
        img_crop_CHW = permutedims(channelview(img_crop), (1,3,2))
        img_CHW_crop_flip_hor = permutedims(channelview(rotr90(img_crop')), (1,3,2))
        img_CHW_crop_flip_ver = permutedims(channelview(rotr90(rotr90(rotr90(img_crop')))), (1,3,2))
        sym_hor = img_crop_CHW - img_CHW_crop_flip_hor
        mean_sym_hor = mean(sym_hor)
        std_sym_hor = std(sym_hor)

        sym_ver = img_crop_CHW - img_CHW_crop_flip_ver
        mean_sym_ver = mean(sym_ver)
        std_sym_ver = std(sym_ver)

        linea = "$mean_sym_hor"*","*"$std_sym_hor"*","*"$mean_sym_ver"*","*"$std_sym_ver"
        println(io, linea)
        println(linea, "\n")
    end
    close(io)
end

println("Processing folder data/manzanas/")
image_processing("data/manzanas/", "manzanas-sym.data")
println("--~~~====::> dataset manzanas-sym obtained <::====~~~--")

println("Processing folder data/bananas/")
image_processing("data/bananas/", "bananas-sym.data")
println("--~~~====::> dataset bananas-sym obtained <::====~~~--")

println("Processing folder data/naranjas/")
image_processing("data/naranjas/", "naranjas-sym.data")
println("--~~~====::> dataset naranjas-sym obtained <::====~~~--")

# Los resultados de este dataset se añaden a los demás por medio de el script de python "dataset-builder-v2.py" para tener más flexibilidad a la hora de añadir o eliminar características.