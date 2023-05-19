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
        img_crop_gray = img_gray[x1 : x2 , y1 : y2]

        img_glcm_0 = glcm(img_crop_gray, 1, 0)
        img_glcm_contrast_0 = contrast(img_glcm_0)
        img_glcm_energy_0 = energy(img_glcm_0)
        img_glcm_homogeneity_0 = sum(img_glcm_0[i,j] / (1 + abs(i-j)) for i in axes(img_glcm_0, 1), j in axes(img_glcm_0,2))

        img_glcm_45 = glcm(img_crop_gray, 1, 45)
        img_glcm_contrast_45 = contrast(img_glcm_45)
        img_glcm_energy_45 = energy(img_glcm_45)
        img_glcm_homogeneity_45 = sum(img_glcm_45[i,j] / (1 + abs(i-j)) for i in axes(img_glcm_45, 1), j in axes(img_glcm_45,2))

        img_glcm_90 = glcm(img_crop_gray, 1, 90)
        img_glcm_contrast_90 = contrast(img_glcm_90)
        img_glcm_energy_90 = energy(img_glcm_90)
        img_glcm_homogeneity_90 = sum(img_glcm_90[i,j] / (1 + abs(i-j)) for i in axes(img_glcm_90, 1), j in axes(img_glcm_90,2))

        linea = "$img_glcm_contrast_0"*","*"$img_glcm_energy_0"*","*"$img_glcm_homogeneity_0"*","*"$img_glcm_contrast_45"*","*"$img_glcm_energy_45"*","*"$img_glcm_homogeneity_45"*","*"$img_glcm_contrast_90"*","*"$img_glcm_energy_90"*","*"$img_glcm_homogeneity_90"
        println(io, linea)
        println(linea, "\n")
    end
    close(io)
end

println("Processing folder data/manzanas/")
image_processing("data/manzanas/", "manzanas-glcm.data")
println("--~~~====::> dataset manzanas-glcm obtained <::====~~~--")

println("Processing folder data/bananas/")
image_processing("data/bananas/", "bananas-glcm.data")
println("--~~~====::> dataset bananas-glcm obtained <::====~~~--")

println("Processing folder data/naranjas/")
image_processing("data/naranjas/", "naranjas-glcm.data")
println("--~~~====::> dataset naranjas-glcm obtained <::====~~~--")

# Los resultados de este dataset se añaden a los demás por medio de el script de python "dataset-builder-v2.py" para tener más flexibilidad a la hora de añadir o eliminar características.