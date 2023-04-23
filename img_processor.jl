using Images, ImageFiltering, StatsBase, ImageFeatures

fruta = "bananas"
n_subcarpetas = 3

function image_processing(carpeta::String, nombre_archivo::String)
    io = open(nombre_archivo, "w+")
    archivos = readdir(carpeta, join = true)
    for (i, archivo) in enumerate(archivos)
        println(i, "/", length(archivos)," - Processing image ", archivo)
        img = RGB.(load(archivo))
        img_gray = Gray.(img)
        img_CHWa = channelview(img)
        img_CHW = permutedims(img_CHWa, (1,3,2))
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
        stds = std(filtered_array, dims = 2)
        skews = zeros(size(filtered_array,1))

        for i in axes(filtered_array, 1)
            skews[i] = skewness(filtered_array[i, :])
        end

        laplace = [-1 -1 -1;-1 8 -1;-1 -1 -1]
        img_edges = imfilter(img_gray, laplace)
        mean_edges = mean(img_edges .> 0.1)
        std_edges = std(img_edges .> 0.1)

        img_glcm = sum(glcm(img_gray, 1, [0,45,90,135]))
        img_glcm_contrast = contrast(img_glcm)
        img_glcm_energy = energy(img_glcm)
        img_glcm_homogeneity = sum(img_glcm[i,j] / (1 + abs(i-j)) for i in axes(img_glcm, 1), j in axes(img_glcm,2))

        linea = join(means, ",")*","*join(stds, ",")*","*join(skews, ",")*","*"$mean_edges"*","*"$std_edges"*","*"$img_glcm_contrast"*","*"$img_glcm_energy"*","*"$img_glcm_homogeneity"*","*fruta
        println(io, linea)
        println(linea, "\n")
    end
    close(io)
end

for i in 3:n_subcarpetas
    println("Processing folder data/$fruta/subcarpeta_$i")
    dataset = image_processing("data/$fruta/subcarpeta_$i", "$fruta$i.data")
    println("============\ndataset $i obtained\n============")
end