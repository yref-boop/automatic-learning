using Images, Clustering, Suppressor

function image_processing(carpeta::String)
    archivos = readdir(carpeta, join = true)
    matriz_inputs = Array{Float32}(undef, length(archivos), 3)
    for (i, archivo) in enumerate(archivos)
        println(i, "/", length(archivos)," - Processing image ", archivo)
        img = RGB.(load(archivo))
        img_CHWa = channelview(img)
        img_CHW = permutedims(img_CHWa, (1,3,2))
        testmat = reshape(img_CHW, (3, size(img)[1]*size(img)[2]))
        n_cols = size(testmat, 2)
        cols_to_remove = []
        for j in 1:n_cols
            if all(testmat[:, j] .> 0.8)
                push!(cols_to_remove, j)
            end
        end
        filtered_array = hcat([testmat[:, j] for j in 1:n_cols if !(j in cols_to_remove)]...)
        sol = kmeans(filtered_array, 1, maxiter=100)
        input = vec(sol.centers)
        matriz_inputs[i, :] = input
        println(matriz_inputs[i, :], "\n")
    end

    return matriz_inputs
end

function io_matrix(matriz::Array{Float32, 2}, nombre_archivo::AbstractString)
    io = open(nombre_archivo, "w+")
    for i in 1:size(matriz, 1)
        linea = join(string.(matriz[i, :]), ",")
        println(io, linea)
    end
    close(io)
end

for i in 6:6
    println("Processing folder data/bananas/subcarpeta_$i")
    dataset = image_processing("data/bananas/subcarpeta_$i")
    println("============\ndataset $i obtained\n============")
    io_matrix(dataset, "bananas$i.data")
end