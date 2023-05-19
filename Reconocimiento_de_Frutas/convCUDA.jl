using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using Images
using ImageTransformations
using CUDA
using Plots
using Measures
using Random
using Random:seed!

trainImgs = cu(Array{Float32}(undef, 50, 50, 3, 0));
trainLabels = Array{Int}(undef,0);
testImgs = cu(Array{Float32}(undef, 50, 50, 3, 0));
testLabels = Array{Int}(undef,0);
testRatio = 10;
labels = 0:2;

seed!(1)

function loadImages(testRatio)
    auxTrainImgs = cu(Array{Float32}(undef, 50, 50, 3, 0));
    auxTestImgs = cu(Array{Float32}(undef, 50, 50, 3, 0));
    bananaDir = "data/bananas/";
    manzanaDir = "data/manzanas/";
    naranjaDir = "data/naranjas/";
    n = 1;

    for file in readdir(bananaDir) 
        image = permutedims(channelview(RGB.(imresize(load("$bananaDir$file"), (50, 50)))), [2, 3, 1]) |> gpu;
        if (n % testRatio == 0) 
            auxTestImgs = cat(auxTestImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(testLabels, 0);
        else
            auxTrainImgs = cat(auxTrainImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(trainLabels, 0);
        end
        n += 1;
    end

    for file in readdir(manzanaDir) 
        image = permutedims(channelview(RGB.(imresize(load("$manzanaDir$file"), (50, 50)))), [2, 3, 1]) |> gpu;
        if (n % testRatio == 0) 
            auxTestImgs = cat(auxTestImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(testLabels, 1);
        else
            auxTrainImgs = cat(auxTrainImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(trainLabels, 1);
        end
        n += 1;
    end

    for file in readdir(naranjaDir) 
        image = permutedims(channelview(RGB.(imresize(load("$naranjaDir$file"), (50, 50)))), [2, 3, 1]) |> gpu;
        if (n % testRatio == 0) 
            auxTestImgs = cat(auxTestImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(testLabels, 2);
        else
            auxTrainImgs = cat(auxTrainImgs, convert.(Float32, image), dims = 4) |> gpu;
            push!(trainLabels, 2);
        end
        n += 1;
    end

    return auxTrainImgs, auxTestImgs;
end

trainImgs, testImgs = loadImages(testRatio);
println("Tamaño de la matriz de entrenamiento: ", size(trainImgs));
println("Valores minimo y maximo de las entradas: (", minimum(trainImgs), ", ", maximum(trainImgs), ")");

batch_size = 30;
gruposIndicesBatch = Iterators.partition(1:size(trainImgs,4), batch_size);
println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");

trainSet = [ (trainImgs[:,:,:,indicesBatch], onehotbatch(trainLabels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch] |> gpu;
testSet = (testImgs, onehotbatch(testLabels, labels)) |> gpu;

trainImgs = nothing;
testImgs = nothing;
GC.gc();

funcionTransferenciaCapasConvolucionales = relu;

ann = Chain(
    Conv((1, 1), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((1, 1), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((1, 1), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(2048, 3),
    softmax
) |> gpu

numBatchCoger = 1; numImagenEnEseBatch = [12, 6];
entradaCapa = trainSet[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
numCapas = length(Flux.params(ann));
println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
    println("   Capa ", numCapa, ": ", ann[numCapa]);
    global entradaCapa
    capa = ann[numCapa];
    salidaCapa = capa(entradaCapa);
    println("      La salida de esta capa tiene dimension ", size(salidaCapa));
    entradaCapa = salidaCapa;
end

ann(trainSet[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);

loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));

println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(trainSet)), " %");

opt = ADAM(0.001);

println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;
precisionTestPlot = [];

while (!criterioFin)
    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin, precisionTestPlot;

    Flux.train!(loss, Flux.params(ann), trainSet, opt);

    numCiclo += 1;

    precisionEntrenamiento = mean(accuracy.(trainSet));
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    if (precisionEntrenamiento >= mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        precisionTest = accuracy(testSet);
        push!(precisionTestPlot, precisionTest);
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(ann);
        numCicloUltimaMejora = numCiclo;
    end

    if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
        opt.eta /= 10.0
        println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
        numCicloUltimaMejora = numCiclo;
    end

    # Criterios de parada:
    if (precisionEntrenamiento >= 0.999)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true;
    end

    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end
end

function plot_acc(precisionTestPlot, filename::AbstractString="plot_dl.png")
    plot(
        x_lims=(0, length(precisionTestPlot)),
        y_lims=(0, 100),
        title="Evolución de los valores de precisión en el conjunto de test",
        xlabel="Ciclo",
        ylabel="Precisión (%)",
        legend= false,
        size=(1000, 700),
        margin = 15mm
    )
    plot!(precisionTestPlot.*100)
    
    savefig(filename)
end

plot_acc(precisionTestPlot);
println("Se ha impreso la gráfica de precisión del conjunto de test");