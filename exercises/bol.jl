include("librerias.jl")

# Codificación

function one_hot_encoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    """
    Recibe el vector con las variables categóricas y las clases únicas que pueden darse.
    En el caso de tener dos categorías devuelve un vector de booleanos,
    y en el otro una matriz codificada.
    """
    if length(classes) == 2
        feature = feature .== classes[1]
        reshape(feature, (length(feature), 1))
        return feature
    else length(classes) > 2
        bool_matrix = falses(length(feature), length(classes))
        for i in eachindex(classes)
            bool_matrix[:,i] = (feature .== classes[i])
        end
    end

    return bool_matrix
end

"""
Función sobrecargada de one_hot_encoding, que extraiga directamente las categorías únicas de salida.
"""
one_hot_encoding(feature::AbstractArray{<:Any,1}) = one_hot_encoding(feature::AbstractArray{<:Any,1}, unique(feature))

"""
Función sobrecargada de one_hot_encoding, que convierte un vector con entradas binarias a una matriz 
de salida con una salida única.
"""
one_hot_encoding(feature::AbstractArray{Bool,1}) = reshape(feature, length(feature), 1)

# Normalización con media y desviación típica

function calculate_zeromean_normalization_parameters(inputs::AbstractArray{<:Real,2})
    """
    Devuelve la media y desviación típica de cada atributo para un vector de dos dimensiones.
    """
    means = mean(inputs, dims = 1)
    stds = std(inputs, dims = 1)

    return (means, stds)
end

function normalize_zeromean!(inputs::AbstractArray{<:Real,2}, meanstd::NTuple{2,AbstractArray{<:Real,2}})
    """
    Realiza la normalización mediante media y desviación típica.
    En el caso de que la desviación típica sea 0, el valor queda en 0.
    """
    normalized_inputs = (inputs .- meanstd[1])./meanstd[2]

    for i = 1:size(inputs,2)
        if meanstd[2][i] == 0
            normalized_inputs[:, i] .= 0
        end
    end

    return normalized_inputs
end

"""
Función sobrecargada de normalize_zeromean!, con un único parámetro que calcule los parámetros
de normalización automáticamente.
"""
normalize_zeromean!(inputs::AbstractArray{<:Real,2}) = normalize_zeromean!(inputs, calculate_zeromean_normalization_parameters(inputs))

function normalize_zeromean(inputs::AbstractArray{<:Real,2}, meanstd::NTuple{2, AbstractArray{<:Real,2}})
    """
    Misma función que normalize_zeromean! pero sin modificar la matriz de datos.
    """
    return normalize_zeromean!(copy(inputs), meanstd)
end;

"""
Misma función sobrecargada que normalize_zeromean! pero sin modificar la matriz de datos.
"""
normalize_zeromean(inputs::AbstractArray{<:Real,2}) = normalize_zeromean!(copy(inputs));

# Normalización con máximo y mínimo

function calculate_minmax_normalization_parameters(inputs::AbstractArray{<:Real,2})
    """
    Devuelve el máximo y mínimo de cada atributo para un vector de dos dimensiones.
    """
    min = minimum(inputs, dims = 1)
    max = maximum(inputs, dims = 1)

    return (min, max)
end

function normalize_minmax!(inputs::AbstractArray{<:Real,2}, minmax::NTuple{2,AbstractArray{<:Real,2}})
    """    
    Realiza la normalización usando el valor mínimo y el máximo.
    En el caso de que el mínimo sea igual al máximo, se añade un valor despreciable
    que evite la división entre 0.
    """
    normalized_inputs = (inputs .- minmax[2])./(minmax[1] .- minmax[2] .+ 1e-8 * (minmax[1] == minmax[2]))

    return normalized_inputs
end

"""
Función sobrecargada de normalize_minmax!, con un único parámetro que calcule los parámetros
de normalización automáticamente.
"""
normalize_minmax!(inputs::AbstractArray{<:Real,2}) = normalize_minmax!(inputs, calculate_minmax_normalization_parameters(inputs))

function normalize_minmax(inputs::AbstractArray{<:Real,2}, minmax::NTuple{2, AbstractArray{<:Real,2}})
    """
    Misma función que normalize_minmax! pero sin modificar la matriz de datos.
    """
    return normalize_minmax!(copy(inputs), minmax)
end;

"""
Misma función sobrecargada que normalize_minmax! pero sin modificar la matriz de datos.
"""
normalize_minmax(inputs::AbstractArray{<:Real,2}) = normalize_minmax!(copy(inputs));

# Clasificar salidas

function classify_outputs(outputs::AbstractArray{<:Real,2}, threshold = 0.5)
    """
    Recibe como entrada una matriz de salidas de un modelo y devuelve una matriz de booleanos
    del mismo tamaño, donde cada fila tiene un único valor true que indica la clase a la que se 
    clasifica ese patrón.
    """
    n_cols = size(outputs, 2)

    if n_cols == 1
        return outputs .>= threshold
    else
        (_, indices_max_each_instance) = findmax(outputs, dims = 2)
        outputs = falses(size(outputs))
        outputs[indices_max_each_instance] = outputs[indices_max_each_instance] .= true
    end

    return outputs
end

# Precisión

"""
Calcula el promedio de los vectores booleanos targets y outputs, resultando en la precisión.
"""
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1}) = mean(outputs .== targets)

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    """
    Función sobrecargada que usa matrices bidimensionales de valores booleanos.
    """
    n_cols = size(outputs, 2)

    if n_cols == 1
        return accuracy(targets[:,1], outputs[:,1])
    else
        class_comparison = targets .== outputs
        correct_classifications = all(class_comparison, dims = 2)
        return mean(correct_classifications)
    end
end

"""
Función sobrecargada donde las salidas no se han interpretado como valores booleanos.
"""
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, threshold = 0.5) = 
    accuracy((outputs .>= threshold), targets)

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    """
    Función sobrecarga donde las salidas reales no se han interpretado todavía como valores
    de pertenencia a N clases.
    """
    if size(outputs, 2) == 1
        return accuracy(targets[:,1], outputs[:,1])
    else
        return accuracy(targets, classify_outputs(outputs))
    end
end

# RNA

function ann(topology::AbstractArray{<:Int,1}, n_inputs, n_outputs)
    ann = Chain();
    n_inputs_layer = n_inputs

    for n_outputs_layer = topology
        ann = Chain(ann..., Dense(n_inputs_layer, n_outputs_layer, σ) );
        n_inputs_layer = n_outputs_layer;
    end
    
    n_outputs <= 2 ? 
        ann = Chain(ann...,  Dense(n_inputs_layer, 1, σ)) : 
        ann = Chain(ann...,  Dense(n_inputs_layer, n_outputs, identity), softmax)
    return ann
end;

function ann_train(
    topology::AbstractArray{<:Int, 1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    max_epochs::Int = 1000, min_loss::Real = 0, learning_rate::Real = 0.01
)
    """
    Crea y entrena una RNA para realizar clasificación.
    """
    inputs, targets = dataset
    rna = ann(topology, size(inputs)[2], size(targets)[2])
    loss(x, y) = (size(targets)[2] == 1) ? Flux.Losses.binarycrossentropy(rna(x),y) : Flux.Losses.crossentropy(rna(x),y)
    optimizer = ADAM(learning_rate)
    losses = []

    for epoch in 1:max_epochs
        print("Iteración ", epoch, "\n")
        Flux.train!(loss, Flux.params(rna), [(inputs', targets')], optimizer)
        push!(losses, loss(inputs', targets'))
        if losses[end] <= min_loss
            break
        end
    end

    return rna, losses
end

ann_train(
    topology::AbstractArray{<:Int, 1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    max_epochs::Int = 1000, min_loss::Real = 0, learning_rate::Real = 0.01
) = ann_train(topology, (dataset[1], reshape(dataset[2], :, 1)), max_epochs, min_loss, learning_rate)
    
# Dataset

dataset = readdlm("iris.data", ',')
targets = dataset[:, 5]
inputs = Float32.(dataset[:, 1:4])

topology = [2];

normalized_inputs = normalize_zeromean!(inputs, calculate_minmax_normalization_parameters(inputs))
encoded_targets = one_hot_encoding(targets)

rna, losses = ann_train(topology, (normalized_inputs, encoded_targets))

print(rna, "\n\n", losses)
