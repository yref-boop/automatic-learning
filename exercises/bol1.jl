include("libraries.jl")

function one_hot_encoding(feature::AbstractArray{<:Any,1}, classes)
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

function std_norm_params(inputs::AbstractArray{<:Real,2})
    """
    Devuelve la media y desviación típica de cada atributo para un vector de dos dimensiones.
    """
    means = mean(inputs, dims=1)
    stds = std(inputs, dims=1)

    return means, stds
end

function std_norm(inputs::AbstractArray{<:Real,2})
    """
    Realiza la normalización mediante media y desviación típica.
    En el caso de que la desviación típica sea 0, el valor queda en 0.
    """
    means, stds = std_norm_params(inputs)
    normalized_inputs = (inputs .- means)./stds

    for i = 1:size(inputs,2)
        if stds[i] == 0
            normalized_inputs[:, i] .= 0
        end
    end

    return normalized_inputs
end;

function minmax_norm_params(inputs::AbstractArray{<:Real,2})
    """
    Devuelve el máximo y mínimo de cada atributo para un vector de dos dimensiones.
    """
    min = minimum(inputs, dims = 1)
    max = maximum(inputs, dims = 1)

    return min, max
end

function minmax_norm(inputs::AbstractArray{<:Real,2})
    """    
    Realiza la normalización usando el valor mínimo y el máximo.
    En el caso de que el mínimo sea igual al máximo, se añade un valor despreciable
    que evite la división entre 0.
    """
    min, max = minmax_norm_params(inputs)
    normalized_inputs = (inputs .- max)./(min .- max .+ 1e-8 * (min == max))

    return normalized_inputs
end;

dataset = readdlm("iris.data", ',')
targets = dataset[:, 5]
targets_classes = unique(targets)

inputs = Float32.(dataset[:, 1:4])
normalized_inputs = std_norm(inputs)
encoded_targets = one_hot_encoding(targets, targets_classes)
