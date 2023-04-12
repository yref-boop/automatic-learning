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
normalize_zeromean!(inputs::AbstractArray{<:Real,2}) = 
    normalize_zeromean!(inputs, calculate_zeromean_normalization_parameters(inputs))

function normalize_zeromean(inputs::AbstractArray{<:Real,2}, meanstd::NTuple{2, AbstractArray{<:Real,2}})
    """
    Misma función que normalize_zeromean! pero sin modificar la matriz de datos.
    """
    return normalize_zeromean!(copy(inputs), meanstd)
end

"""
Misma función sobrecargada que normalize_zeromean! pero sin modificar la matriz de datos.
"""
normalize_zeromean(inputs::AbstractArray{<:Real,2}) = normalize_zeromean!(copy(inputs))

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
normalize_minmax!(inputs::AbstractArray{<:Real,2}) = 
    normalize_minmax!(inputs, calculate_minmax_normalization_parameters(inputs))

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

function classify_outputs(outputs::AbstractArray{<:Real,2}; threshold = 0.5)
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
accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}; threshold = 0.5) = 
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

function build_class_ann(topology::AbstractArray{<:Int,1}, n_inputs::Int, n_outputs)
    ann = Chain()
    n_inputs_layer = n_inputs

    for n_outputs_layer = topology
        ann = Chain(ann..., Dense(n_inputs_layer, n_outputs_layer, σ) )
        n_inputs_layer = n_outputs_layer
    end
    
    n_outputs <= 2 ? 
        ann = Chain(ann...,  Dense(n_inputs_layer, 1, σ)) : 
        ann = Chain(ann...,  Dense(n_inputs_layer, n_outputs, identity), softmax)
    return ann
end

function train_class_ann(
    topology::AbstractArray{<:Int, 1}, train_set::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    val_set::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}} = tuple(zeros(0, 0), falses(0, 0)),
    test_set::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}} = tuple(zeros(0, 0), falses(0, 0)),
    max_epochs::Int = 1000, max_epochs_val::Int = 20, min_loss::Real = 0, learning_rate::Real = 0.01
)
    """
    Crea y entrena una RNA de clasificación con una arquitectura especificada utilizando un conjunto de datos de entrenamiento.
    Utiliza el algoritmo de optimización ADAM para minimizar la función de pérdida de la RNA en cada iteración.
    También puede utilizar conjuntos de datos de validación y de prueba para evaluar la precisión de la RNA durante el entrenamiento.

    Si el rendimiento de la RNA en el conjunto de validación no mejora durante max_epochs_val iteraciones consecutivas, el entrenamiento 
    se detiene para evitar el sobreajuste. También se puede detener si la pérdida en el conjunto de entrenamiento cae por debajo del valor 
    mínimo especificado.

    La función devuelve la RNA entrenada y los valores de pérdida en cada conjunto de datos en forma de vectores 
    (losses_train, losses_val, y losses_test).
    """
    inputs, targets = train_set
    rna = build_class_ann(topology, size(inputs)[2], size(targets)[2])
    loss(x, y) = (size(targets)[2] == 1) ? Flux.Losses.binarycrossentropy(rna(x), y) : Flux.Losses.crossentropy(rna(x), y)
    optimizer = ADAM(learning_rate)
    losses_train, losses_val, losses_test = [], [], []
    best_loss_val, no_improve, best_epoch = Inf, 0, 0
    best_rna = deepcopy(rna)

    for epoch in 1:max_epochs
        print("Iteración ", epoch, "\n")
        Flux.train!(loss, Flux.params(rna), [(inputs', targets')], optimizer)
        push!(losses_train, loss(inputs', targets'))

        if !isempty(val_set[1])
            loss_val = loss(val_set[1]', val_set[2]')
            push!(losses_val, loss_val)

            if loss_val < best_loss_val
                best_loss_val = loss_val
                best_rna = deepcopy(rna)
                best_epoch = epoch
                no_improve = 0
            else
                no_improve += 1
            end

            if no_improve >= max_epochs_val
                break
            end
        end

        if !isempty(test_set[1])
            push!(losses_test, loss(test_set[1]', test_set[2]'))
        end

        if losses_train[end] <= min_loss
            break
        end
    end

    if !isempty(val_set[1])
        rna = best_rna
    end

    if isempty(losses_val)
        return rna, losses_train, losses_test
    else
        print("best rna achieved at epoch ", best_epoch, "\n")
        return rna, losses_train, losses_val, losses_test
    end
end

function train_class_ann(
    topology::AbstractArray{<:Int, 1}, train_set::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    val_set::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} = tuple(zeros(0,0), falses(0)),
    test_set::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} = tuple(zeros(0,0), falses(0)),
    max_epochs::Int = 1000, max_epochs_val = 20, min_loss::Real = 0, learning_rate::Real = 0.01
)
    if !isempty(val_set[1])
        val_set = tuple(val_set[1], reshape(val_set[2], (length(val_set[2]), 1)))
    end 

    if !isempty(test_set)
        test_set = tuple(test_set[1], reshape(test_set[2], (length(test_set[2]), 1)))
    end

    return ann_train(topology, (train_set[1], reshape(train_set[2], :, 1)), 
        val_set, test_set, max_epochs, max_epochs_val, min_loss, learning_rate)
end

# Hold-out

function hold_out(n_samples::Int, test_ratio::Float64)
    """
    Separa un conjunto de datos en dos subconjuntos, uno para entrenamiento y otro para prueba,
    utilizando una proporción dada y generando índices aleatorios para seleccionar las muestras 
    para cada subconjunto. 
    Devuelve los índices de los patrones de entramiento y los de test, en ese orden.
    """
    index = randperm(n_samples)
    n_test = round(Int, n_samples*test_ratio)

    return (index[n_test + 1:end], index[1:n_test])
end

function hold_out(n_samples::Int, val_ratio::Float64, test_ratio::Float64)
    """
    Basándose en la función hold_out anterior, devuelve una tupla con tres vectores que contienen
    los índices de los patrones para los conjuntos de entrenamiento, validación y test.
    """
    index = randperm(n_samples)
    n_val = round(Int, n_samples*val_ratio)
    n_test = round(Int, n_samples*test_ratio)

    return (index[n_val + n_test + 1:end], index[n_test + 1:n_val + n_test], index[1:n_test])
end

# Plot

function plot_losses(losses_train, losses_val, losses_test)
    plot(
        x_label="Iteraciones",
        y_label="Pérdida",
        x_lims=(0, length(losses_train)),
        y_lims=(minimum([losses_train; losses_val; losses_test]), maximum([losses_train; losses_val; losses_test])),
        legend=:topleft,
        title="Evolución de los valores de loss",
        size=(800, 500),
    )
    plot!(losses_train, label="Entrenamiento")
    plot!(losses_val, label="Validación")
    plot!(losses_test, label="Prueba")
end


# Dataset

begin
    dataset = readdlm("iris.data", ',')
    targets = dataset[:, 5]
    encoded_targets = one_hot_encoding(targets)
    inputs = Float32.(dataset[:, 1:4])

    topology = [2];

    index_train, index_val, index_test = hold_out(size(inputs)[1], 0.2, 0.1)
    inputs_train, targets_train = inputs[index_train, :], encoded_targets[index_train, :]
    inputs_val, targets_val = inputs[index_val, :], encoded_targets[index_val, :]
    inputs_test, targets_test = inputs[index_test, :], encoded_targets[index_test, :]

    normalization_parameters = calculate_zeromean_normalization_parameters(inputs_train)
    normalized_inputs_train = normalize_zeromean!(inputs_train, normalization_parameters)
    normalized_inputs_val = normalize_zeromean!(inputs_val, normalization_parameters)
    normalized_inputs_test = normalize_zeromean!(inputs_test, normalization_parameters)

    rna, losses_train, losses_val, losses_test =  train_class_ann(topology, (inputs_train, targets_train), val_set = (inputs_val, targets_val), test_set = (inputs_test, targets_test), max_epochs = 10000, max_epochs_val = 20, min_loss = 0, learning_rate = 0.01)

    plot_losses(losses_train, losses_val, losses_test)
end