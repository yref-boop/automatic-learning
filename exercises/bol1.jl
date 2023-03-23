using DelimitedFiles

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes)
    if length(classes) == 2
        feature = feature .== classes[1]
        reshape(feature, (length(feature),1))
        return feature;
    else length(classes) > 2
        bool_matrix = falses(length(feature),length(classes))
        for i in eachindex(classes)
            bool_matrix[:,i] = (feature .== classes[i])
        end
    return bool_matrix;
    end
end;

dataset = readdlm("iris.data",',')
targets = dataset[:,5]
targets_classes = unique(targets);

inputs = Float32.(dataset[:,1:4]);

targets = oneHotEncoding(targets, targets_classes);

print("...\n")
targets