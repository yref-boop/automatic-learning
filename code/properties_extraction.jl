using FileIO
using Images
#using ImageView (para enseñarme las fotos desde la terminal)

# cargar imagen
image = load("imagen.jpg");

# tamaño imagen
size(image)

# extraer canales de la imagen:
red.(image)
green.(image)
blue.(image)

# blanco y negro
Gray.(image)

"""
en lo referente a los colores, lo mas util dada nuestra base de datos, es extraer el color más abundante en la imagen (el color de la fruta)
basandonos en los ejemplos dados en clase, podemos crear una matriz booleana en base a la imagen dada que indique en que zonas de la imagen es mas prevalente que color, con una sensibilidad especificada
"""
minimal_difference = 0.3

red_channel = red.(image)
green_channel = green.(image)
blue_channel = blue.(image)

# red is more prevalent
red_matrix = (red_channel.>(green_channel.+minimal_difference)) .& (red_channel.>(blue_channel.+minimal_difference));

# green is more prevalent
green_matrix = (green_channel.>(red_channel.+minimal_difference)) .& (green_channel.>(blue_channel.+minimal_difference));

# blue is more prevalent
blue_matrix = (blue_channel.>(green_channel.+minimal_difference)) .& (blue_channel.>(red_channel.+minimal_difference));

# to look for yellow, we compare the difference between red & green & between both of them with blue
# this is much less sensitive than the other, thus the lower minimal difference
yellow_matrix = (green_channel.>(blue_channel.+0.1)) .& (red_channel.>(blue_channel.+0.1))


"""
las imagenes se trataran de tal forma que se escoja el color principal
"""


# get the most representative matrix:
 function most_common(red_matrix, green_matrix, blue_matrix, yellow_matrix)
    maximum_value = max(sum(red_matrix),sum(green_matrix),sum(blue_matrix),sum(yellow_matrix))
    if maximum_value == sum(red_matrix)
        red_matrix
    else
        if maximum_value == sum(green_matrix)
            green_matrix
        else
            if maximum_value == sum(blue_matrix)
                blue_matrix
            else
                yellow_matrix
            end
        end
    end
end

bolean_matrix = most_common(red_matrix, green_matrix, blue_matrix, yellow_matrix)

# muchisimo texto hacer lo de las simetrias, lo dejamos para la siguiente iteracion ok?
"""
dada la simpleza de los datos extraidos, al menos en esta iteracion, resulta de interes suponer un sistema inteligente extremadamenta básico, que proponga el tipo de fruta en base al color detectado en la figura de tal forma que:

red_matrix      -> apple
green_matrix    -> apple
yellow_matrix   -> banana

esta relacion obviamente no recoge sutilezas suficientes (existen platanos verdes y manzanas amarillas) pero dados los datos es posible que ssea util
"""

# loading the dataset
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            print(fileName);
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (images);
end;

function loadTrainingDataset()
    (positivesColor, positivesGray) = loadFolderImages("positivos");
    (negativesColor, negativesGray) = loadFolderImages("negativos");
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], [positivesGray; negativesGray], targets);
end;
loadTestDataset() = ((colorMatrix,_) = loadFolderImages("data/"); return colorMatrix; );

# aproximacion pocha artesanal:
function identify(image)
    minimal_difference = 0.3

    red_channel = red.(image)
    green_channel = green.(image)
    blue_channel = blue.(image)

    basic_red_matrix = (red_channel.>green_channel) .& (red_channel.>(blue_channel));

    basic_green_matrix = (green_channel.>red_channel) .& (green_channel.>(blue_channel));

    blue_matrix = (blue_channel.>green_channel) .& (blue_channel.>(red_channel));

    yellow_matrix = (green_channel.>blue_channel) .& (red_channel.>blue_channel) .& (red_channel.>green_channel);

    red_matrix = basic_red_matrix .& .!(yellow_matrix);
    green_matrix = basic_green_matrix .& .!(yellow_matrix);

    function most_common(red_matrix, green_matrix, blue_matrix, yellow_matrix)
        maximum_value = max(sum(red_matrix),sum(green_matrix),sum(blue_matrix),sum(yellow_matrix))
        if maximum_value == sum(red_matrix)
            "manzana roja"
        else
            if maximum_value == sum(green_matrix)
                "manzana verde"
            else
                if maximum_value == sum(blue_matrix)
                    "???"
                else
                    "banana amarilla"
                end
            end
        end
    end

    return most_common(red_matrix, green_matrix, blue_matrix, yellow_matrix)
end;

# use example:
manzanas = loadFolderImages("data/manzanas")
map(identify, manzanas)
