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
las imagenes se trataran de tal forma que primero se escoja el color principal y a partir de ahi se depurarán posibles artefactos, para poder tratar la imagen de la forma mas limpia posible, evitando posibles detecciones fuera de la fruta como tal
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

umbral_matrix = most_common(red_matrix, green_matrix, blue_matrix, yellow_matrix)

# recognize objects inside this umbral matrix
labelArray = ImageMorphology.label_components(umbral_matrix);

# extra data that can be extracted
boundingBoxes = ImageMorphology.component_boxes(labelArray);
sizes = ImageMorphology.component_lengths(labelArray);
pixels = ImageMorphology.component_indices(labelArray);
pixels = ImageMorphology.component_subscripts(labelArray);
centroids = ImageMorphology.component_centroids(labelArray);

# to erase noise-like small objects:
minimum_size = 30

sizes = component_lengths(labelArray)
clean_labels = findall(sizes .<= minimum_size) .- 1;
boolean_matrix = [!in(label,clean_labels) && (label!=0) for label in labelArray];

# in our specific case, maybe just storing the biggest element would be enough
labelArray = ImageMorphology.label_components(boolean_matrix)


"""
leyendo la literatura relacionada y tras analizar el problema, llegamos a la conclusion de la importancia que tienen las simetrias, tanto verticales como horizontales a la hora de distinguir entre estas dos frutas
resulta especialmente util para discernir ambas figuras facilmente
"""

# to get the centroid

