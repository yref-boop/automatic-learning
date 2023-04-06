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










