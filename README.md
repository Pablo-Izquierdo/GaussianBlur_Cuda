# GaussianBlur_Cuda

## compile.sh
This file have the code to run the gaussianBlur program.

## stb_image.h and  stb_imagen_write.h
Are package with the funtions to read and write a imagen.

## facultad.png
this png is the imagen that I use as example

## ref.png
Is an example of the correct result that the program have to return us.

## gaussianBlur.cu

This file contain the main program. It have three implementation:

-CPU implementation, which is the original impleentation of the algorithm.

-GPU Basic implementation, Is a working implementation of the gaussianBlur that split the operations in the three color channels.

-GPU implementation, Is a attempt to optimize the algorithm using the cuda nodes cache. But it does work properly and need to be debug.

