#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>

struct Mat{
    int mrows;
    int mcols;
    unsigned char* m;
};

typedef struct Mat Matrix;

void transferData (Matrix* dev, Matrix* src, int dataTypeSize, cudaMemcpyKind k)
void printMat (Matrix* Mat);
int newMat(Matrix* Mat, int rows, int cols);
int createDevMat(Matrix* dev, int rows, int cols);
void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float * filter, int  filterWidth);
void clearDevMat(Matrix* Mat);
__global__ void ComputeConvolutionOnGPU(uchar4* const blurredChannel, Matrix* inputChannel, float * filter, int filterWidth);
__global__ void ComputeConvolutionOnGPUBasic(Matrix* blurredChannel, Matrix* dev_channel, float * filter, int filterWidth)
double second();

#define SIZE 8;

float * createFilter(int width)
{
        const float sigma       = 2.f;                          // Standard deviation of the Gaussian distribution.

        const int       half    = width / 2;
        float           sum             = 0.f;


        // Create convolution matrix
        float * res=(float *)malloc(width*width*sizeof(float));


        // Calculate filter sum first
        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        // e (natural logarithm base) to the power x, where x is what's in the brackets
                        float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
                        int idx = (r + half) * width + c + half;

                        res[idx] = weight;
                        sum += weight;
                }
        }

        // Normalize weight: sum of weights must equal 1
        float normal = 1.f / sum;

        for (int r = -half; r <= half; ++r)
        {
                for (int c = -half; c <= half; ++c)
                {
                        int idx = (r + half) * width + c + half;

                        res[idx] *= normal;
                }
        }
        return res;
}


// Copmute gaussian blur per channel on the CPU.
// Call this function for each of red, green, and blue channels
// Returns blurred channel.
void ComputeConvolutionOnCPU(unsigned char* const blurredChannel, const unsigned char* const inputChannel, int rows, int cols, float * filter, int filterWidth)
{
        // Filter width should be odd as we are calculating average blur for a pixel plus some offset in all directions

        const int half   = filterWidth / 2;
        const int width  = cols - 1;
        const int height = rows - 1;

        // Compute blur
        for (int r = 0; r < rows; ++r)
        {
                for (int c = 0; c < cols; ++c)
                {
                        float blur = 0.f;

                        // Average pixel color summing up adjacent pixels.
                        for (int i = -half; i <= half; ++i)
                        {
                                for (int j = -half; j <= half; ++j)
                                {
                                        // Clamp filter to the image border
                                        int h = min(max(r + i, 0), height);
                                        int w = min(max(c + j, 0), width);

                                        // Blur is a product of current pixel value and weight of that pixel.
                                        // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
                                        int             idx             = w + cols * h;                                                                                 // current pixel index
                                        float   pixel   = static_cast<float>(inputChannel[idx]);

                                        idx                             = (i + half) * filterWidth + j + half;
                                        float   weight  = filter[idx];

                                        blur += pixel * weight;
                                }
                        }

                        blurredChannel[c + cols * r] = static_cast<unsigned char>(blur);
                }
        }
}

void GaussianBlurOnCPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float * filter, int  filterWidth)
{
        const int numPixels = rows * cols;

        // Create channel variables
        unsigned char* red                      = new unsigned char[numPixels];
        unsigned char* green            = new unsigned char[numPixels];
        unsigned char* blue                     = new unsigned char[numPixels];

        unsigned char* redBlurred       = new unsigned char[numPixels];
        unsigned char* greenBlurred = new unsigned char[numPixels];
        unsigned char* blueBlurred      = new unsigned char[numPixels];

        // Separate RGBAimage into red, green, and blue components
        for (int p = 0; p < numPixels; ++p)
        {
                uchar4 pixel = rgba[p];

                red[p]   = pixel.x;
                green[p] = pixel.y;
                blue [p] = pixel.z;
        }

        // Compute convolution for each individual channel
        ComputeConvolutionOnCPU(redBlurred, red, rows, cols, filter, filterWidth);
        ComputeConvolutionOnCPU(greenBlurred, green, rows, cols, filter, filterWidth);
        ComputeConvolutionOnCPU(blueBlurred, blue, rows, cols, filter, filterWidth);

        // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
        for (int p = 0; p < numPixels; ++p)
        {
                unsigned char r = redBlurred[p];
                unsigned char g = greenBlurred[p];
                unsigned char b = blueBlurred[p];

                modifiedImage[p] = make_uchar4(r, g, b, 255);
        }

        delete[] red;
        delete[] green;
        delete[] blue;
        delete[] redBlurred;
        delete[] greenBlurred;
        delete[] blueBlurred;
}



// Main entry into the application
int main(int argc, char** argv)
{
	char * imagePath;
	char * outputPath;
	
	int height, width, bpp, channels=4;
	uchar4 * originalImage, * blurredImage;

	int filterWidth=9;
	float * filter=createFilter(filterWidth);


	if (argc > 2)
	{
		imagePath = argv[1];
		outputPath = argv[2];
	}
	else
	{
		printf("Please provide input and output image files as arguments to this application.");
		exit(1);
	}



	//Read the image
	uint8_t* rgb_image = stbi_load(imagePath, &width, &height, &bpp, channels);
	
	if(rgb_image==NULL) printf("Could not load image file: %s\n",imagePath);
	
	//Allocate and copy
	originalImage=(uchar4 *)malloc(width*height*sizeof(uchar4));
	blurredImage=(uchar4 *)malloc(width*height*sizeof(uchar4));
        blurredImageGPUbasic=(uchar4 *)malloc(width*height*sizeof(uchar4));
        blurredImageGPU=(uchar4 *)malloc(width*height*sizeof(uchar4));
	printf("Width:%d, Height:%d Size(in Bytes):%lu\n", width, height, width*height*bpp*channels);
	for(int i=0;i<width*height*channels;i++)
	{
		int mod=i%channels;
		switch(mod)
		{
			case 0:
				originalImage[i/channels].x=rgb_image[i];
				break;
			case 1:
				originalImage[i/channels].y=rgb_image[i];
				break;
			case 2:
				originalImage[i/channels].z=rgb_image[i];
				break;
			case 3:
				originalImage[i/channels].w=rgb_image[i];
				break;
		}
	}

	//Tu práctica empieza aquí
	//CUDA	
        double Tcpu0,Tcpu1,Tcpu,Tgpu,Tgpu0,Tgpu1,Tgpubase,Tgpubase0,Tgpubase1;

	//Version CPU (Comentar cuando se trabaje con la GPU!)
        Tcpu0 = second(); //get initial time
	GaussianBlurOnCPU(blurredImage, originalImage, height, width, filter, filterWidth);
        Tcpu1 = second(); //get initial time
        Tcpu = Tcpu1 - Tcpu0;

        //Version GPU Basica (Comentar cuando se trabaje con la GPU!)
        Tgpubase0 = second(); //get initial time
	GaussianBlurOnGPUBasic(blurredImageGPUbasic, originalImage, height, width, filter, filterWidth); //TODO falta por hacer
        Tgpubase1 = second(); //get initial time
        Tgpubase = Tgpubase1 - Tgpubase0;

        //Version GPU with cache (Comentar cuando se trabaje con la GPU!)
        Tgpu0 = second(); //get initial time
	GaussianBlurOnGPU(blurredImageGPU, originalImage, height, width, filter, filterWidth);
        Tgpu1 = second(); //get initial time
        Tgpu = Tgpu1 - Tgpu0;


	for(int i=0;i<width*height;i++)
	{
		rgb_image[i*channels]=blurredImage[i].x;
		rgb_image[(i*channels)+1]=blurredImage[i].y;
		rgb_image[(i*channels)+2]=blurredImage[i].z;
		rgb_image[(i*channels)+3]=blurredImage[i].w;
	}	
	stbi_write_jpg(outputPath, width, height, 4, blurredImage, 100);

	printf("Done!\n");
	return 0;
}

void GaussianBlurOnGPU(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float * filter, int  filterWidth)
{
        const int numPixels = rows * cols;

        //Create Host Matrix
        Matrix red, green, blue;
        newMat(&red, rows, cols);
        newMat(&green, rows, cols);
        newMat(&blue, rows, cols);
        // Separate RGBAimage into red, green, and blue components
        for (int p = 0; p < numPixels; ++p) //Inicializar matrices
        {
                uchar4 pixel = rgba[p];

                red[p]   = pixel.x;
                green[p] = pixel.y;
                blue[p] = pixel.z;
        }

        //Create Device Matrix
        Matrix dev_red, dev_green, dev_blue;
        createDevMat(&dev_red);
        createDevMat(&dev_green);
        createDevMat(&dev_blue);
        uchar4* dev_bluredImage;
        cudaMalloc(&(dev_bluredImage), numPixels * sizeof(uchar4));
        float * dev_filter;
        cudaMalloc(&(dev_filter), filterWidth * filterWidth * sizeof(float));

        transferData(&dev_red, &red, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_green, &green, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_blue, &blue, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_filter, &filter, sizeof(float), cudaMemcpyHostToDevice);

        //Cuda parameters
        dim3 blocks(cols/SIZE,rows/SIZE);
        dim3 threadPerBlock(SIZE,SIZE);
        int filterSize = (filterWidth*filterWidth*sizeof(float));
        int blurredMatrixSize = 3*SIZE*SIZE*sizeof(unsigned char);
        int colorMatrixSize = 3*((SIZE+filterWidth/2)*2)*sizeof(unsigned char);
        int sharedSize = filterSize + blurredMatrixSize + colorMatrixSize;
        //Compute TODO
        ComputeConvolutionOnGPU <<<blocks, threadPerBlock, sharedSize>>> (dev_bluredImage, 
                                                        dev_red, dev_green, dev_blue, dev_filter, filterWidth);

        //Return Data
        transferData(&modifiedImage, &dev_bluredImage, sizeof(uchar4), cudaMemcpyDeviceToHost);

        //free
        clearMat(&red);
        clearMat(&green);
        clearMat(&blue);
        clearDevMat(&dev_red);
        clearDevMat(&dev_green);
        clearDevMat(&dev_blue);
        clearDevMat(&dev_filter);
}

__global__ void ComputeConvolutionOnGPU(uchar4* const blurredImage, Matrix* dev_red, Matrix* dev_green, Matrix* dev_blue, float * filter, int filterWidth)
{
        //identificar los threads
        int cols= (blockIdx.x * blockDim.x) + threadIdx.x;
        int rows= (blockIdx.y * blockDim.y) + threadIdx.y;

        int filterx=filterWidth / 2; //coordenadas centro del filtro
        int filtery=filterWidth / 2;
        const unsigned char* Scolor;
        const unsigned char* color;
        const unsigned char* blurredColor;
        extern __shared__ unsigned char blurred[];
        unsigned char *ScolorRed = blurred;
        unsigned char *ScolorGreen = &ScolorRed[(blockDim.x+filterx)*(blockDim.y+filtery)]; //12X12
        unsigned char *ScolorBlue = &ScolorGreen[(blockDim.x+filterx)*(blockDim.y+filtery)];
        unsigned char *SblurRed = &ScolorBlue[(blockDim.x+filterx)*(blockDim.y+filtery)];
        unsigned char *SblurGreen = &SblurRed[blockDim.x*blockDim.y]; //8X8
        unsigned char *SblurBlue = &SblurGreen[blockDim.x*blockDim.y];
        float * sharedFilter = (float*)&SblurBlue[blockDim.x*blockDim.y]; //Necesario Rellenar
        //Para bloques incompletos
        if (rows < dev_red.mrows && cols < dev_red.mcols){
                switch(threadIdx.z){ // Dependiendo de Z calcula un color diferente 
                        case 0 :
                                color = dev_red.m;
                                Scolor = ScolorRed;
                                blurredColor = SblurRed;
                        break;
                        case 1 : 
                                color = dev_green.m;
                                Scolor = ScolorGreen;
                                blurredColor = SblurGreen;
                        break;
                        case 2 :
                                color = dev_blue.m;
                                Scolor = ScolorBlue;
                                blurredColor = SblurBlue;
                        break;
                }

                if(threadIdx.z == 0){ //guardar el filtro 9x9 linealmente, las 3 dimensiones comparten el mismo filtro
                        //He indexado los threads para rellenar el filtro
                        for(int i=0; i<(filterWidth*filterWidth); i+=(blockDim.x*blockDim.y)){ 
                                sharedFilter[i] = filter[i];                
                        }
                }

                //Guardar color 12x12 linealmente for 0..63 
                for(int i=0; i<((blockDim.x+filterx)*(blockDim.y+filtery)); i+=(blockDim.x*blockDim.y)){  
                                int fila = rows-filterx;
                                int columna =cols-filtery;
                                //si sobrepasa limites de la imagen ponemos 0
                                if((fila < 0 || fila >= dev_red.mrows) || (columna < 0 || columna >= dev_red.mcols)){
                                        Scolor[i] = 0; //relleno con 0 los pixels exteriores
                                }else{
                                        Scolor[i] = color[fila*blockDim.x+columna+i];   
                                }             
                }

                __syncthreads(); //Que todos acaben;

                float blur = 0.f;
                for(int i=0; i<filterWidth; i++){
                        for(int j=0; j<filterWidth; j++){
                                int fila = threadIdx.x+i;
                                int columna =threadIdx.y+j;
                                float   pixel   = static_cast<float>(Scolor[fila * blockDim.x + columna]);
                                float   weight  = sharedFilter[i*filterWidth +j];
                                blur += pixel*weight;
                        }
                }

                blurredColor[threadIdx.x * blockDim.x + threadIdx.y] = static_cast<unsigned char>(blur); //Guardamos el pixel en la matrix correspondiente

                __syncthreads(); //Que todos acaben;

                if(threadIdx.z == 0){
                        unsigned char r = blurRed[threadIdx.x * blockDim.x + threadIdx.y];
                        unsigned char g = blurGreen[threadIdx.x * blockDim.x + threadIdx.y];
                        unsigned char b = blurBlue[threadIdx.x * blockDim.x + threadIdx.y];
                        blurredImage[rows * blockDim.x + cols] = make_uchar4(r, g, b, 255);
                }

        }
}


void clearMat(Matrix* Mat)
{
    if (Mat->m) {
        free(Mat->m);
    }
    Mat->m = NULL;
    Mat->mrows = 0;
    Mat->mcols = 0;
    
}

void clearDevMat(Matrix* Mat)
{
    if (Mat->m) {
        cudafree(Mat->m);
    }
    Mat->m = NULL;
    Mat->mrows = 0;
    Mat->mcols = 0;
    
}

void printMat (Matrix* Mat)
{
    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            printf("%c \n", Mat->m[i * SIZE + j]);
        }
    }

}

int newMat(Matrix* Mat, int rows, int cols)
{
    Mat->mrows = SIZE;
    Mat->mcols = SIZE;
    Mat->m = NULL;
    Mat->m = (unsigned char*)
        malloc(Mat->mrows * Mat->mcols * sizeof(unsigned char));

    return(Mat->m != NULL) ? 1 : 0;
}

int createDevMat(Matrix* dev, int rows, int cols)
{
    dev->mrows = rows;
    dev->mcols = cols;
    dev->m = NULL;

    HANDLE_ERROR(cudaMalloc(&(dev->m), dev->mrows * dev->mcols * sizeof(unsigned char)));

    return(dev->m != NULL) ? 1 : 0;
}

void transferData (Matrix* dev, Matrix* src, int dataTypeSize, cudaMemcpyKind k)
{
    size_t transfer_size = dev->mrows * dev->mcols * dataTypeSize;
    
    switch(k){
    case cudaMemcpyHostToDevice:
        cudaMemcpy(dev->m, src->m, transfer_size, cudaMemcpyHostToDevice);
        break;
    case cudaMemcpyDeviceToHost:
        cudaMemcpy(src->m, dev->m, transfer_size, cudaMemcpyDeviceToHost);
        break;
    }

}

/** second
    * This fuction return the time on the instant that it is call
    * */
double second()
{

    struct timeval tm;
    double t;

    static int base_sec = 0, base_usec = 0;

    gettimeofday(&tm, NULL);

    if (base_sec == 0 && base_usec == 0)
    {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    }
    else {
        t = (double)(tm.tv_sec - base_sec) +
            ((double)(tm.tv_usec - base_usec)) / 1.0e6;
    }

    return t;
}

void GaussianBlurOnGPUbasic(uchar4* const modifiedImage, const uchar4* const rgba, int rows, int cols, float * filter, int  filterWidth)
{
        const int numPixels = rows * cols;

        //Create Host Matrix
        Matrix red, green, blue;
        newMat(&red, rows, cols);
        newMat(&green, rows, cols);
        newMat(&blue, rows, cols);
        newMat(&redBlurred, rows, cols);
        newMat(&greenBlurred, rows, cols);
        newMat(&blueBlurred, rows, cols);
        // Separate RGBAimage into red, green, and blue components
        for (int p = 0; p < numPixels; ++p) //Inicializar matrices
        {
                uchar4 pixel = rgba[p];

                red[p]   = pixel.x;
                green[p] = pixel.y;
                blue[p] = pixel.z;
        }

        //Create Device Matrix
        Matrix dev_red, dev_green, dev_blue, dev_bluredImage;
        createDevMat(&dev_red);
        createDevMat(&dev_green);
        createDevMat(&dev_blue);
        createDevMat(&dev_blurredChannel);
        float * dev_filter;
        cudaMalloc(&(dev_filter), filterWidth * filterWidth * sizeof(float));

        transferData(&dev_red, &red, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_green, &green, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_blue, &blue, sizeof(unsigned char), cudaMemcpyHostToDevice);
        transferData(&dev_filter, &filter, sizeof(float), cudaMemcpyHostToDevice);

        //Cuda parameters
        dim3 blocks(cols/SIZE,rows/SIZE);
        dim3 threadPerBlock(SIZE,SIZE);

        //Compute TODO
        ComputeConvolutionOnGPU <<<blocks, threadPerBlock>>> (dev_blurredChannel, 
                                                        dev_red, dev_filter, filterWidth);
        //Return Data
        transferData(&redBlurred, &dev_blurredChannel, sizeof(unsigned char),cudaMemcpyDeviceToHost);
                                                        
        ComputeConvolutionOnGPU <<<blocks, threadPerBlock>>> (dev_blurredChannel, 
                                                        dev_green, dev_filter, filterWidth);
        //Return Data
        transferData(&greenBlurred, &dev_blurredChannel, sizeof(unsigned char),cudaMemcpyDeviceToHost);

        ComputeConvolutionOnGPU <<<blocks, threadPerBlock>>> (dev_blurredChannel, 
                                                        dev_blue, dev_filter, filterWidth);

        //Return Data
        transferData(&blueBlurred, &dev_blurredChannel, sizeof(unsigned char),cudaMemcpyDeviceToHost);

                // Recombine channels back into an RGBAimage setting alpha to 255, or fully opaque
        for (int p = 0; p < numPixels; ++p)
        {
                unsigned char r = redBlurred[p];
                unsigned char g = greenBlurred[p];
                unsigned char b = blueBlurred[p];

                modifiedImage[p] = make_uchar4(r, g, b, 255);
        }

        //free
        clearMat(&red);
        clearMat(&green);
        clearMat(&blue);
        clearDevMat(&dev_red);
        clearDevMat(&dev_green);
        clearDevMat(&dev_blue);
        clearDevMat(&dev_filter);
}

__global__ void ComputeConvolutionOnGPUBasic(Matrix* blurredChannel, Matrix* dev_channel, float * filter, int filterWidth)
{
        //identificar los threads
        int cols= (blockIdx.x * blockDim.x) + threadIdx.x;
        int rows= (blockIdx.y * blockDim.y) + threadIdx.y;

        int filterx=filterWidth / 2; //coordenadas centro del filtro
        int filtery=filterWidth / 2;
        const unsigned char* color = dev_channel.m;

        float blur = 0.f;
        for(int i=0; i<filterWidth; i++){
                for(int j=0; j<filterWidth; j++){
                        int fila = rows-filterx+i;
                        int columna =cols-filtery+j;
                        if((fila < 0 || fila >= dev_channel.mrows) || (columna < 0 || columna >= dev_channel.mcols)){
                                float   pixel   = static_cast<float>(color[fila * blockDim.x + columna]);
                                float   weight  = filter[i*blockDim.x+j];
                                blur += pixel*weight;
                        }
                }
        }

        blurredChannel.m[rows * blockDim.x + cols] = static_cast<unsigned char>(blur); //Guardamos el pixel en la matrix correspondiente
}