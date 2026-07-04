#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Box blur kernel (3x3 average)
__global__ void box_blur_kernel(unsigned char *d_out, const unsigned char *d_in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;
        int count = 0;

        // Loop over the 3x3 window around the pixel
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                // Boundary check (ensure we don't read outside image borders)
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += d_in[ny * width + nx];
                    count++;
                }
            }
        }
        
        // Write the average value
        d_out[y * width + x] = (unsigned char)(sum / count);
    }
}

int main() {
    const char *input_path = "test_pattern.jpg";
    const char *output_path = "blurred_pattern.jpg";

    int width = 0, height = 0, channels = 0;

    // 1. Load image as Grayscale (1 channel)
    unsigned char *h_in = stbi_load(input_path, &width, &height, &channels, 1);
    if (!h_in) {
        std::cerr << "Error: Could not load input image: " << input_path << "\n";
        return -1;
    }

    std::cout << "Loaded image: " << input_path << " (" << width << "x" << height << ", original channels: " << channels << ")\n";

    size_t img_size = width * height * sizeof(unsigned char);

    // Allocate memory for the output on CPU
    unsigned char *h_out = (unsigned char *)malloc(img_size);

    // 2. Allocate memory on the GPU (Device)
    unsigned char *d_in = nullptr;
    unsigned char *d_out = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_in, img_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc d_in failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    cudaMalloc(&d_out, img_size);

    // 3. Copy image data from CPU to GPU
    cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);

    // 4. Configure thread block and grid dimensions
    // We use a 2D block of 16x16 threads (256 threads total)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Launching kernel with grid size (" << numBlocks.x << ", " << numBlocks.y 
              << ") and block size (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")\n";

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // 5. Launch the blur kernel
    box_blur_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_in, width, height);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Kernel execution time: " << duration.count() << " ms\n";

    // 6. Copy results back from GPU to CPU
    cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);

    // 7. Save the output image
    if (stbi_write_jpg(output_path, width, height, 1, h_out, 90)) {
        std::cout << "Successfully saved blurred image to: " << output_path << "\n";
    } else {
        std::cerr << "Error: Could not save output image: " << output_path << "\n";
    }

    // 8. Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);
    stbi_image_free(h_in);

    return 0;
}
