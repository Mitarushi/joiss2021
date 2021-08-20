#include <iostream>
#include <opencv2/opencv.hpp>

using num = float;

__global__ void
mandelbrot(unsigned char *count, const unsigned int image_size, const unsigned int max_iter,
           const num point_x, const num point_y, const num point_size) {
    const unsigned int x_idx = blockIdx.x;
    const unsigned int y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (!(x_idx < image_size && y_idx < image_size)) return;

    const unsigned int idx = x_idx * image_size + y_idx;


    const num c_x = (num) x_idx / (num) image_size * point_size + point_x;
    const num c_y = (num) y_idx / (num) image_size * point_size + point_y;
    num z_x = 0, z_y = 0;

    for (int i = 0; i < max_iter; i++) {
        const num prev_z_x = z_x;

        z_x = z_x * z_x - z_y * z_y + c_x;
        z_y = prev_z_x * z_y * 2 + c_y;

        if (z_x * z_x + z_y * z_y > 4) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 18.0 * 256.0);
            return;
        }
    }

    count[idx] = 255;
}

int main() {
    const unsigned int image_size = 51200;
    const unsigned int max_iter = 262144;
    const num point_x = -2.0;
    const num point_y = -2.0;
    const num point_size = 4.0;
    const unsigned int block_size = 256;

    dim3 grid(image_size, (image_size + block_size - 1) / block_size);
    dim3 block(block_size);

    unsigned int n_bytes = image_size * image_size * sizeof(unsigned char);

    unsigned char *count, *count_gpu;
    count = (unsigned char *) malloc(n_bytes);
    cudaMalloc((unsigned char **) &count_gpu, n_bytes);

    mandelbrot<<<grid, block>>>(count_gpu, image_size, max_iter,
                                point_x, point_y, point_size);

    cudaMemcpy(count, count_gpu, n_bytes, cudaMemcpyDeviceToHost);
    printf("gpu done\n");


    cv::Mat image = cv::Mat::zeros(image_size, image_size, CV_8UC3);
    for (int i = 0; i < image_size; i++) {
        auto *src = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image_size; j++) {
            const unsigned char cnt = count[i * image_size + j];
            if (cnt != 255) {
                src[j][0] = (255 - cnt) / 2;
                src[j][1] = cnt;
                src[j][2] = cnt;
            }
        }
    }

    cv::imwrite("./mandel.png", image);
    return 0;
}
