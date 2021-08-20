#include <iostream>
#include <opencv2/opencv.hpp>

using num = float;

__global__ void
mandelbrot(unsigned char *count, const unsigned int image_size_x, const unsigned int image_size_y,
           const unsigned int max_iter,
           const num point_x, const num point_y, const num point_size) {
    const unsigned int x_idx = blockIdx.x;
    const unsigned int y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (!(x_idx < image_size_x && y_idx < image_size_y)) return;

    const unsigned int idx = x_idx * image_size_y + y_idx;


    const num c_x = ((num) x_idx / (num) image_size_y - 0.5f) * point_size + point_x;
    const num c_y = ((num) y_idx / (num) image_size_y - 0.5f) * point_size + point_y;
    num z_x = 0, z_y = 0;

    for (int i = 0; i < max_iter; i++) {
        const num prev_z_x = z_x;

        z_x = z_x * z_x - z_y * z_y + c_x;
        z_y = prev_z_x * z_y * 2 + c_y;

        if (z_x * z_x + z_y * z_y > 4) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0 * 256.0);
            return;
        }
    }

    count[idx] = 255;
}

void array_to_image(const unsigned char *count, cv::Mat &image,
                    const unsigned int image_size_x, const unsigned int image_size_y) {
    for (int i = 0; i < image_size_x; i++) {
        auto *src = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image_size_y; j++) {
            const unsigned char cnt = count[i * image_size_y + j];
            if (cnt == 255) {
                src[j][0] = src[j][1] = src[j][2] = 0;
            } else {
                src[j][0] = (255 - cnt) / 2;
                src[j][1] = cnt;
                src[j][2] = cnt;
            }
        }
    }
}

const unsigned int
        image_size_x = 1440,
        image_size_y = 2560;
num point_x = 0;
num point_y = 0;
num point_size = 4.0;
int mouse_prev_x = 0, mouse_prev_y = 0;
bool mouse_flag = false;

void change_point_size(int event, int x, int y, int flags, void *userdata) {
    if (event == cv::EVENT_MOUSEWHEEL) {
        if (cv::getMouseWheelDelta(flags) > 0) {
            point_size *= 0.9;
        }
        if (cv::getMouseWheelDelta(flags) < 0) {
            point_size /= 0.9;
        }
    }
    if (event == cv::EVENT_LBUTTONDOWN) {
        mouse_flag = true;
        mouse_prev_x = x;
        mouse_prev_y = y;
    }
    if (mouse_flag && event == cv::EVENT_MOUSEMOVE) {
        num pixel = point_size / image_size_y;
        point_y -= num(x - mouse_prev_x) * pixel;
        point_x -= num(y - mouse_prev_y) * pixel;
        mouse_prev_x = x;
        mouse_prev_y = y;
    }
    if (event == cv::EVENT_LBUTTONUP) {
        mouse_flag = false;
    }
}

int main() {

    const unsigned int max_iter = 65536;

    const unsigned int block_size = 256;

    dim3 grid(image_size_x, (image_size_y + block_size - 1) / block_size);
    dim3 block(block_size);

    unsigned int n_bytes = image_size_x * image_size_y * sizeof(unsigned char);

    unsigned char *count, *count_gpu;
    count = (unsigned char *) malloc(n_bytes);
    cudaMalloc((unsigned char **) &count_gpu, n_bytes);

    cv::Mat image = cv::Mat::zeros(image_size_x, image_size_y, CV_8UC3);

    cv::namedWindow("image");

    cv::setMouseCallback("image", change_point_size);

    while (true) {
        mandelbrot<<<grid, block>>>(count_gpu, image_size_x, image_size_y, max_iter,
                                    point_x, point_y, point_size);
        cudaMemcpy(count, count_gpu, n_bytes, cudaMemcpyDeviceToHost);
        array_to_image(count, image, image_size_x, image_size_y);

        cv::imshow("image", image);
        int key = cv::waitKey(10);

        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
