#include <opencv2/opencv.hpp>

using num = float;
constexpr unsigned int
        IMAGE_SIZE_X = 1440,
        IMAGE_SIZE_Y = 2560;
constexpr unsigned int MAX_ITER = 65536;
constexpr unsigned int BLOCK_SIZE = 256;


__global__ void
mandelbrot(unsigned char *count,
           const num point_x, const num point_y, const num point_size) {
    const unsigned int x_idx = blockIdx.x;
    const unsigned int y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;

    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    const num c_x = ((num) x_idx / (num) IMAGE_SIZE_Y - 0.5f) * point_size + point_x;
    const num c_y = ((num) y_idx / (num) IMAGE_SIZE_Y - 0.5f) * point_size + point_y;
    num z_x = 0, z_y = 0;

    for (int i = 0; i < MAX_ITER; i++) {
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

void array_to_image(const unsigned char *count, cv::Mat &image) {
    for (int i = 0; i < IMAGE_SIZE_X; i++) {
        auto *src = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < IMAGE_SIZE_Y; j++) {
            const unsigned char cnt = count[i * IMAGE_SIZE_Y + j];
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

num point_x = 0;
num point_y = 0;
num point_size = 4.0;
int mouse_prev_x = 0, mouse_prev_y = 0;
bool mouse_flag = false;

void mouse_callback(int event, int x, int y, int flags, void *userdata) {
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
        num pixel = point_size / IMAGE_SIZE_Y;
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
    dim3 grid(IMAGE_SIZE_X, (IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    unsigned int n_bytes = IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(unsigned char);

    unsigned char *count, *count_gpu;
    count = (unsigned char *) malloc(n_bytes);
    cudaMalloc((unsigned char **) &count_gpu, n_bytes);

    cv::Mat image = cv::Mat::zeros(IMAGE_SIZE_X, IMAGE_SIZE_Y, CV_8UC3);

    cv::namedWindow("image");

    cv::setMouseCallback("image", mouse_callback);

    while (true) {
        mandelbrot<<<grid, block>>>(count_gpu, point_x, point_y, point_size);
        cudaMemcpy(count, count_gpu, n_bytes, cudaMemcpyDeviceToHost);
        array_to_image(count, image);

        cv::imshow("image", image);
        int key = cv::waitKey(10);

        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
