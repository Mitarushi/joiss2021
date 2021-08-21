#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <cstdio>
#include <cassert>

using num = float;
constexpr unsigned int
        IMAGE_SIZE_X = 1440,
        IMAGE_SIZE_Y = 2560;
constexpr unsigned int MAX_ITER = 65536;
constexpr unsigned int BLOCK_SIZE = 512;
constexpr unsigned int IMAGE_GRID_SIZE = 65;

constexpr unsigned int
        GRID_IMAGE_SIZE_X = (IMAGE_SIZE_X + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE,
        GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
constexpr unsigned int GRID_X_PAR_BLOCK = BLOCK_SIZE / GRID_IMAGE_SIZE_Y;
constexpr unsigned int
        NON_GRID_IMAGE_SIZE_X = IMAGE_SIZE_X - GRID_IMAGE_SIZE_X,
        NON_GRID_IMAGE_SIZE_Y = IMAGE_SIZE_Y - GRID_IMAGE_SIZE_Y;

__device__ void inline
eval_point(unsigned char *count,
           const num point_x, const num point_y, const num point_size,
           const unsigned int x_idx, const unsigned int y_idx) {
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;

    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    const num c_x = (num) x_idx / (num) IMAGE_SIZE_Y * point_size + point_x;
    const num c_y = (num) y_idx / (num) IMAGE_SIZE_Y * point_size + point_y;
    num z_x = 0, z_y = 0;

    #pragma unroll
    for (int i = 0; i < MAX_ITER; i++) {
        const num prev_z_x = z_x;

        z_x = z_x * z_x - z_y * z_y + c_x;
        z_y = prev_z_x * z_y * 2 + c_y;

        if (z_x * z_x + z_y * z_y > 4) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

    count[idx] = 255;
}

__global__ void
fill_x_grid(unsigned char *count,
            const num point_x, const num point_y, const num point_size) {
    const unsigned int
            x_idx = blockIdx.x * IMAGE_GRID_SIZE,
            y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

__global__ void
fill_y_grid(unsigned char *count,
            const num point_x, const num point_y, const num point_size) {
    const unsigned int
            x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            y_idx = threadIdx.y * IMAGE_GRID_SIZE;
    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

__global__ void
check_all_black(const unsigned char *count, bool *all_black) {
    const unsigned int
            grid_x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            grid_y_idx = threadIdx.y;
    const unsigned int
            x_idx = grid_x_idx * IMAGE_GRID_SIZE,
            y_idx = grid_y_idx * IMAGE_GRID_SIZE;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;
    const unsigned int grid_idx = grid_x_idx * GRID_IMAGE_SIZE_Y + grid_y_idx;
    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    bool all = true;

    for (int i = 0; i < IMAGE_GRID_SIZE + 1; i++) {
        if (y_idx + i < IMAGE_SIZE_Y) {
            all = all && count[idx + i] == 255;
        }
    }

    if (x_idx + IMAGE_GRID_SIZE < IMAGE_SIZE_X) {
        for (int i = 0; i < IMAGE_GRID_SIZE + 1; i++) {
            if (y_idx + i < IMAGE_SIZE_Y) {
                all = all && count[idx + IMAGE_GRID_SIZE * IMAGE_SIZE_Y + i] == 255;
            }
        }
    }

    for (int i = 1; i < IMAGE_GRID_SIZE; i++) {
        if (x_idx + i < IMAGE_SIZE_X) {
            all = all && count[idx + i * IMAGE_SIZE_Y] == 255;

            if (y_idx + IMAGE_GRID_SIZE < IMAGE_SIZE_Y) {
                all = all && count[idx + i * IMAGE_SIZE_Y + IMAGE_GRID_SIZE] == 255;
            }
        }
    }

    all_black[grid_idx] = all;
}

__global__ void
mandelbrot(unsigned char *count, const bool *all_black,
           const num point_x, const num point_y, const num point_size) {
    const unsigned int
            non_grid_x_idx = blockIdx.x,
            non_grid_y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int
            grid_idx_x = non_grid_x_idx / (IMAGE_GRID_SIZE - 1),
            grid_idx_y = non_grid_y_idx / (IMAGE_GRID_SIZE - 1);
    const unsigned int grid_idx = grid_idx_x * GRID_IMAGE_SIZE_Y + grid_idx_y;
    const unsigned int
            x_idx = non_grid_x_idx + grid_idx_x + 1,
            y_idx = non_grid_y_idx + grid_idx_y + 1;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;
    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    if (all_black[grid_idx]) {
        count[idx] = 255;
        return;
    }
    eval_point(count, point_x, point_y, point_size,
               x_idx, y_idx);
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

num point_x = -2.0;
num point_y = -2.0;
num point_size = 4.0;
int mouse_prev_x = 0, mouse_prev_y = 0;
bool mouse_flag = false,
        show_fps = true;

void mouse_callback(int event, int x, int y, int flags, void *userdata) {
    if (event == cv::EVENT_MOUSEWHEEL) {
        constexpr num zoom_scale = 0.9;
        num pixel = point_size / IMAGE_SIZE_Y;
        if (cv::getMouseWheelDelta(flags) > 0) {
            point_y += (num) x * pixel * (1 - zoom_scale);
            point_x += (num) y * pixel * (1 - zoom_scale);
            point_size *= zoom_scale;
        }
        if (cv::getMouseWheelDelta(flags) < 0) {
            point_y += (num) x * pixel * (1 - 1 / zoom_scale);
            point_x += (num) y * pixel * (1 - 1 / zoom_scale);
            point_size /= zoom_scale;
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

    if (event == cv::EVENT_RBUTTONDOWN) {
        show_fps ^= true;
    }
}

template<typename ... Args>
std::string format(const std::string &fmt, Args ... args) {
    size_t len = std::snprintf(nullptr, 0, fmt.c_str(), args ...);
    std::vector<char> buf(len + 1);
    std::snprintf(&buf[0], len + 1, fmt.c_str(), args ...);
    return std::string(&buf[0], &buf[0] + len);
}

int main() {
    dim3 main_grid(NON_GRID_IMAGE_SIZE_X, (NON_GRID_IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 main_block(BLOCK_SIZE);
    dim3 x_grid(GRID_IMAGE_SIZE_X, (IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 x_block(BLOCK_SIZE);
    assert(GRID_IMAGE_SIZE_Y < BLOCK_SIZE);
    dim3 y_grid((IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
    dim3 y_block(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);
    dim3 check_grid((GRID_IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
    dim3 check_block(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);

    unsigned int n_bytes = IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(unsigned char);
    unsigned int n_bytes_bool = GRID_IMAGE_SIZE_X * GRID_IMAGE_SIZE_Y * sizeof(bool);

    unsigned char *count, *count_gpu;
    bool *all_black;
    count = (unsigned char *) malloc(n_bytes);
    cudaMalloc((unsigned char **) &count_gpu, n_bytes);
    cudaMalloc((bool **) &all_black, n_bytes_bool);

    cv::Mat image = cv::Mat::zeros(IMAGE_SIZE_X, IMAGE_SIZE_Y, CV_8UC3);
    cv::namedWindow("image");
    cv::setMouseCallback("image", mouse_callback);

    std::chrono::system_clock::time_point prev_time, now_time;
    prev_time = std::chrono::system_clock::now();

    while (true) {
        fill_x_grid<<<x_grid, x_block>>>(count_gpu, point_x, point_y, point_size);
        fill_y_grid<<<y_grid, y_block>>>(count_gpu, point_x, point_y, point_size);
        cudaDeviceSynchronize();
        check_all_black<<<check_grid, check_block>>>(count_gpu, all_black);
        cudaDeviceSynchronize();

        mandelbrot<<<main_grid, main_block>>>(count_gpu, all_black, point_x, point_y, point_size);
        cudaMemcpy(count, count_gpu, n_bytes, cudaMemcpyDeviceToHost);
        array_to_image(count, image);

        now_time = std::chrono::system_clock::now();
        auto process_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - prev_time);
        double fps = 1000000.0 / (double) process_time.count();
        if (show_fps) {
            cv::putText(image,
                        format("FPS: %.3f", fps),
                        cv::Point(0, 20),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(255, 255, 255),
                        1);
        }
        prev_time = now_time;

        cv::imshow("image", image);

        int key = cv::waitKey(3);
        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
