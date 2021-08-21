#include <opencv2/opencv.hpp>
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

constexpr unsigned int GRID_LEVEL_NUM = 5;
constexpr unsigned int GRID_SIZE_ARRAY[GRID_LEVEL_NUM] = {256, 128, 64, 32, 16};
constexpr unsigned int GRID_SIZE_SCALE = 2;

constexpr unsigned int MIN_GRID_SIZE = GRID_SIZE_ARRAY[GRID_LEVEL_NUM - 1];
constexpr unsigned int
        NON_GRID_IMAGE_SIZE_X = IMAGE_SIZE_X - (IMAGE_SIZE_X + MIN_GRID_SIZE - 1) / MIN_GRID_SIZE,
        NON_GRID_IMAGE_SIZE_Y = IMAGE_SIZE_Y - (IMAGE_SIZE_Y + MIN_GRID_SIZE - 1) / MIN_GRID_SIZE;
//constexpr unsigned int IMAGE_GRID_SIZE = 159;
//
//constexpr unsigned int
//        GRID_IMAGE_SIZE_X = (IMAGE_SIZE_X + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE,
//        GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
//constexpr unsigned int GRID_X_PAR_BLOCK = BLOCK_SIZE / GRID_IMAGE_SIZE_Y;
//constexpr unsigned int
//        NON_GRID_IMAGE_SIZE_X = IMAGE_SIZE_X - GRID_IMAGE_SIZE_X,
//        NON_GRID_IMAGE_SIZE_Y = IMAGE_SIZE_Y - GRID_IMAGE_SIZE_Y;

__device__ void inline
eval_point(unsigned char *count,
           const num point_x, const num point_y, const num point_size,
           const unsigned int x_idx, const unsigned int y_idx) {
    constexpr unsigned int
            LOOP_SEP1 = 96,
            LOOP_SEP2 = 192,
            LOOP_SEP3 = 384,
            LOOP_SEP4 = 768;

    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;

    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    const num c_x = (num) x_idx / (num) IMAGE_SIZE_Y * point_size + point_x;
    const num c_y = (num) y_idx / (num) IMAGE_SIZE_Y * point_size + point_y;
    num z_x = 0, z_y = 0;


#pragma unroll
    for (int i = 0; i < LOOP_SEP1; i++) {
        const num prev_z_x = z_x;

        z_x = z_x * z_x - z_y * z_y + c_x;
        z_y = prev_z_x * z_y * 2 + c_y;

        if (z_x * z_x + z_y * z_y > 4) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

#pragma unroll
    for (int i = LOOP_SEP1; i < LOOP_SEP2; i += 4) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const num prev_z_x = z_x;

            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = prev_z_x * z_y * 2 + c_y;
        }

        const num r2 = z_x * z_x + z_y * z_y;
        if (r2 > 4 || isnan(r2)) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

#pragma unroll
    for (int i = LOOP_SEP2; i < LOOP_SEP3; i += 8) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const num prev_z_x = z_x;

            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = prev_z_x * z_y * 2 + c_y;
        }

        const num r2 = z_x * z_x + z_y * z_y;
        if (r2 > 4 || isnan(r2)) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

#pragma unroll
    for (int i = LOOP_SEP3; i < LOOP_SEP4; i += 16) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const num prev_z_x = z_x;

            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = prev_z_x * z_y * 2 + c_y;
        }

        const num r2 = z_x * z_x + z_y * z_y;
        if (r2 > 4 || isnan(r2)) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

#pragma unroll
    for (int i = LOOP_SEP4; i < MAX_ITER; i += 32) {
#pragma unroll
        for (int j = 0; j < 32; j++) {
            const num prev_z_x = z_x;

            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = prev_z_x * z_y * 2 + c_y;
        }

        const num r2 = z_x * z_x + z_y * z_y;
        if (r2 > 4 || isnan(r2)) {
            count[idx] = (unsigned char) (log2((num) i + 1) / 16.0f * 256.0f);
            return;
        }
    }

    count[idx] = 255;
}

template<unsigned int IMAGE_GRID_SIZE>
__global__ void
fill_x_grid(unsigned char *count,
            const num point_x, const num point_y, const num point_size) {
    const unsigned int
            x_idx = blockIdx.x * IMAGE_GRID_SIZE,
            y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

template<unsigned int IMAGE_GRID_SIZE>
__global__ void
fill_y_grid(unsigned char *count,
            const num point_x, const num point_y, const num point_size) {
    const unsigned int
            x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            y_idx = (blockIdx.y * blockDim.y + threadIdx.y) * IMAGE_GRID_SIZE;
    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

template<unsigned int IMAGE_GRID_SIZE, unsigned int GRID_IMAGE_SIZE_Y>
__global__ void
fill_next_x_grid(unsigned char *count, const bool *prev_all_black,
                 const num point_x, const num point_y, const num point_size) {
    const unsigned int prev_grid_x_idx = blockIdx.x;
    const unsigned int
            x_idx = (2 * prev_grid_x_idx + 1) * IMAGE_GRID_SIZE,
            y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;
    const unsigned int prev_grid_y_idx = y_idx / (IMAGE_GRID_SIZE * GRID_SIZE_SCALE);
    constexpr unsigned int PREV_GRID_IMAGE_SIZE_Y = (GRID_IMAGE_SIZE_Y + GRID_SIZE_SCALE - 1) / GRID_SIZE_SCALE;
    const unsigned int prev_grid_idx = prev_grid_x_idx * PREV_GRID_IMAGE_SIZE_Y + prev_grid_y_idx;

    if (prev_all_black[prev_grid_idx]) {
        return;
    }

    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

template<unsigned int IMAGE_GRID_SIZE, unsigned int GRID_IMAGE_SIZE_Y>
__global__ void
fill_next_y_grid(unsigned char *count, const bool *prev_all_black,
                 const num point_x, const num point_y, const num point_size) {
    const unsigned int prev_grid_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int
            x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            y_idx = (2 * (prev_grid_y_idx) + 1) * IMAGE_GRID_SIZE;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;
    const unsigned int prev_grid_x_idx = x_idx / (IMAGE_GRID_SIZE * GRID_SIZE_SCALE);
    constexpr unsigned int PREV_GRID_IMAGE_SIZE_Y = (GRID_IMAGE_SIZE_Y + GRID_SIZE_SCALE - 1) / GRID_SIZE_SCALE;
    const unsigned int prev_grid_idx = prev_grid_x_idx * PREV_GRID_IMAGE_SIZE_Y + prev_grid_y_idx;

    if (prev_all_black[prev_grid_idx]) {
        return;
    }

    eval_point(count, point_x, point_y, point_size, x_idx, y_idx);
}

template<unsigned int IMAGE_GRID_SIZE, unsigned int GRID_IMAGE_SIZE_Y>
__global__ void
check_all_black(const unsigned char *count, bool *all_black) {
    const unsigned int
            grid_x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            grid_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
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

template<unsigned int IMAGE_GRID_SIZE, unsigned int GRID_IMAGE_SIZE_Y>
__global__ void
check_all_black(const unsigned char *count, const bool *prev_all_black, bool *all_black) {
    const unsigned int
            grid_x_idx = blockIdx.x * blockDim.x + threadIdx.x,
            grid_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int
            x_idx = grid_x_idx * IMAGE_GRID_SIZE,
            y_idx = grid_y_idx * IMAGE_GRID_SIZE;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;
    const unsigned int grid_idx = grid_x_idx * GRID_IMAGE_SIZE_Y + grid_y_idx;

    const unsigned int
            prev_grid_x_idx = grid_x_idx / GRID_SIZE_SCALE,
            prev_grid_y_idx = grid_y_idx / GRID_SIZE_SCALE;
    constexpr unsigned int PREV_GRID_IMAGE_SIZE_Y = (GRID_IMAGE_SIZE_Y + GRID_SIZE_SCALE - 1) / GRID_SIZE_SCALE;
    const unsigned int prev_grid_idx = prev_grid_x_idx * PREV_GRID_IMAGE_SIZE_Y + prev_grid_y_idx;
    if (prev_all_black[prev_grid_idx]) {
        all_black[grid_idx] = true;
        return;
    }

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

template<unsigned int IMAGE_GRID_SIZE>
void check_first_grid(unsigned char *count_gpu, bool *all_black,
                      const num point_x, const num point_y, const num point_size) {
    constexpr unsigned int
            GRID_IMAGE_SIZE_X = (IMAGE_SIZE_X + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE,
            GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
    constexpr unsigned int GRID_X_PAR_BLOCK = BLOCK_SIZE / GRID_IMAGE_SIZE_Y;

    dim3 x_grid(GRID_IMAGE_SIZE_X, (IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 x_block(BLOCK_SIZE);

    dim3 y_grid;
    dim3 y_block;
    if (GRID_IMAGE_SIZE_Y < BLOCK_SIZE) {
        y_grid = dim3((IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
        y_block = dim3(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);
    } else {
        y_grid = dim3(IMAGE_SIZE_X, (GRID_IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        y_block = dim3(1, BLOCK_SIZE);
    }

    dim3 check_grid;
    dim3 check_block;
    if (GRID_IMAGE_SIZE_Y < BLOCK_SIZE) {
        check_grid = dim3((GRID_IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
        check_block = dim3(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);
    } else {
        check_grid = dim3(GRID_IMAGE_SIZE_X, (GRID_IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        check_block = dim3(1, BLOCK_SIZE);
    }

    fill_x_grid<IMAGE_GRID_SIZE> <<<x_grid, x_block>>>(count_gpu, point_x, point_y, point_size);
    fill_y_grid<IMAGE_GRID_SIZE> <<<y_grid, y_block>>>(count_gpu, point_x, point_y, point_size);
    cudaDeviceSynchronize();
    check_all_black<IMAGE_GRID_SIZE, GRID_IMAGE_SIZE_Y> <<<check_grid, check_block>>>(count_gpu, all_black);
    cudaDeviceSynchronize();
}

template<unsigned int IMAGE_GRID_SIZE>
void check_next_grid(unsigned char *count_gpu, const bool *prev_all_black, bool *all_black,
                     const num point_x, const num point_y, const num point_size) {
    constexpr unsigned int
            GRID_IMAGE_SIZE_X = (IMAGE_SIZE_X + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE,
            GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
    constexpr unsigned int GRID_X_PAR_BLOCK = BLOCK_SIZE / GRID_IMAGE_SIZE_Y;

    dim3 x_grid(GRID_IMAGE_SIZE_X, (IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 x_block(BLOCK_SIZE);

    dim3 y_grid;
    dim3 y_block;
    if (GRID_IMAGE_SIZE_Y < BLOCK_SIZE) {
        y_grid = dim3((IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
        y_block = dim3(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);
    } else {
        y_grid = dim3(IMAGE_SIZE_X, (GRID_IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        y_block = dim3(1, BLOCK_SIZE);
    }

    dim3 check_grid;
    dim3 check_block;
    if (GRID_IMAGE_SIZE_Y < BLOCK_SIZE) {
        check_grid = dim3((GRID_IMAGE_SIZE_X + GRID_X_PAR_BLOCK - 1) / GRID_X_PAR_BLOCK);
        check_block = dim3(GRID_X_PAR_BLOCK, GRID_IMAGE_SIZE_Y);
    } else {
        check_grid = dim3(GRID_IMAGE_SIZE_X, (GRID_IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        check_block = dim3(1, BLOCK_SIZE);
    }

    fill_next_x_grid<IMAGE_GRID_SIZE, GRID_IMAGE_SIZE_Y><<<x_grid, x_block>>>
            (count_gpu, prev_all_black, point_x, point_y, point_size);
    fill_next_y_grid<IMAGE_GRID_SIZE, GRID_IMAGE_SIZE_Y><<<y_grid, y_block>>>
            (count_gpu, prev_all_black, point_x, point_y, point_size);
    cudaDeviceSynchronize();
    check_all_black<IMAGE_GRID_SIZE, GRID_IMAGE_SIZE_Y><<<check_grid, check_block>>>
            (count_gpu, prev_all_black, all_black);
    cudaDeviceSynchronize();
}

template<unsigned int IMAGE_GRID_SIZE>
__global__ void
fill_not_black(unsigned char *count, const bool *all_black,
               const num point_x, const num point_y, const num point_size) {
    constexpr unsigned int GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
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

    if (all_black[grid_idx]) {
        return;
    }
    eval_point(count, point_x, point_y, point_size,
               x_idx, y_idx);
}

template<unsigned int IMAGE_GRID_SIZE>
__global__ void
fill_black(unsigned char *count, const bool *all_black) {
    constexpr unsigned int GRID_IMAGE_SIZE_Y = (IMAGE_SIZE_Y + IMAGE_GRID_SIZE - 1) / IMAGE_GRID_SIZE;
    const unsigned int
            x_idx = blockIdx.x,
            y_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int
            grid_idx_x = x_idx / IMAGE_GRID_SIZE,
            grid_idx_y = y_idx / IMAGE_GRID_SIZE;
    const unsigned int grid_idx = grid_idx_x * GRID_IMAGE_SIZE_Y + grid_idx_y;
    if (!(x_idx < IMAGE_SIZE_X && y_idx < IMAGE_SIZE_Y)) return;

    const unsigned int idx = x_idx * IMAGE_SIZE_Y + y_idx;

    if (all_black[grid_idx]) {
        count[idx] = 255;
    }
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
    dim3 black_grid(IMAGE_SIZE_X, (IMAGE_SIZE_Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 black_block(BLOCK_SIZE);
    unsigned int n_bytes = IMAGE_SIZE_X * IMAGE_SIZE_Y * sizeof(unsigned char);


    unsigned char *count, *count_gpu;
    bool *all_black[GRID_LEVEL_NUM];
    count = (unsigned char *) malloc(n_bytes);
    cudaMalloc((unsigned char **) &count_gpu, n_bytes);

    for (int i = 0; i < GRID_LEVEL_NUM; i++) {
        const unsigned int image_grid_size = GRID_SIZE_ARRAY[i];
        const unsigned int
                grid_image_size_x = (IMAGE_SIZE_X + image_grid_size - 1) / image_grid_size,
                grid_image_size_y = (IMAGE_SIZE_Y + image_grid_size - 1) / image_grid_size;
        unsigned int n_bytes_bool = grid_image_size_x * grid_image_size_y * sizeof(bool);
        cudaMalloc((bool **) &(all_black[i]), n_bytes_bool);
    }

    cv::Mat image = cv::Mat::zeros(IMAGE_SIZE_X, IMAGE_SIZE_Y, CV_8UC3);
    cv::namedWindow("image");
    cv::setMouseCallback("image", mouse_callback);

    std::chrono::system_clock::time_point prev_time, now_time;
    prev_time = std::chrono::system_clock::now();

    while (true) {
        check_first_grid<GRID_SIZE_ARRAY[0]>(count_gpu, all_black[0],
                                             point_x, point_y, point_size);
        check_next_grid<GRID_SIZE_ARRAY[1]>(count_gpu, all_black[0], all_black[1],
                                            point_x, point_y, point_size);
        check_next_grid<GRID_SIZE_ARRAY[2]>(count_gpu, all_black[1], all_black[2],
                                            point_x, point_y, point_size);
        check_next_grid<GRID_SIZE_ARRAY[3]>(count_gpu, all_black[2], all_black[3],
                                            point_x, point_y, point_size);
        check_next_grid<GRID_SIZE_ARRAY[4]>(count_gpu, all_black[3], all_black[4],
                                            point_x, point_y, point_size);

        fill_not_black<MIN_GRID_SIZE><<<main_grid, main_block>>>(count_gpu, all_black[GRID_LEVEL_NUM - 1],
                                                                 point_x, point_y, point_size);
        fill_black<MIN_GRID_SIZE><<<black_grid, black_block>>>(count_gpu, all_black[GRID_LEVEL_NUM - 1]);
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
