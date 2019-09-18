#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "rod_align_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


    __global__ void RODAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width,
                                    const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
	float bin_size_h = (float)(height - 1.001) / (aligned_height - 1.);
        float bin_size_w = (float)(width - 1.001) / (aligned_width - 1.);
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            // int n = index;
            // int pw = n % aligned_width;
            // n /= aligned_width;
            // int ph = n % aligned_height;
            // n /= aligned_height;
            // int c = n % channels;
            // n /= channels;

            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            // bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[n * 5 + 0];
            float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;


            float h = (float)(ph) * bin_size_h;
            float w = (float)(pw) * bin_size_w;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (h >= roi_start_h && h <= roi_end_h && w >= roi_start_w && w <= roi_end_w){
                top_data[index] = 0.;
            } else {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
                int upleft = img_start + (c * height + hstart) * width + wstart;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                    + bottom_data[upright] * (1. - h_ratio) * w_ratio
                    + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                    + bottom_data[downright] * h_ratio * w_ratio;
            }
        }
    }


    int RODAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
                               const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        RODAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, bottom_data, spatial_scale, height, width, channels,
          aligned_height, aligned_width, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


    __global__ void RODAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width,
                                     const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
	float bin_size_h = (float)(height - 1.001) / (aligned_height - 1.);
        float bin_size_w = (float)(width - 1.001) / (aligned_width - 1.);
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            float roi_batch_ind = bottom_rois[n * 5 + 0];
            float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;
            

            float h = (float)(ph) * bin_size_h;
            float w = (float)(pw) * bin_size_w;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (!(h >= roi_start_h && h <= roi_end_h && w >= roi_start_w && w <= roi_end_w)) {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
                int upleft = img_start + (c * height + hstart) * width + wstart;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
                atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
                atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
                atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
            }
        }
    }

    int RODAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
                                const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        RODAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, top_diff, spatial_scale, height, width, channels,
          aligned_height, aligned_width, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


#ifdef __cplusplus
}
#endif
