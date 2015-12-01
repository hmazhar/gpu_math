#pragma once

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_CONSTANT __device__ __constant__
#define CUDA_SHARED __shared__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_CONSTANT
#define CUDA_SHARED
#define CUDA_GLOBAL
#endif


#define C_Pi 4 * atan(1.0f)
#define C_2Pi 2.0f * C_Pi
#define C_InvPi 1.0f / C_Pi
#define C_Inv2Pi 0.5f / C_Pi

#define C_DegToRad C_Pi / 180.0f
#define C_RadToDeg 180.0f / C_Pi

#define C_EPSILON FLT_EPSILON
#define C_LARGE_REAL FLT_MAX