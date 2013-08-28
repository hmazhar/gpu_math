#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#pragma once
//====================================INCLUDES=================================//
#include <fenv.h>
#include <vector_types.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/system/omp/execution_policy.h>
#include <omp.h>
#include <vector>
#include <string.h>

using namespace std;

//====================================DEFINITIONS=================================//

typedef unsigned int uint;


#ifdef __CDT_PARSER__
#define __host__
#define __device__
#define __global__
#define __constant__
#define __shared__
#define CUDA_KERNEL_DIM(...) ()
#define __KERNEL__(...) ()
#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>
#define __KERNEL__(...)  <<< __VA_ARGS__ >>>
#endif


#ifdef SIM_ENABLE_GPU_MODE
#define custom_vector thrust::device_vector
#else
#define custom_vector thrust::host_vector
#endif
