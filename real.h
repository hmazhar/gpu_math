#pragma once
#include "defines.h"
#include <float.h>
#define real float

// Trig Functions
// ========================================================================================
static inline CUDA_HOST_DEVICE real Sin(real theta) {
    return sinf(theta);
}
static inline CUDA_HOST_DEVICE real Cos(real theta) {
    return cosf(theta);
}
static inline CUDA_HOST_DEVICE real Tan(real theta) {
    return tanf(theta);
}
static inline CUDA_HOST_DEVICE real ASin(real theta) {
    return asinf(theta);
}
static inline CUDA_HOST_DEVICE real ACos(real theta) {
    return acosf(theta);
}
static inline CUDA_HOST_DEVICE real ATan(real theta) {
    return atanf(theta);
}
static inline CUDA_HOST_DEVICE real ATan2(real x, real y) {
    return atan2f(x, y);
}
static inline CUDA_HOST_DEVICE real DegToRad(real t) {
    return t * C_DegToRad;
}
static inline CUDA_HOST_DEVICE real RadToDeg(real t) {
    return t * C_RadToDeg;
}

// Geometric Functions
// ========================================================================================
static inline CUDA_HOST_DEVICE real Sqr(real x) {
    return x * x;
}
static inline CUDA_HOST_DEVICE real Cube(real x) {
    return x * x * x;
}
static inline CUDA_HOST_DEVICE real Sqrt(real x) {
    return sqrtf(x);
}
static inline CUDA_HOST_DEVICE real InvSqrt(real x) {
    return 1.0f / sqrtf(x);  // could also use rsqrtf(x) here and avoid division
}
static inline CUDA_HOST_DEVICE real Abs(real x) {
    return fabsf(x);
}
static inline CUDA_HOST_DEVICE real Pow(real b, real e) {
    return powf(b, e);
}
static inline CUDA_HOST_DEVICE real Mod(real x, real y) {
    return fmod(x, y);
}
static inline CUDA_HOST_DEVICE real Exp(real x) {
    return expf(x);
}
static inline CUDA_HOST_DEVICE real Sign(real x) {
    return x < 0.0f ? -1.0f : 1.0f;
}
static inline CUDA_HOST_DEVICE bool IsZero(real x) {
    return Abs(x) < C_EPSILON;
}
static inline CUDA_HOST_DEVICE real Min(real a, real b) {
    return fminf(a, b);
}
static inline CUDA_HOST_DEVICE real Max(real a, real b) {
    return fmaxf(a, b);
}

// Templated Functions, Will work on real vector types
// ========================================================================================

template <typename T>
inline CUDA_HOST_DEVICE bool IsEqual(const T& x, const T& y) {
    return IsZero(x - y);
}

template <typename T>
inline CUDA_HOST_DEVICE real LengthSq(const T v) {
    return Dot(v);
}

template <typename T>
inline CUDA_HOST_DEVICE real Length(const T& v) {
    return Sqrt(LengthSq(v));
}

template <typename T>
inline CUDA_HOST_DEVICE real SafeLength(const T& v) {
    real len_sq = LengthSq(v);
    if (len_sq) {
        return Sqrt(len_sq);
    } else {
        return 0.0f;
    }
}

template <typename T>
inline CUDA_HOST_DEVICE T Normalize(const T& v) {
   return v / Length(v);
}

template <typename T>
inline CUDA_HOST_DEVICE T SafeNormalize(const T& v, const T& safe = T()) {
    real len_sq = LengthSq(v);
    if (len_sq > 0.0f) {
        return v * InvSqrt(len_sq);
    } else {
        return safe;
    }
}

template <typename T, typename U>
inline CUDA_HOST_DEVICE T Lerp(const T& start, const T& end, const U& t) {
    return start + (end - start) * t;
}

template <typename T>
inline CUDA_HOST_DEVICE void Swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
inline CUDA_HOST_DEVICE T Clamp(T x, T low, T high) {
    if (low > high) {
        Swap(low, high);
    }
    return Max(low, Min(x, high));
}

template <typename T, typename U>
inline CUDA_HOST_DEVICE T ClampMin(T x, U low) {
    return Max(low, x);
}

template <typename T, typename U>
inline CUDA_HOST_DEVICE T ClampMax(T x, U high) {
    return Min(x, high);
}

// code adopted from http://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
// ========================================================================================
static inline CUDA_DEVICE float AtomicMaxf(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (value > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    }

    return __int_as_float(old);
}

static inline CUDA_DEVICE float AtomicMinf(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    }

    return __int_as_float(old);
}