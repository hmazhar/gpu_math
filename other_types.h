#pragma once
#include "defines.h"
#include "real3.h"
#include "real4.h"

// Analytical Plane
// ========================================================================================
class Plane : public real4 {
  public:
    CUDA_HOST_DEVICE Plane() {}
    CUDA_HOST_DEVICE Plane(real x, real y, real z, real w) : real4(x, y, z, w) {}
    CUDA_HOST_DEVICE Plane(const real3& v) : real4(v.x, v.y, v.z, 1.0f) {}
    CUDA_HOST_DEVICE Plane(const real4& v) : real4(v) {}
    CUDA_HOST_DEVICE Plane(const real3& p, const real3& n) {
        x = n.x;
        y = n.y;
        z = n.z;
        w = -Dot(p, n);
    }

    CUDA_HOST_DEVICE real3 Normal() const { return real3(x, y, z); }
    CUDA_HOST_DEVICE real3 Point() const { return real3(x * -w, y * -w, z * -w); }
};

// Other Functions
// ========================================================================================
// nearest power of two to the input
static inline uint NearestPow(const uint& num) {
    uint n = num > 0 ? num - 1 : 0;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}