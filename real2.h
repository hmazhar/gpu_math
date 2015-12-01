#pragma once
#include "real.h"
#include <stdio.h>
class real2 {
  public:
    CUDA_HOST_DEVICE real2() : x(0.0f), y(0.0f) {}
    CUDA_HOST_DEVICE real2(real _x) : x(_x), y(_x) {}
    CUDA_HOST_DEVICE real2(real _x, real _y) : x(_x), y(_y) {}
    CUDA_HOST_DEVICE real2(const real* p) : x(p[0]), y(p[1]) {}
    CUDA_HOST_DEVICE explicit real2(const real2& v) : x(v.x), y(v.y) {}

    CUDA_HOST_DEVICE operator real*() { return &x; }
    CUDA_HOST_DEVICE operator const real*() const { return &x; };

    CUDA_HOST_DEVICE void Set(real x_, real y_) {
        x = x_;
        y = y_;
    }

    CUDA_HOST_DEVICE real2 operator*(real scale) const {
        real2 r(*this);
        r *= scale;
        return r;
    }
    CUDA_HOST_DEVICE real2 operator/(real scale) const {
        real2 r(*this);
        r /= scale;
        return r;
    }
    CUDA_HOST_DEVICE real2 operator+(const real2& v) const {
        real2 r(*this);
        r += v;
        return r;
    }
    CUDA_HOST_DEVICE real2 operator-(const real2& v) const {
        real2 r(*this);
        r -= v;
        return r;
    }

    CUDA_HOST_DEVICE real2& operator*=(real scale) {
        x *= scale;
        y *= scale;
        return *this;
    }
    CUDA_HOST_DEVICE real2& operator/=(real scale) {
        real s(1.0f / scale);
        x *= s;
        y *= s;
        return *this;
    }
    CUDA_HOST_DEVICE real2& operator+=(const real2& v) {
        x += v.x;
        y += v.y;
        return *this;
    }
    CUDA_HOST_DEVICE real2& operator-=(const real2& v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    CUDA_HOST_DEVICE real2& operator*=(const real2& scale) {
        x *= scale.x;
        y *= scale.y;
        return *this;
    }
    CUDA_HOST_DEVICE real2 operator-() const { return real2(-x, -y); }

    real x;
    real y;
};

static CUDA_HOST_DEVICE real2 operator*(real lhs, const real2& rhs) {
    real2 r(rhs);
    r *= lhs;
    return r;
}

static CUDA_HOST_DEVICE real2 operator*(const real2& lhs, const real2& rhs) {
    real2 r(lhs);
    r *= rhs;
    return r;
}

static CUDA_HOST_DEVICE bool operator==(const real2& lhs, const real2& rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

static CUDA_HOST_DEVICE real2 Max(const real2& a, const real2& b) {
    return real2(Max(a.x, b.x), Max(a.y, b.y));
}

static CUDA_HOST_DEVICE real2 Min(const real2& a, const real2& b) {
    return real2(Min(a.x, b.x), Min(a.y, b.y));
}

static CUDA_HOST_DEVICE inline real Dot(const real2& v1, const real2& v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

static CUDA_HOST_DEVICE inline real Dot(const real2& v) {
    return v.x * v.x + v.y * v.y;
}

static CUDA_HOST_DEVICE inline real Length2(const real2& v1) {
    return v1.x * v1.x + v1.y * v1.y;
}

static inline CUDA_DEVICE void AtomicAdd(real2* pointer, real2 val) {
    atomicAdd(&pointer->x, val.x);
    atomicAdd(&pointer->y, val.y);
}

static inline CUDA_DEVICE void AtomicMax(real2* pointer, real2 val) {
    AtomicMaxf(&pointer->x, val.x);
    AtomicMaxf(&pointer->y, val.y);
}

static inline CUDA_DEVICE void AtomicMin(real2* pointer, real2 val) {
    AtomicMinf(&pointer->x, val.x);
    AtomicMinf(&pointer->y, val.y);
}

static CUDA_HOST_DEVICE void Print(real2 v, const char* name) {
    printf("%s\n", name);
    printf("%f %f\n", v.x, v.y);
}
