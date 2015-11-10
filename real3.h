#pragma once
#include "real.h"

class real3 {
  public:
    CUDA_HOST_DEVICE inline real3() : x(0), y(0), z(0) {}
    CUDA_HOST_DEVICE inline real3(real a) : x(a), y(a), z(a) {}
    CUDA_HOST_DEVICE inline real3(real _x, real _y, real _z) : x(_x), y(_y), z(_z) {}
    CUDA_HOST_DEVICE inline real3(const real* p) : x(p[0]), y(p[1]), z(p[2]) {}
    // ========================================================================================

    CUDA_HOST_DEVICE inline operator real*() { return &x; }
    CUDA_HOST_DEVICE inline operator const real*() const { return &x; };
    // ========================================================================================

    CUDA_HOST_DEVICE inline void Set(real _x, real _y, real _z) {
        x = _x;
        y = _y;
        z = _z;
    }
    CUDA_HOST_DEVICE inline real3 operator*(real scale) const {
        real3 r(*this);
        r *= scale;
        return r;
    }
    CUDA_HOST_DEVICE inline real3 operator/(real scale) const {
        real3 r(*this);
        r /= scale;
        return r;
    }
    CUDA_HOST_DEVICE inline real3 operator+(const real3& v) const {
        real3 r(*this);
        r += v;
        return r;
    }
    CUDA_HOST_DEVICE inline real3 operator-(const real3& v) const {
        real3 r(*this);
        r -= v;
        return r;
    }
    CUDA_HOST_DEVICE inline real3 operator/(const real3& v) const {
        real3 r(*this);
        r /= v;
        return r;
    }
    CUDA_HOST_DEVICE inline real3 operator*(const real3& v) const {
        real3 r(*this);
        r *= v;
        return r;
    }
    // ========================================================================================

    CUDA_HOST_DEVICE inline real3& operator*=(real scale) {
        x *= scale;
        y *= scale;
        z *= scale;
        return *this;
    }
    CUDA_HOST_DEVICE inline real3& operator/=(real scale) {
        real s(1.0f / scale);
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    CUDA_HOST_DEVICE inline real3& operator+=(const real3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    CUDA_HOST_DEVICE inline real3& operator-=(const real3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    CUDA_HOST_DEVICE inline real3& operator/=(const real3& v) {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    CUDA_HOST_DEVICE inline real3& operator*=(const real3& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    CUDA_HOST_DEVICE inline bool operator!=(const real3& v) const { return (x != v.x || y != v.y || z != v.z); }

    CUDA_HOST_DEVICE inline real3 operator-() const { return real3(-x, -y, -z); }

    real x, y, z;
};

static CUDA_HOST_DEVICE real3 operator*(real lhs, const real3& rhs) {
    real3 r(rhs);
    r *= lhs;
    return r;
}

static CUDA_HOST_DEVICE real3 operator/(real lhs, const real3& rhs) {
    real3 r(rhs);
    r.x = lhs / r.x;
    r.y = lhs / r.y;
    r.z = lhs / r.z;
    return r;
}

static CUDA_HOST_DEVICE bool operator==(const real3& lhs, const real3& rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

static CUDA_HOST_DEVICE inline real Dot(const real3& v1, const real3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

static CUDA_HOST_DEVICE inline real Dot(const real3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

static CUDA_HOST_DEVICE inline real3 Sqrt(real3 v) {
    return real3(Sqrt(v.x), Sqrt(v.y), Sqrt(v.z));
}

static CUDA_HOST_DEVICE inline real Length2(const real3& v1) {
    return v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
}

static CUDA_HOST_DEVICE inline real3 Cross(const real3& b, const real3& c) {
    return real3(b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z, b.x * c.y - b.y * c.x);
}

static CUDA_HOST_DEVICE inline real3 Max(const real3& a, const real3& b) {
    return real3(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
}

static CUDA_HOST_DEVICE inline real3 Min(const real3& a, const real3& b) {
    return real3(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
}

static CUDA_HOST_DEVICE inline real3 Max(const real3& a, const real& b) {
    return real3(Max(a.x, b), Max(a.y, b), Max(a.z, b));
}

static CUDA_HOST_DEVICE inline real3 Min(const real3& a, const real& b) {
    return real3(Min(a.x, b), Min(a.y, b), Min(a.z, b));
}

static inline CUDA_HOST_DEVICE bool IsZero(const real3& v) {
    return Abs(v.x) < C_EPSILON && Abs(v.y) < C_EPSILON && Abs(v.z) < C_EPSILON;
}

// Used by cub
struct real3Min {
    inline CUDA_DEVICE real3 operator()(const real3& a, const real3& b) { return Min(a, b); }
};

struct real3Max {
    inline CUDA_DEVICE real3 operator()(const real3& a, const real3& b) { return Max(a, b); }
};

static inline CUDA_DEVICE void AtomicAdd(real3* pointer, real3 val) {
    atomicAdd(&pointer->x, val.x);
    atomicAdd(&pointer->y, val.y);
    atomicAdd(&pointer->z, val.z);
}

static inline CUDA_DEVICE void AtomicMax(real3* pointer, real3 val) {
    AtomicMaxf(&pointer->x, val.x);
    AtomicMaxf(&pointer->y, val.y);
    AtomicMaxf(&pointer->z, val.z);
}

static inline CUDA_DEVICE void AtomicMin(real3* pointer, real3 val) {
    AtomicMinf(&pointer->x, val.x);
    AtomicMinf(&pointer->y, val.y);
    AtomicMinf(&pointer->z, val.z);
}

static inline CUDA_DEVICE real3 Clamp(const real3& v, float max_length) {
    real3 x = v;
    float len_sq = Dot(x, x);
    float inv_len = rsqrtf(len_sq);

    if (len_sq > Sqr(max_length))
        x *= inv_len * max_length;

    return x;
}

static inline CUDA_HOST_DEVICE real3 OrthogonalVector(const real3& v) {
    real abs_x = Abs(v.x);
    real abs_y = Abs(v.y);
    real abs_z = Abs(v.z);
    if (abs_x < abs_y) {
        return abs_x < abs_z ? real3(0, v.z, -v.y) : real3(v.y, -v.x, 0);
    } else {
        return abs_y < abs_z ? real3(-v.z, 0, v.x) : real3(v.y, -v.x, 0);
    }
}

static inline CUDA_HOST_DEVICE real3 UnitOrthogonalVector(const real3& v) {
    return Normalize(OrthogonalVector(v));
}

static inline CUDA_HOST_DEVICE void Sort(real& a, real& b, real& c) {
    if (a > b)
        Swap(a, b);
    if (b > c)
        Swap(b, c);
    if (a > b)
        Swap(a, b);
}
