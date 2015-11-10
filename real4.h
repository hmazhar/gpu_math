#pragma once
#include "real3.h"

class real4 {
  public:
    CUDA_HOST_DEVICE real4() : x(0), y(0), z(0), w(0) {}
    CUDA_HOST_DEVICE real4(real a) : x(a), y(a), z(a), w(a) {}
    CUDA_HOST_DEVICE real4(real _x, real _y, real _z, real _w) : x(_x), y(_y), z(_z), w(_w) {}
    CUDA_HOST_DEVICE real4(const real* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}

    CUDA_HOST_DEVICE real4(const real3& v, real w) : x(v.x), y(v.y), z(v.z), w(w) {}

    CUDA_HOST_DEVICE operator real*() { return &x; }
    CUDA_HOST_DEVICE operator const real*() const { return &x; };

    CUDA_HOST_DEVICE void Set(real _x, real _y, real _z, real _w) {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    CUDA_HOST_DEVICE real4 operator+(const real4& v) const {
        real4 r(*this);
        r += v;
        return r;
    }
    CUDA_HOST_DEVICE real4 operator+(const real3& v) const {
        real4 r(*this);
        r += v;
        return r;
    }
    CUDA_HOST_DEVICE real4 operator-(const real4& v) const {
        real4 r(*this);
        r -= v;
        return r;
    }
    CUDA_HOST_DEVICE real4 operator-(const real3& v) const {
        real4 r(*this);
        r -= v;
        return r;
    }
    CUDA_HOST_DEVICE real4 operator*(real scale) const {
        real4 r(*this);
        r *= scale;
        return r;
    }
    CUDA_HOST_DEVICE real4 operator/(real scale) const {
        real4 r(*this);
        r /= scale;
        return r;
    }

    CUDA_HOST_DEVICE real4 operator*(real4 scale) const {
        real4 r(*this);
        r *= scale;
        return r;
    }
    CUDA_HOST_DEVICE real4& operator+=(const real4& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator+=(const real3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator-=(const real4& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator-=(const real3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator*=(real scale) {
        x *= scale;
        y *= scale;
        z *= scale;
        w *= scale;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator*=(const real4& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }
    CUDA_HOST_DEVICE real4& operator/=(real scale) {
        real s(1.0f / scale);
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }

    CUDA_HOST_DEVICE bool operator!=(const real4& v) const { return (x != v.x || y != v.y || z != v.z || w != v.w); }

    CUDA_HOST_DEVICE real4 operator-() const { return real4(-x, -y, -z, -w); }

    real x, y, z, w;
};

static CUDA_HOST_DEVICE bool operator==(const real4& lhs, const real4& rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}

inline CUDA_DEVICE void AtomicAdd(real4* pointer, real4 val) {
    atomicAdd(&pointer->x, val.x);
    atomicAdd(&pointer->y, val.y);
    atomicAdd(&pointer->z, val.z);
    atomicAdd(&pointer->w, val.w);
}

inline CUDA_DEVICE void AtomicMax(real4* pointer, real4 val) {
    AtomicMaxf(&pointer->x, val.x);
    AtomicMaxf(&pointer->y, val.y);
    AtomicMaxf(&pointer->z, val.z);
    AtomicMaxf(&pointer->w, val.w);
}

inline CUDA_DEVICE void AtomicMin(real4* pointer, real4 val) {
    AtomicMinf(&pointer->x, val.x);
    AtomicMinf(&pointer->y, val.y);
    AtomicMinf(&pointer->z, val.z);
    AtomicMinf(&pointer->w, val.w);
}

// Quaternion Class
// ========================================================================================
class quaternion {
  public:
    CUDA_HOST_DEVICE quaternion() : x(0), y(0), z(0), w(0) {}
    CUDA_HOST_DEVICE quaternion(real a) : x(a), y(a), z(a), w(a) {}
    CUDA_HOST_DEVICE quaternion(real _w, real _x, real _y, real _z) : x(_x), y(_y), z(_z), w(_w) {}
    CUDA_HOST_DEVICE quaternion(const real* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}

    CUDA_HOST_DEVICE quaternion(const real3& v, real w) : x(v.x), y(v.y), z(v.z), w(w) {}

    CUDA_HOST_DEVICE operator real*() { return &x; }
    CUDA_HOST_DEVICE operator const real*() const { return &x; };

    CUDA_HOST_DEVICE void Set(real _w, real _x, real _y, real _z) {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }

    CUDA_HOST_DEVICE quaternion operator*(real scale) const {
        quaternion q;
        q.x = x * scale;
        q.y = y * scale;
        q.z = z * scale;
        q.w = w * scale;
        return q;
    }

    CUDA_HOST_DEVICE quaternion operator/(real scale) const {
        quaternion q;
        q.x = x / scale;
        q.y = y / scale;
        q.z = z / scale;
        q.w = w / scale;
        return q;
    }

    CUDA_HOST_DEVICE quaternion& operator*=(real scale) {
        x *= scale;
        y *= scale;
        z *= scale;
        w *= scale;
        return *this;
    }

    CUDA_HOST_DEVICE quaternion& operator/=(real scale) {
        x /= scale;
        y /= scale;
        z /= scale;
        w /= scale;
        return *this;
    }
    real x, y, z, w;
};

inline CUDA_HOST_DEVICE quaternion operator~(quaternion const& a) {
    return quaternion(a.w, -a.x, -a.y, -a.z);
}

static CUDA_HOST_DEVICE inline real Dot(const quaternion& v1, const quaternion& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

static CUDA_HOST_DEVICE inline real Dot(const quaternion& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

inline CUDA_HOST_DEVICE quaternion Mult(const quaternion& a, const quaternion& b) {
    quaternion temp;
    temp.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    temp.x = a.w * b.x + a.x * b.w - a.z * b.y + a.y * b.z;
    temp.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
    temp.z = a.w * b.z + a.z * b.w - a.y * b.x + a.x * b.y;
    return temp;
}

inline CUDA_HOST_DEVICE real3 Rotate(const real3& v, const quaternion& q) {
    real3 t = 2 * Cross(real3(q.x, q.y, q.z), v);
    return v + q.w * t + Cross(real3(q.x, q.y, q.z), t);
}

inline CUDA_HOST_DEVICE real3 RotateT(const real3& v, const quaternion& q) {
    return Rotate(v, ~q);
}

