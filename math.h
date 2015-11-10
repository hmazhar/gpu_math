#pragma once 
#include "defines.h"
#include "real.h"
#include "real2.h"
#include "real3.h"
#include "real4.h"
#include "matrix.h"
#include "other_types.h"

//Math Functions that use multiple types
// ========================================================================================

static inline CUDA_HOST_DEVICE real3 TransformLocalToParent(const real3& p, const quaternion& q, const real3& rl) {
    return p + Rotate(rl, q);
}

static inline CUDA_HOST_DEVICE real3 TransformParentToLocal(const real3& p, const quaternion& q, const real3& rp) {
  return RotateT(rp - p, q);
}

