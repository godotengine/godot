// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/ray.h"
#include "../common/geometry.h"

namespace embree
{
  namespace isa
  {
    struct CurvePrecalculations1
    {
      float depth_scale;
      LinearSpace3fa ray_space;
           
      __forceinline CurvePrecalculations1() {}

      __forceinline CurvePrecalculations1(const Ray& ray, const void* ptr)
      {
        depth_scale = rsqrt(dot(ray.dir,ray.dir));
        LinearSpace3fa space = frame(depth_scale*ray.dir);
        space.vz *= depth_scale;
        ray_space = space.transposed();
      }
    };
    
    template<int K>
      struct CurvePrecalculationsK
    {
      vfloat<K> depth_scale;
      LinearSpace3fa ray_space[K];

      __forceinline CurvePrecalculationsK(const vbool<K>& valid, const RayK<K>& ray)
      {
        size_t mask = movemask(valid);
        depth_scale = rsqrt(dot(ray.dir,ray.dir));
        while (mask) {
          size_t k = bscf(mask);
          Vec3fa ray_dir_k = Vec3fa(ray.dir.x[k],ray.dir.y[k],ray.dir.z[k]);
          LinearSpace3fa ray_space_k = frame(depth_scale[k]*ray_dir_k);
          ray_space_k.vz *= depth_scale[k];
          ray_space[k] = ray_space_k.transposed();
        }
      }
    };
  }
}
