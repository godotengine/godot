// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gregory_patch.h"

namespace embree
{  
  class __aligned(64) DenseGregoryPatch3fa
  {
    typedef Vec3fa Vec3fa_4x4[4][4];
  public:

    __forceinline DenseGregoryPatch3fa (const GregoryPatch3fa& patch)
    {
      for (size_t y=0; y<4; y++)
	for (size_t x=0; x<4; x++)
	  matrix[y][x] = Vec3ff(patch.v[y][x], 0.0f);
      
      matrix[0][0].w = patch.f[0][0].x;
      matrix[0][1].w = patch.f[0][0].y;
      matrix[0][2].w = patch.f[0][0].z;
      matrix[0][3].w = 0.0f;
      
      matrix[1][0].w = patch.f[0][1].x;
      matrix[1][1].w = patch.f[0][1].y;
      matrix[1][2].w = patch.f[0][1].z;
      matrix[1][3].w = 0.0f;
      
      matrix[2][0].w = patch.f[1][1].x;
      matrix[2][1].w = patch.f[1][1].y;
      matrix[2][2].w = patch.f[1][1].z;
      matrix[2][3].w = 0.0f;
      
      matrix[3][0].w = patch.f[1][0].x;
      matrix[3][1].w = patch.f[1][0].y;
      matrix[3][2].w = patch.f[1][0].z;
      matrix[3][3].w = 0.0f;
    }

    __forceinline void extract_f_m(Vec3fa f_m[2][2]) const
    {
      f_m[0][0] = Vec3fa( matrix[0][0].w, matrix[0][1].w, matrix[0][2].w );
      f_m[0][1] = Vec3fa( matrix[1][0].w, matrix[1][1].w, matrix[1][2].w );
      f_m[1][1] = Vec3fa( matrix[2][0].w, matrix[2][1].w, matrix[2][2].w );
      f_m[1][0] = Vec3fa( matrix[3][0].w, matrix[3][1].w, matrix[3][2].w );      
    }

    __forceinline Vec3fa eval(const float uu, const float vv) const
    {
      __aligned(64) Vec3fa f_m[2][2]; extract_f_m(f_m);
      return GregoryPatch3fa::eval(*(Vec3fa_4x4*)&matrix,f_m,uu,vv);
    }

    __forceinline Vec3fa normal(const float uu, const float vv) const
    {
      __aligned(64) Vec3fa f_m[2][2]; extract_f_m(f_m);
      return GregoryPatch3fa::normal(*(Vec3fa_4x4*)&matrix,f_m,uu,vv);
    }

    template<class T>
      __forceinline Vec3<T> eval(const T &uu, const T &vv) const 
    {
      Vec3<T> f_m[2][2];
      f_m[0][0] = Vec3<T>( matrix[0][0].w, matrix[0][1].w, matrix[0][2].w );
      f_m[0][1] = Vec3<T>( matrix[1][0].w, matrix[1][1].w, matrix[1][2].w );
      f_m[1][1] = Vec3<T>( matrix[2][0].w, matrix[2][1].w, matrix[2][2].w );
      f_m[1][0] = Vec3<T>( matrix[3][0].w, matrix[3][1].w, matrix[3][2].w );
      return GregoryPatch3fa::eval_t(*(Vec3fa_4x4*)&matrix,f_m,uu,vv);
    }
    
    template<class T>
      __forceinline Vec3<T> normal(const T &uu, const T &vv) const 
    {
      Vec3<T> f_m[2][2];
      f_m[0][0] = Vec3<T>( matrix[0][0].w, matrix[0][1].w, matrix[0][2].w );
      f_m[0][1] = Vec3<T>( matrix[1][0].w, matrix[1][1].w, matrix[1][2].w );
      f_m[1][1] = Vec3<T>( matrix[2][0].w, matrix[2][1].w, matrix[2][2].w );
      f_m[1][0] = Vec3<T>( matrix[3][0].w, matrix[3][1].w, matrix[3][2].w );
      return GregoryPatch3fa::normal_t(*(Vec3fa_4x4*)&matrix,f_m,uu,vv);
    }

    __forceinline void eval(const float u, const float v, 
                            Vec3fa* P, Vec3fa* dPdu, Vec3fa* dPdv, Vec3fa* ddPdudu, Vec3fa* ddPdvdv, Vec3fa* ddPdudv,
                            const float dscale = 1.0f) const
    {
      __aligned(64) Vec3fa f_m[2][2]; extract_f_m(f_m);
      if (P) {
        *P    = GregoryPatch3fa::eval(*(Vec3fa_4x4*)&matrix,f_m,u,v); 
      }
      if (dPdu) {
        assert(dPdu); *dPdu = GregoryPatch3fa::eval_du(*(Vec3fa_4x4*)&matrix,f_m,u,v)*dscale; 
        assert(dPdv); *dPdv = GregoryPatch3fa::eval_dv(*(Vec3fa_4x4*)&matrix,f_m,u,v)*dscale; 
      }
      if (ddPdudu) {
        assert(ddPdudu); *ddPdudu = GregoryPatch3fa::eval_dudu(*(Vec3fa_4x4*)&matrix,f_m,u,v)*sqr(dscale); 
        assert(ddPdvdv); *ddPdvdv = GregoryPatch3fa::eval_dvdv(*(Vec3fa_4x4*)&matrix,f_m,u,v)*sqr(dscale); 
        assert(ddPdudv); *ddPdudv = GregoryPatch3fa::eval_dudv(*(Vec3fa_4x4*)&matrix,f_m,u,v)*sqr(dscale); 
      }
    }

    template<typename vbool, typename vfloat>
    __forceinline void eval(const vbool& valid, const vfloat& uu, const vfloat& vv, float* P, float* dPdu, float* dPdv, const float dscale, const size_t dstride, const size_t N) const 
    {
      __aligned(64) Vec3fa f_m[2][2]; extract_f_m(f_m);
      GregoryPatch3fa::eval(matrix,f_m,valid,uu,vv,P,dPdu,dPdv,dscale,dstride,N);
    }

  private:
    Vec3ff matrix[4][4]; // f_p/m points are stored in 4th component
  };
}
