/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* adapted from FreeBSD:
 *    lib/msun/src/math_private.h
 */
#pragma once

#include <thrust/detail/config.h>
#include <thrust/complex.h>
#include <thrust/detail/cstdint.h>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{

using thrust::complex;

typedef union
{
  float value;
  uint32_t word;
} ieee_float_shape_type;
  
__host__ __device__
inline void get_float_word(uint32_t & i, float d){
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__
inline void get_float_word(int32_t & i, float d){
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__
inline void set_float_word(float & d, uint32_t i){
  ieee_float_shape_type sf_u;
  sf_u.word = (i);
  (d) = sf_u.value;
}

// Assumes little endian ordering
typedef union
{
  double value;
  struct
  {
    uint32_t lsw;
    uint32_t msw;
  } parts;
  struct
  {
    uint64_t w;
  } xparts;
} ieee_double_shape_type;
  
__host__ __device__ inline
void get_high_word(uint32_t & i,double d){
  ieee_double_shape_type gh_u;
  gh_u.value = (d);
  (i) = gh_u.parts.msw;                                   
}
  
/* Set the more significant 32 bits of a double from an int.  */
__host__ __device__ inline
void set_high_word(double & d, uint32_t v){
  ieee_double_shape_type sh_u;
  sh_u.value = (d);
  sh_u.parts.msw = (v);
  (d) = sh_u.value;
}
  
  
__host__ __device__ inline 
void  insert_words(double & d, uint32_t ix0, uint32_t ix1){
  ieee_double_shape_type iw_u;
  iw_u.parts.msw = (ix0);
  iw_u.parts.lsw = (ix1);
  (d) = iw_u.value;
}
  
/* Get two 32 bit ints from a double.  */
__host__ __device__ inline
void  extract_words(uint32_t & ix0,uint32_t & ix1, double d){
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}
  
/* Get two 32 bit ints from a double.  */
__host__ __device__ inline
void  extract_words(int32_t & ix0,int32_t & ix1, double d){
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}
  
} // namespace complex

} // namespace detail

THRUST_NAMESPACE_END


#include <thrust/detail/complex/c99math.h>
