/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_
#define VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_

#if defined(__GNUC__) && __GNUC__ < 4
/* From emmintrin.h (gcc 4.5.3) */
/* Casts between various SP, DP, INT vector types.  Note that these do no
   conversion of values, they just change the type.  */
extern __inline __m128
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castpd_ps(__m128d __A) {
  return (__m128)__A;
}

extern __inline __m128i
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castpd_si128(__m128d __A) {
  return (__m128i)__A;
}

extern __inline __m128d
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castps_pd(__m128 __A) {
  return (__m128d)__A;
}

extern __inline __m128i
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castps_si128(__m128 __A) {
  return (__m128i)__A;
}

extern __inline __m128
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castsi128_ps(__m128i __A) {
  return (__m128)__A;
}

extern __inline __m128d
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castsi128_pd(__m128i __A) {
  return (__m128d)__A;
}
#endif

#endif  // VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_
