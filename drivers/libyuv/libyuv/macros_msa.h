/*
 *  Copyright 2016 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef __MACROS_MSA_H__
#define __MACROS_MSA_H__

#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#include <stdint.h>
#include <msa.h>

#define LD_B(RTYPE, psrc) *((RTYPE*)(psrc))
#define LD_UB(...) LD_B(v16u8, __VA_ARGS__)

#define ST_B(RTYPE, in, pdst) *((RTYPE*)(pdst)) = (in)
#define ST_UB(...) ST_B(v16u8, __VA_ARGS__)

/* Description : Load two vectors with 16 'byte' sized elements
   Arguments   : Inputs  - psrc, stride
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Load 16 byte elements in 'out0' from (psrc)
                 Load 16 byte elements in 'out1' from (psrc + stride)
*/
#define LD_B2(RTYPE, psrc, stride, out0, out1) {  \
  out0 = LD_B(RTYPE, (psrc));                     \
  out1 = LD_B(RTYPE, (psrc) + stride);            \
}
#define LD_UB2(...) LD_B2(v16u8, __VA_ARGS__)
#define LD_SB2(...) LD_B2(v16i8, __VA_ARGS__)

#define LD_B4(RTYPE, psrc, stride, out0, out1, out2, out3) {  \
  LD_B2(RTYPE, (psrc), stride, out0, out1);                   \
  LD_B2(RTYPE, (psrc) + 2 * stride , stride, out2, out3);     \
}
#define LD_UB4(...) LD_B4(v16u8, __VA_ARGS__)
#define LD_SB4(...) LD_B4(v16i8, __VA_ARGS__)

/* Description : Store two vectors with stride each having 16 'byte' sized
                 elements
   Arguments   : Inputs - in0, in1, pdst, stride
   Details     : Store 16 byte elements from 'in0' to (pdst)
                 Store 16 byte elements from 'in1' to (pdst + stride)
*/
#define ST_B2(RTYPE, in0, in1, pdst, stride) {  \
  ST_B(RTYPE, in0, (pdst));                     \
  ST_B(RTYPE, in1, (pdst) + stride);            \
}
#define ST_UB2(...) ST_B2(v16u8, __VA_ARGS__)
#define ST_SB2(...) ST_B2(v16i8, __VA_ARGS__)

#define ST_B4(RTYPE, in0, in1, in2, in3, pdst, stride) {  \
  ST_B2(RTYPE, in0, in1, (pdst), stride);                 \
  ST_B2(RTYPE, in2, in3, (pdst) + 2 * stride, stride);    \
}
#define ST_UB4(...) ST_B4(v16u8, __VA_ARGS__)
#define ST_SB4(...) ST_B4(v16i8, __VA_ARGS__)

/* Description : Shuffle byte vector elements as per mask vector
   Arguments   : Inputs  - in0, in1, in2, in3, mask0, mask1
                 Outputs - out0, out1
                 Return Type - as per RTYPE
   Details     : Byte elements from 'in0' & 'in1' are copied selectively to
                 'out0' as per control vector 'mask0'
*/
#define VSHF_B2(RTYPE, in0, in1, in2, in3, mask0, mask1, out0, out1) {   \
  out0 = (RTYPE) __msa_vshf_b((v16i8) mask0, (v16i8) in1, (v16i8) in0);  \
  out1 = (RTYPE) __msa_vshf_b((v16i8) mask1, (v16i8) in3, (v16i8) in2);  \
}
#define VSHF_B2_UB(...) VSHF_B2(v16u8, __VA_ARGS__)
#endif  /* !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa) */
#endif  /* __MACROS_MSA_H__ */
