/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_PORTS_MEM_OPS_H_
#define VPX_VPX_PORTS_MEM_OPS_H_

/* \file
 * \brief Provides portable memory access primitives
 *
 * This function provides portable primitives for getting and setting of
 * signed and unsigned integers in 16, 24, and 32 bit sizes. The operations
 * can be performed on unaligned data regardless of hardware support for
 * unaligned accesses.
 *
 * The type used to pass the integral values may be changed by defining
 * MEM_VALUE_T with the appropriate type. The type given must be an integral
 * numeric type.
 *
 * The actual functions instantiated have the MEM_VALUE_T type name pasted
 * on to the symbol name. This allows the developer to instantiate these
 * operations for multiple types within the same translation unit. This is
 * of somewhat questionable utility, but the capability exists nonetheless.
 * Users not making use of this functionality should call the functions
 * without the type name appended, and the preprocessor will take care of
 * it.
 *
 * NOTE: This code is not supported on platforms where char > 1 octet ATM.
 */

#ifndef MAU_T
/* Minimum Access Unit for this target */
#define MAU_T unsigned char
#endif

#ifndef MEM_VALUE_T
#define MEM_VALUE_T int
#endif

#undef MEM_VALUE_T_SZ_BITS
#define MEM_VALUE_T_SZ_BITS (sizeof(MEM_VALUE_T) << 3)

#undef mem_ops_wrap_symbol
#define mem_ops_wrap_symbol(fn) mem_ops_wrap_symbol2(fn, MEM_VALUE_T)
#undef mem_ops_wrap_symbol2
#define mem_ops_wrap_symbol2(fn, typ) mem_ops_wrap_symbol3(fn, typ)
#undef mem_ops_wrap_symbol3
#define mem_ops_wrap_symbol3(fn, typ) fn##_as_##typ

/*
 * Include aligned access routines
 */
#define INCLUDED_BY_MEM_OPS_H
#include "mem_ops_aligned.h"
#undef INCLUDED_BY_MEM_OPS_H

#undef mem_get_be16
#define mem_get_be16 mem_ops_wrap_symbol(mem_get_be16)
static unsigned MEM_VALUE_T mem_get_be16(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = mem[0] << 8;
  val |= mem[1];
  return val;
}

#undef mem_get_be24
#define mem_get_be24 mem_ops_wrap_symbol(mem_get_be24)
static unsigned MEM_VALUE_T mem_get_be24(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = mem[0] << 16;
  val |= mem[1] << 8;
  val |= mem[2];
  return val;
}

#undef mem_get_be32
#define mem_get_be32 mem_ops_wrap_symbol(mem_get_be32)
static unsigned MEM_VALUE_T mem_get_be32(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = ((unsigned MEM_VALUE_T)mem[0]) << 24;
  val |= mem[1] << 16;
  val |= mem[2] << 8;
  val |= mem[3];
  return val;
}

#undef mem_get_le16
#define mem_get_le16 mem_ops_wrap_symbol(mem_get_le16)
static unsigned MEM_VALUE_T mem_get_le16(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = mem[1] << 8;
  val |= mem[0];
  return val;
}

#undef mem_get_le24
#define mem_get_le24 mem_ops_wrap_symbol(mem_get_le24)
static unsigned MEM_VALUE_T mem_get_le24(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = mem[2] << 16;
  val |= mem[1] << 8;
  val |= mem[0];
  return val;
}

#undef mem_get_le32
#define mem_get_le32 mem_ops_wrap_symbol(mem_get_le32)
static unsigned MEM_VALUE_T mem_get_le32(const void *vmem) {
  unsigned MEM_VALUE_T val;
  const MAU_T *mem = (const MAU_T *)vmem;

  val = ((unsigned MEM_VALUE_T)mem[3]) << 24;
  val |= mem[2] << 16;
  val |= mem[1] << 8;
  val |= mem[0];
  return val;
}

#define mem_get_s_generic(end, sz)                                            \
  static VPX_INLINE signed MEM_VALUE_T mem_get_s##end##sz(const void *vmem) { \
    const MAU_T *mem = (const MAU_T *)vmem;                                   \
    signed MEM_VALUE_T val = mem_get_##end##sz(mem);                          \
    return (val << (MEM_VALUE_T_SZ_BITS - sz)) >> (MEM_VALUE_T_SZ_BITS - sz); \
  }

/* clang-format off */
#undef  mem_get_sbe16
#define mem_get_sbe16 mem_ops_wrap_symbol(mem_get_sbe16)
mem_get_s_generic(be, 16)

#undef  mem_get_sbe24
#define mem_get_sbe24 mem_ops_wrap_symbol(mem_get_sbe24)
mem_get_s_generic(be, 24)

#undef  mem_get_sbe32
#define mem_get_sbe32 mem_ops_wrap_symbol(mem_get_sbe32)
mem_get_s_generic(be, 32)

#undef  mem_get_sle16
#define mem_get_sle16 mem_ops_wrap_symbol(mem_get_sle16)
mem_get_s_generic(le, 16)

#undef  mem_get_sle24
#define mem_get_sle24 mem_ops_wrap_symbol(mem_get_sle24)
mem_get_s_generic(le, 24)

#undef  mem_get_sle32
#define mem_get_sle32 mem_ops_wrap_symbol(mem_get_sle32)
mem_get_s_generic(le, 32)

#undef  mem_put_be16
#define mem_put_be16 mem_ops_wrap_symbol(mem_put_be16)
static VPX_INLINE void mem_put_be16(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >> 8) & 0xff);
  mem[1] = (MAU_T)((val >> 0) & 0xff);
}

#undef  mem_put_be24
#define mem_put_be24 mem_ops_wrap_symbol(mem_put_be24)
static VPX_INLINE void mem_put_be24(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >> 16) & 0xff);
  mem[1] = (MAU_T)((val >>  8) & 0xff);
  mem[2] = (MAU_T)((val >>  0) & 0xff);
}

#undef  mem_put_be32
#define mem_put_be32 mem_ops_wrap_symbol(mem_put_be32)
static VPX_INLINE void mem_put_be32(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >> 24) & 0xff);
  mem[1] = (MAU_T)((val >> 16) & 0xff);
  mem[2] = (MAU_T)((val >>  8) & 0xff);
  mem[3] = (MAU_T)((val >>  0) & 0xff);
}

#undef  mem_put_le16
#define mem_put_le16 mem_ops_wrap_symbol(mem_put_le16)
static VPX_INLINE void mem_put_le16(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >> 0) & 0xff);
  mem[1] = (MAU_T)((val >> 8) & 0xff);
}

#undef  mem_put_le24
#define mem_put_le24 mem_ops_wrap_symbol(mem_put_le24)
static VPX_INLINE void mem_put_le24(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >>  0) & 0xff);
  mem[1] = (MAU_T)((val >>  8) & 0xff);
  mem[2] = (MAU_T)((val >> 16) & 0xff);
}

#undef  mem_put_le32
#define mem_put_le32 mem_ops_wrap_symbol(mem_put_le32)
static VPX_INLINE void mem_put_le32(void *vmem, MEM_VALUE_T val) {
  MAU_T *mem = (MAU_T *)vmem;

  mem[0] = (MAU_T)((val >>  0) & 0xff);
  mem[1] = (MAU_T)((val >>  8) & 0xff);
  mem[2] = (MAU_T)((val >> 16) & 0xff);
  mem[3] = (MAU_T)((val >> 24) & 0xff);
}
/* clang-format on */
#endif  // VPX_VPX_PORTS_MEM_OPS_H_
