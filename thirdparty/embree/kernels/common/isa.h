// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../common/sys/platform.h"
#include "../../common/sys/sysinfo.h"

namespace embree
{
#define DEFINE_SYMBOL2(type,name)               \
  typedef type (*name##Func)();                 \
  name##Func name;
  
#define DECLARE_SYMBOL2(type,name)                                       \
  namespace sse2   { extern type name(); }                           \
  namespace sse42  { extern type name(); }                           \
  namespace avx    { extern type name(); }                           \
  namespace avx2   { extern type name(); }                           \
  namespace avx512 { extern type name(); }                           \
  void name##_error2() { throw_RTCError(RTC_ERROR_UNKNOWN,"internal error in ISA selection for " TOSTRING(name)); } \
  type name##_error() { return type(name##_error2); }                   \
  type name##_zero() { return type(nullptr); }

#define DECLARE_ISA_FUNCTION(type,symbol,args)                            \
  namespace sse2   { extern type symbol(args); }                       \
  namespace sse42  { extern type symbol(args); }                       \
  namespace avx    { extern type symbol(args); }                       \
  namespace avx2   { extern type symbol(args); }                       \
  namespace avx512 { extern type symbol(args); }                     \
  inline type symbol##_error(args) { throw_RTCError(RTC_ERROR_UNSUPPORTED_CPU,"function " TOSTRING(symbol) " not supported by your CPU"); } \
  typedef type (*symbol##Ty)(args);                                       \
  
#define DEFINE_ISA_FUNCTION(type,symbol,args)   \
  typedef type (*symbol##Func)(args);           \
  symbol##Func symbol;
  
#define ZERO_SYMBOL(features,intersector)                      \
  intersector = intersector##_zero;

#define INIT_SYMBOL(features,intersector)                      \
  intersector = decltype(intersector)(intersector##_error);

#define SELECT_SYMBOL_DEFAULT(features,intersector) \
  intersector = isa::intersector;

#if defined(__SSE__) || defined(__ARM_NEON)
#if !defined(EMBREE_TARGET_SIMD4)
#define EMBREE_TARGET_SIMD4
#endif
#endif

#if defined(EMBREE_TARGET_SSE42)
#define SELECT_SYMBOL_SSE42(features,intersector) \
  if ((features & SSE42) == SSE42) intersector = sse42::intersector;
#else
#define SELECT_SYMBOL_SSE42(features,intersector)
#endif

#if defined(EMBREE_TARGET_AVX) || defined(__AVX__)
#if !defined(EMBREE_TARGET_SIMD8)
#define EMBREE_TARGET_SIMD8
#endif
#if defined(__AVX__) // if default ISA is >= AVX we treat AVX target as default target
#define SELECT_SYMBOL_AVX(features,intersector)                 \
  if ((features & ISA) == ISA) intersector = isa::intersector;
#else
#define SELECT_SYMBOL_AVX(features,intersector)                 \
  if ((features & AVX) == AVX) intersector = avx::intersector;
#endif
#else
#define SELECT_SYMBOL_AVX(features,intersector)
#endif

#if defined(EMBREE_TARGET_AVX2)
#if !defined(EMBREE_TARGET_SIMD8)
#define EMBREE_TARGET_SIMD8
#endif
#define SELECT_SYMBOL_AVX2(features,intersector) \
  if ((features & AVX2) == AVX2) intersector = avx2::intersector;
#else
#define SELECT_SYMBOL_AVX2(features,intersector)
#endif

#if defined(EMBREE_TARGET_AVX512)
#if !defined(EMBREE_TARGET_SIMD16)
#define EMBREE_TARGET_SIMD16
#endif
#define SELECT_SYMBOL_AVX512(features,intersector) \
  if ((features & AVX512) == AVX512) intersector = avx512::intersector;
#else
#define SELECT_SYMBOL_AVX512(features,intersector)
#endif

#define SELECT_SYMBOL_DEFAULT_SSE42(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);            \
  SELECT_SYMBOL_SSE42(features,intersector);                                  
  
#define SELECT_SYMBOL_DEFAULT_SSE42_AVX(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                \
  SELECT_SYMBOL_SSE42(features,intersector);                  \
  SELECT_SYMBOL_AVX(features,intersector);                        
  
#define SELECT_SYMBOL_DEFAULT_SSE42_AVX_AVX2(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                     \
  SELECT_SYMBOL_SSE42(features,intersector);                       \
  SELECT_SYMBOL_AVX(features,intersector);                         \
  SELECT_SYMBOL_AVX2(features,intersector);                       

#define SELECT_SYMBOL_DEFAULT_SSE42_AVX_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                          \
  SELECT_SYMBOL_SSE42(features,intersector);                            \
  SELECT_SYMBOL_AVX(features,intersector);                              \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_AVX_AVX2_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                                   \
  SELECT_SYMBOL_AVX(features,intersector);                                       \
  SELECT_SYMBOL_AVX2(features,intersector);                                      \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_AVX_AVX2_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                         \
  SELECT_SYMBOL_AVX(features,intersector);                             \
  SELECT_SYMBOL_AVX2(features,intersector);                            \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_SSE42_AVX_AVX2_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                                         \
  SELECT_SYMBOL_SSE42(features,intersector);                                           \
  SELECT_SYMBOL_AVX(features,intersector);                                             \
  SELECT_SYMBOL_AVX2(features,intersector);                                            \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_SSE42_AVX_AVX2_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                               \
  SELECT_SYMBOL_SSE42(features,intersector);                                 \
  SELECT_SYMBOL_AVX(features,intersector);                                   \
  SELECT_SYMBOL_AVX2(features,intersector);                                  \
  SELECT_SYMBOL_AVX512(features,intersector);
  
#define SELECT_SYMBOL_DEFAULT_AVX(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);          \
  SELECT_SYMBOL_AVX(features,intersector);                        
  
#define SELECT_SYMBOL_DEFAULT_AVX_AVX2(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);               \
  SELECT_SYMBOL_AVX(features,intersector);                   \
  SELECT_SYMBOL_AVX2(features,intersector);                       
  
#define SELECT_SYMBOL_DEFAULT_AVX(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                    \
  SELECT_SYMBOL_AVX(features,intersector);
  
#define SELECT_SYMBOL_DEFAULT_AVX_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                              \
  SELECT_SYMBOL_AVX(features,intersector);                                  \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_AVX_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                    \
  SELECT_SYMBOL_AVX(features,intersector);                        \
  SELECT_SYMBOL_AVX512(features,intersector);
  
#define SELECT_SYMBOL_INIT_AVX(features,intersector) \
  INIT_SYMBOL(features,intersector);                 \
  SELECT_SYMBOL_AVX(features,intersector);                                
  
#define SELECT_SYMBOL_INIT_AVX_AVX2(features,intersector) \
  INIT_SYMBOL(features,intersector);                      \
  SELECT_SYMBOL_AVX(features,intersector);                \
  SELECT_SYMBOL_AVX2(features,intersector);

#define SELECT_SYMBOL_INIT_AVX_AVX2_AVX512(features,intersector) \
  INIT_SYMBOL(features,intersector);                                \
  SELECT_SYMBOL_AVX(features,intersector);                          \
  SELECT_SYMBOL_AVX2(features,intersector);                         \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_INIT_SSE42_AVX_AVX2(features,intersector) \
  INIT_SYMBOL(features,intersector);                            \
  SELECT_SYMBOL_SSE42(features,intersector);                    \
  SELECT_SYMBOL_AVX(features,intersector);                      \
  SELECT_SYMBOL_AVX2(features,intersector);
  
#define SELECT_SYMBOL_INIT_AVX(features,intersector) \
  INIT_SYMBOL(features,intersector);                           \
  SELECT_SYMBOL_AVX(features,intersector);

#define SELECT_SYMBOL_INIT_AVX_AVX512(features,intersector) \
  INIT_SYMBOL(features,intersector);                                     \
  SELECT_SYMBOL_AVX(features,intersector);                               \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_INIT_AVX_AVX2(features,intersector) \
  INIT_SYMBOL(features,intersector);                                \
  SELECT_SYMBOL_AVX(features,intersector);                          \
  SELECT_SYMBOL_AVX2(features,intersector);

#define SELECT_SYMBOL_INIT_AVX_AVX2_AVX512(features,intersector) \
  INIT_SYMBOL(features,intersector);                                          \
  SELECT_SYMBOL_AVX(features,intersector);                                    \
  SELECT_SYMBOL_AVX2(features,intersector);                                   \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_INIT_SSE42_AVX_AVX2_AVX512(features,intersector) \
  INIT_SYMBOL(features,intersector);                                                \
  SELECT_SYMBOL_SSE42(features,intersector);                                        \
  SELECT_SYMBOL_AVX(features,intersector);                                          \
  SELECT_SYMBOL_AVX2(features,intersector);                                         \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_ZERO_SSE42_AVX_AVX2_AVX512(features,intersector) \
  ZERO_SYMBOL(features,intersector);                                    \
  SELECT_SYMBOL_SSE42(features,intersector);                            \
  SELECT_SYMBOL_AVX(features,intersector);                              \
  SELECT_SYMBOL_AVX2(features,intersector);                             \
  SELECT_SYMBOL_AVX512(features,intersector);

#define SELECT_SYMBOL_DEFAULT_AVX_AVX2_AVX512(features,intersector) \
  SELECT_SYMBOL_DEFAULT(features,intersector);                                   \
  SELECT_SYMBOL_AVX(features,intersector);                                       \
  SELECT_SYMBOL_AVX2(features,intersector);                                      \
  SELECT_SYMBOL_AVX512(features,intersector);
  
#define SELECT_SYMBOL_INIT_AVX512(features,intersector) \
  INIT_SYMBOL(features,intersector);                                 \
  SELECT_SYMBOL_AVX512(features,intersector);
  
#define SELECT_SYMBOL_SSE42_AVX_AVX2(features,intersector) \
  SELECT_SYMBOL_SSE42(features,intersector);               \
  SELECT_SYMBOL_AVX(features,intersector);                 \
  SELECT_SYMBOL_AVX2(features,intersector);

  struct VerifyMultiTargetLinking {
    static __noinline int getISA(int depth = 5) { 
      if (depth == 0) return ISA; 
      else return getISA(depth-1); 
    }
  };
  namespace sse2   { int getISA(); };
  namespace sse42  { int getISA(); };
  namespace avx    { int getISA(); };
  namespace avx2   { int getISA(); };
  namespace avx512 { int getISA(); };
}
