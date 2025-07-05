/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/
#if !defined(_internal_H)
# define _internal_H (1)
# include <stdlib.h>
# include <limits.h>
# if defined(HAVE_CONFIG_H)
#  include "config.h"
# endif
# include "theora/codec.h"
# include "theora/theora.h"
# include "ocintrin.h"

# if !defined(__GNUC_PREREQ)
#  if defined(__GNUC__)&&defined(__GNUC_MINOR__)
#   define __GNUC_PREREQ(_maj,_min) \
 ((__GNUC__<<16)+__GNUC_MINOR__>=((_maj)<<16)+(_min))
#  else
#   define __GNUC_PREREQ(_maj,_min) 0
#  endif
# endif

# if defined(_MSC_VER)
/*Disable missing EMMS warnings.*/
#  pragma warning(disable:4799)
/*Thank you Microsoft, I know the order of operations.*/
#  pragma warning(disable:4554)
# endif
/*You, too, gcc.*/
# if __GNUC_PREREQ(4,2)
#  pragma GCC diagnostic ignored "-Wparentheses"
# endif
/*Et tu, clang.*/
# if defined(__clang__)
#  pragma clang diagnostic ignored "-Wparentheses"
# endif

/*Some assembly constructs require aligned operands.
  The following macros are _only_ intended for structure member declarations.
  Although they will sometimes work on stack variables, gcc will often silently
   ignore them.
  A separate set of macros could be made for manual stack alignment, but we
   don't actually require it anywhere.*/
# if defined(OC_X86_ASM)||defined(OC_ARM_ASM)
#  if defined(__GNUC__)
#   define OC_ALIGN8(expr) expr __attribute__((aligned(8)))
#   define OC_ALIGN16(expr) expr __attribute__((aligned(16)))
#  elif defined(_MSC_VER)
#   define OC_ALIGN8(expr) __declspec (align(8)) expr
#   define OC_ALIGN16(expr) __declspec (align(16)) expr
#  else
#   error "Alignment macros required for this platform."
#  endif
# endif
# if !defined(OC_ALIGN8)
#  define OC_ALIGN8(expr) expr
# endif
# if !defined(OC_ALIGN16)
#  define OC_ALIGN16(expr) expr
# endif



/*This library's version.*/
# define OC_VENDOR_STRING "Xiph.Org libtheora 1.2.0 20250329 (Ptalarbvorm)"

/*Theora bitstream version.*/
# define TH_VERSION_MAJOR (3)
# define TH_VERSION_MINOR (2)
# define TH_VERSION_SUB   (1)
# define TH_VERSION_CHECK(_info,_maj,_min,_sub) \
 ((_info)->version_major>(_maj)||(_info)->version_major==(_maj)&& \
 ((_info)->version_minor>(_min)||(_info)->version_minor==(_min)&& \
 (_info)->version_subminor>=(_sub)))



/*A map from the index in the zig zag scan to the coefficient number in a
   block.*/
extern const unsigned char OC_FZIG_ZAG[128];
/*A map from the coefficient number in a block to its index in the zig zag
   scan.*/
extern const unsigned char OC_IZIG_ZAG[64];
/*A map from physical macro block ordering to bitstream macro block
   ordering within a super block.*/
extern const unsigned char OC_MB_MAP[2][2];
/*A list of the indices in the oc_mb_map array that can be valid for each of
   the various chroma decimation types.*/
extern const unsigned char OC_MB_MAP_IDXS[TH_PF_NFORMATS][12];
/*The number of indices in the oc_mb_map array that can be valid for each of
   the various chroma decimation types.*/
extern const unsigned char OC_MB_MAP_NIDXS[TH_PF_NFORMATS];



int oc_ilog(unsigned _v);
void *oc_aligned_malloc(size_t _sz,size_t _align);
void oc_aligned_free(void *_ptr);
void **oc_malloc_2d(size_t _height,size_t _width,size_t _sz);
void **oc_calloc_2d(size_t _height,size_t _width,size_t _sz);
void oc_free_2d(void *_ptr);

void oc_ycbcr_buffer_flip(th_ycbcr_buffer _dst,
 const th_ycbcr_buffer _src);

#endif
