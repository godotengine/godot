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

#if !defined(_x86_x86enc_H)
# define _x86_x86enc_H (1)
# include "x86int.h"

# if defined(OC_X86_ASM)
#  define oc_enc_accel_init oc_enc_accel_init_x86
#  if defined(OC_X86_64_ASM)
/*x86-64 guarantees SIMD support up through at least SSE2.
  If the best routine we have available only needs SSE2 (which at the moment
   covers all of them), then we can avoid runtime detection and the indirect
   call.*/
#   define oc_enc_frag_sub(_enc,_diff,_x,_y,_stride) \
  oc_enc_frag_sub_mmx(_diff,_x,_y,_stride)
#   define oc_enc_frag_sub_128(_enc,_diff,_x,_stride) \
  oc_enc_frag_sub_128_mmx(_diff,_x,_stride)
#   define oc_enc_frag_sad(_enc,_src,_ref,_ystride) \
  oc_enc_frag_sad_mmxext(_src,_ref,_ystride)
#   define oc_enc_frag_sad_thresh(_enc,_src,_ref,_ystride,_thresh) \
  oc_enc_frag_sad_thresh_mmxext(_src,_ref,_ystride,_thresh)
#   define oc_enc_frag_sad2_thresh(_enc,_src,_ref1,_ref2,_ystride,_thresh) \
  oc_enc_frag_sad2_thresh_mmxext(_src,_ref1,_ref2,_ystride,_thresh)
#   define oc_enc_frag_satd(_enc,_dc,_src,_ref,_ystride) \
  oc_enc_frag_satd_sse2(_dc,_src,_ref,_ystride)
#   define oc_enc_frag_satd2(_enc,_dc,_src,_ref1,_ref2,_ystride) \
  oc_enc_frag_satd2_sse2(_dc,_src,_ref1,_ref2,_ystride)
#   define oc_enc_frag_intra_satd(_enc,_dc,_src,_ystride) \
  oc_enc_frag_intra_satd_sse2(_dc,_src,_ystride)
#   define oc_enc_frag_ssd(_enc,_src,_ref,_ystride) \
  oc_enc_frag_ssd_sse2(_src,_ref,_ystride)
#   define oc_enc_frag_border_ssd(_enc,_src,_ref,_ystride,_mask) \
  oc_enc_frag_border_ssd_sse2(_src,_ref,_ystride,_mask)
#   define oc_enc_frag_copy2(_enc,_dst,_src1,_src2,_ystride) \
  oc_int_frag_copy2_mmxext(_dst,_ystride,_src1,_src2,_ystride)
#   define oc_enc_enquant_table_init(_enc,_enquant,_dequant) \
  oc_enc_enquant_table_init_x86(_enquant,_dequant)
#   define oc_enc_enquant_table_fixup(_enc,_enquant,_nqis) \
  oc_enc_enquant_table_fixup_x86(_enquant,_nqis)
#  define oc_enc_quantize(_enc,_qdct,_dct,_dequant,_enquant) \
  oc_enc_quantize_sse2(_qdct,_dct,_dequant,_enquant)
#   define oc_enc_frag_recon_intra(_enc,_dst,_ystride,_residue) \
  oc_frag_recon_intra_mmx(_dst,_ystride,_residue)
#   define oc_enc_frag_recon_inter(_enc,_dst,_src,_ystride,_residue) \
  oc_frag_recon_inter_mmx(_dst,_src,_ystride,_residue)
#   define oc_enc_fdct8x8(_enc,_y,_x) \
  oc_enc_fdct8x8_x86_64sse2(_y,_x)
#  else
#   define OC_ENC_USE_VTABLE (1)
#  endif
# endif

# include "../encint.h"

void oc_enc_accel_init_x86(oc_enc_ctx *_enc);

void oc_enc_frag_sub_mmx(ogg_int16_t _diff[64],
 const unsigned char *_x,const unsigned char *_y,int _stride);
void oc_enc_frag_sub_128_mmx(ogg_int16_t _diff[64],
 const unsigned char *_x,int _stride);
unsigned oc_enc_frag_sad_mmxext(const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_sad_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _thresh);
unsigned oc_enc_frag_sad2_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
 unsigned _thresh);
unsigned oc_enc_frag_satd_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_satd_sse2(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_satd2_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride);
unsigned oc_enc_frag_satd2_sse2(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride);
unsigned oc_enc_frag_intra_satd_mmxext(int *_dc,
 const unsigned char *_src,int _ystride);
unsigned oc_enc_frag_intra_satd_sse2(int *_dc,
 const unsigned char *_src,int _ystride);
unsigned oc_enc_frag_ssd_sse2(const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_border_ssd_sse2(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,ogg_int64_t _mask);
void oc_int_frag_copy2_mmxext(unsigned char *_dst,int _dst_ystride,
 const unsigned char *_src1,const unsigned char *_src2,int _src_ystride);
void oc_enc_frag_copy2_mmxext(unsigned char *_dst,
 const unsigned char *_src1,const unsigned char *_src2,int _ystride);
void oc_enc_enquant_table_init_x86(void *_enquant,
 const ogg_uint16_t _dequant[64]);
void oc_enc_enquant_table_fixup_x86(void *_enquant[3][3][2],int _nqis);
int oc_enc_quantize_sse2(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
 const ogg_uint16_t _dequant[64],const void *_enquant);
void oc_enc_fdct8x8_mmxext(ogg_int16_t _y[64],const ogg_int16_t _x[64]);

# if defined(OC_X86_64_ASM)
void oc_enc_fdct8x8_x86_64sse2(ogg_int16_t _y[64],const ogg_int16_t _x[64]);
# endif

#endif
