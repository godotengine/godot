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
#include "x86enc.h"

#if defined(OC_X86_ASM)

void oc_enc_accel_init_x86(oc_enc_ctx *_enc){
  ogg_uint32_t cpu_flags;
  cpu_flags=_enc->state.cpu_flags;
  oc_enc_accel_init_c(_enc);
# if defined(OC_ENC_USE_VTABLE)
  if(cpu_flags&OC_CPU_X86_MMX){
    _enc->opt_vtable.frag_sub=oc_enc_frag_sub_mmx;
    _enc->opt_vtable.frag_sub_128=oc_enc_frag_sub_128_mmx;
    _enc->opt_vtable.frag_recon_intra=oc_frag_recon_intra_mmx;
    _enc->opt_vtable.frag_recon_inter=oc_frag_recon_inter_mmx;
  }
  if(cpu_flags&OC_CPU_X86_MMXEXT){
    _enc->opt_vtable.frag_sad=oc_enc_frag_sad_mmxext;
    _enc->opt_vtable.frag_sad_thresh=oc_enc_frag_sad_thresh_mmxext;
    _enc->opt_vtable.frag_sad2_thresh=oc_enc_frag_sad2_thresh_mmxext;
    _enc->opt_vtable.frag_satd=oc_enc_frag_satd_mmxext;
    _enc->opt_vtable.frag_satd2=oc_enc_frag_satd2_mmxext;
    _enc->opt_vtable.frag_intra_satd=oc_enc_frag_intra_satd_mmxext;
    _enc->opt_vtable.frag_copy2=oc_enc_frag_copy2_mmxext;
    _enc->opt_vtable.fdct8x8=oc_enc_fdct8x8_mmxext;
  }
  if(cpu_flags&OC_CPU_X86_SSE2){
#  if defined(OC_X86_64_ASM)
    _enc->opt_vtable.fdct8x8=oc_enc_fdct8x8_x86_64sse2;
#  endif
    _enc->opt_vtable.frag_ssd=oc_enc_frag_ssd_sse2;
    _enc->opt_vtable.frag_border_ssd=oc_enc_frag_border_ssd_sse2;
    _enc->opt_vtable.frag_satd=oc_enc_frag_satd_sse2;
    _enc->opt_vtable.frag_satd2=oc_enc_frag_satd2_sse2;
    _enc->opt_vtable.frag_intra_satd=oc_enc_frag_intra_satd_sse2;
    _enc->opt_vtable.enquant_table_init=oc_enc_enquant_table_init_x86;
    _enc->opt_vtable.enquant_table_fixup=oc_enc_enquant_table_fixup_x86;
    _enc->opt_vtable.quantize=oc_enc_quantize_sse2;
# else
    (void) cpu_flags;
# endif
    _enc->opt_data.enquant_table_size=128*sizeof(ogg_uint16_t);
    _enc->opt_data.enquant_table_alignment=16;
# if defined(OC_ENC_USE_VTABLE)
  }
# endif
}
#endif
