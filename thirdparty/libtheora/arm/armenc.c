/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2010                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: x86state.c 17344 2010-07-21 01:42:18Z tterribe $

 ********************************************************************/
#include "armenc.h"

#if defined(OC_ARM_ASM)

void oc_enc_accel_init_arm(oc_enc_ctx *_enc){
  ogg_uint32_t cpu_flags;
  cpu_flags=_enc->state.cpu_flags;
  oc_enc_accel_init_c(_enc);
# if defined(OC_ENC_USE_VTABLE)
  /*TODO: Add ARMv4 functions here.*/
# endif
# if defined(OC_ARM_ASM_EDSP)
  if(cpu_flags&OC_CPU_ARM_EDSP){
#  if defined(OC_STATE_USE_VTABLE)
    /*TODO: Add EDSP functions here.*/
#  endif
  }
#  if defined(OC_ARM_ASM_MEDIA)
  if(cpu_flags&OC_CPU_ARM_MEDIA){
#   if defined(OC_STATE_USE_VTABLE)
    /*TODO: Add Media functions here.*/
#   endif
  }
#   if defined(OC_ARM_ASM_NEON)
  if(cpu_flags&OC_CPU_ARM_NEON){
#    if defined(OC_STATE_USE_VTABLE)
    _enc->opt_vtable.frag_satd=oc_enc_frag_satd_neon;
    _enc->opt_vtable.frag_satd2=oc_enc_frag_satd2_neon;
    _enc->opt_vtable.frag_intra_satd=oc_enc_frag_intra_satd_neon;
    _enc->opt_vtable.enquant_table_init=oc_enc_enquant_table_init_neon;
    _enc->opt_vtable.enquant_table_fixup=oc_enc_enquant_table_fixup_neon;
    _enc->opt_vtable.quantize=oc_enc_quantize_neon;
#    endif
    _enc->opt_data.enquant_table_size=128*sizeof(ogg_uint16_t);
    _enc->opt_data.enquant_table_alignment=16;
  }
#   endif
#  endif
# endif
}
#endif
