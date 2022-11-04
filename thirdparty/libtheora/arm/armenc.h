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
    last mod: $Id: x86int.h 17344 2010-07-21 01:42:18Z tterribe $

 ********************************************************************/
#if !defined(_arm_armenc_H)
# define _arm_armenc_H (1)
# include "armint.h"

# if defined(OC_ARM_ASM)
#  define oc_enc_accel_init oc_enc_accel_init_arm
#  define OC_ENC_USE_VTABLE (1)
# endif

# include "../encint.h"

# if defined(OC_ARM_ASM)
void oc_enc_accel_init_arm(oc_enc_ctx *_enc);

#  if defined(OC_ARM_ASM_EDSP)
#   if defined(OC_ARM_ASM_MEDIA)
#    if defined(OC_ARM_ASM_NEON)
unsigned oc_enc_frag_satd_neon(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride);
unsigned oc_enc_frag_satd2_neon(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride);
unsigned oc_enc_frag_intra_satd_neon(int *_dc,
 const unsigned char *_src,int _ystride);

void oc_enc_enquant_table_init_neon(void *_enquant,
 const ogg_uint16_t _dequant[64]);
void oc_enc_enquant_table_fixup_neon(void *_enquant[3][3][2],int _nqis);
int oc_enc_quantize_neon(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
 const ogg_uint16_t _dequant[64],const void *_enquant);
#    endif
#   endif
#  endif
# endif

#endif
