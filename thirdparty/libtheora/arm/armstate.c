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
#include "armint.h"

#if defined(OC_ARM_ASM)

# if defined(OC_ARM_ASM_NEON)
/*This table has been modified from OC_FZIG_ZAG by baking an 8x8 transpose into
   the destination.*/
static const unsigned char OC_FZIG_ZAG_NEON[128]={
   0, 8, 1, 2, 9,16,24,17,
  10, 3, 4,11,18,25,32,40,
  33,26,19,12, 5, 6,13,20,
  27,34,41,48,56,49,42,35,
  28,21,14, 7,15,22,29,36,
  43,50,57,58,51,44,37,30,
  23,31,38,45,52,59,60,53,
  46,39,47,54,61,62,55,63,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64
};
# endif

void oc_state_accel_init_arm(oc_theora_state *_state){
  oc_state_accel_init_c(_state);
  _state->cpu_flags=oc_cpu_flags_get();
# if defined(OC_STATE_USE_VTABLE)
  _state->opt_vtable.frag_copy_list=oc_frag_copy_list_arm;
  _state->opt_vtable.frag_recon_intra=oc_frag_recon_intra_arm;
  _state->opt_vtable.frag_recon_inter=oc_frag_recon_inter_arm;
  _state->opt_vtable.frag_recon_inter2=oc_frag_recon_inter2_arm;
  _state->opt_vtable.idct8x8=oc_idct8x8_arm;
  _state->opt_vtable.state_frag_recon=oc_state_frag_recon_arm;
  /*Note: We _must_ set this function pointer, because the macro in armint.h
     calls it with different arguments, so the C version will segfault.*/
  _state->opt_vtable.state_loop_filter_frag_rows=
   (oc_state_loop_filter_frag_rows_func)oc_loop_filter_frag_rows_arm;
# endif
# if defined(OC_ARM_ASM_EDSP)
  if(_state->cpu_flags&OC_CPU_ARM_EDSP){
#  if defined(OC_STATE_USE_VTABLE)
    _state->opt_vtable.frag_copy_list=oc_frag_copy_list_edsp;
#  endif
  }
#  if defined(OC_ARM_ASM_MEDIA)
  if(_state->cpu_flags&OC_CPU_ARM_MEDIA){
#   if defined(OC_STATE_USE_VTABLE)
    _state->opt_vtable.frag_recon_intra=oc_frag_recon_intra_v6;
    _state->opt_vtable.frag_recon_inter=oc_frag_recon_inter_v6;
    _state->opt_vtable.frag_recon_inter2=oc_frag_recon_inter2_v6;
    _state->opt_vtable.idct8x8=oc_idct8x8_v6;
    _state->opt_vtable.state_frag_recon=oc_state_frag_recon_v6;
    _state->opt_vtable.loop_filter_init=oc_loop_filter_init_v6;
    _state->opt_vtable.state_loop_filter_frag_rows=
     (oc_state_loop_filter_frag_rows_func)oc_loop_filter_frag_rows_v6;
#   endif
  }
#   if defined(OC_ARM_ASM_NEON)
  if(_state->cpu_flags&OC_CPU_ARM_NEON){
#    if defined(OC_STATE_USE_VTABLE)
    _state->opt_vtable.frag_copy_list=oc_frag_copy_list_neon;
    _state->opt_vtable.frag_recon_intra=oc_frag_recon_intra_neon;
    _state->opt_vtable.frag_recon_inter=oc_frag_recon_inter_neon;
    _state->opt_vtable.frag_recon_inter2=oc_frag_recon_inter2_neon;
    _state->opt_vtable.state_frag_recon=oc_state_frag_recon_neon;
    _state->opt_vtable.loop_filter_init=oc_loop_filter_init_neon;
    _state->opt_vtable.state_loop_filter_frag_rows=
     (oc_state_loop_filter_frag_rows_func)oc_loop_filter_frag_rows_neon;
    _state->opt_vtable.idct8x8=oc_idct8x8_neon;
#    endif
    _state->opt_data.dct_fzig_zag=OC_FZIG_ZAG_NEON;
  }
#   endif
#  endif
# endif
}

void oc_state_frag_recon_arm(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant){
  unsigned char *dst;
  ptrdiff_t      frag_buf_off;
  int            ystride;
  int            refi;
  /*Apply the inverse transform.*/
  /*Special case only having a DC component.*/
  if(_last_zzi<2){
    ogg_uint16_t p;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_uint16_t)(_dct_coeffs[0]*(ogg_int32_t)_dc_quant+15>>5);
    oc_idct8x8_1_arm(_dct_coeffs+64,p);
  }
  else{
    /*First, dequantize the DC coefficient.*/
    _dct_coeffs[0]=(ogg_int16_t)(_dct_coeffs[0]*(int)_dc_quant);
    oc_idct8x8_arm(_dct_coeffs+64,_dct_coeffs,_last_zzi);
  }
  /*Fill in the target buffer.*/
  frag_buf_off=_state->frag_buf_offs[_fragi];
  refi=_state->frags[_fragi].refi;
  ystride=_state->ref_ystride[_pli];
  dst=_state->ref_frame_data[OC_FRAME_SELF]+frag_buf_off;
  if(refi==OC_FRAME_SELF)oc_frag_recon_intra_arm(dst,ystride,_dct_coeffs+64);
  else{
    const unsigned char *ref;
    int                  mvoffsets[2];
    ref=_state->ref_frame_data[refi]+frag_buf_off;
    if(oc_state_get_mv_offsets(_state,mvoffsets,_pli,
     _state->frag_mvs[_fragi])>1){
      oc_frag_recon_inter2_arm(dst,ref+mvoffsets[0],ref+mvoffsets[1],ystride,
       _dct_coeffs+64);
    }
    else oc_frag_recon_inter_arm(dst,ref+mvoffsets[0],ystride,_dct_coeffs+64);
  }
}

# if defined(OC_ARM_ASM_MEDIA)
void oc_state_frag_recon_v6(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant){
  unsigned char *dst;
  ptrdiff_t      frag_buf_off;
  int            ystride;
  int            refi;
  /*Apply the inverse transform.*/
  /*Special case only having a DC component.*/
  if(_last_zzi<2){
    ogg_uint16_t p;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_uint16_t)(_dct_coeffs[0]*(ogg_int32_t)_dc_quant+15>>5);
    oc_idct8x8_1_v6(_dct_coeffs+64,p);
  }
  else{
    /*First, dequantize the DC coefficient.*/
    _dct_coeffs[0]=(ogg_int16_t)(_dct_coeffs[0]*(int)_dc_quant);
    oc_idct8x8_v6(_dct_coeffs+64,_dct_coeffs,_last_zzi);
  }
  /*Fill in the target buffer.*/
  frag_buf_off=_state->frag_buf_offs[_fragi];
  refi=_state->frags[_fragi].refi;
  ystride=_state->ref_ystride[_pli];
  dst=_state->ref_frame_data[OC_FRAME_SELF]+frag_buf_off;
  if(refi==OC_FRAME_SELF)oc_frag_recon_intra_v6(dst,ystride,_dct_coeffs+64);
  else{
    const unsigned char *ref;
    int                  mvoffsets[2];
    ref=_state->ref_frame_data[refi]+frag_buf_off;
    if(oc_state_get_mv_offsets(_state,mvoffsets,_pli,
     _state->frag_mvs[_fragi])>1){
      oc_frag_recon_inter2_v6(dst,ref+mvoffsets[0],ref+mvoffsets[1],ystride,
       _dct_coeffs+64);
    }
    else oc_frag_recon_inter_v6(dst,ref+mvoffsets[0],ystride,_dct_coeffs+64);
  }
}

# if defined(OC_ARM_ASM_NEON)
void oc_state_frag_recon_neon(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant){
  unsigned char *dst;
  ptrdiff_t      frag_buf_off;
  int            ystride;
  int            refi;
  /*Apply the inverse transform.*/
  /*Special case only having a DC component.*/
  if(_last_zzi<2){
    ogg_uint16_t p;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_uint16_t)(_dct_coeffs[0]*(ogg_int32_t)_dc_quant+15>>5);
    oc_idct8x8_1_neon(_dct_coeffs+64,p);
  }
  else{
    /*First, dequantize the DC coefficient.*/
    _dct_coeffs[0]=(ogg_int16_t)(_dct_coeffs[0]*(int)_dc_quant);
    oc_idct8x8_neon(_dct_coeffs+64,_dct_coeffs,_last_zzi);
  }
  /*Fill in the target buffer.*/
  frag_buf_off=_state->frag_buf_offs[_fragi];
  refi=_state->frags[_fragi].refi;
  ystride=_state->ref_ystride[_pli];
  dst=_state->ref_frame_data[OC_FRAME_SELF]+frag_buf_off;
  if(refi==OC_FRAME_SELF)oc_frag_recon_intra_neon(dst,ystride,_dct_coeffs+64);
  else{
    const unsigned char *ref;
    int                  mvoffsets[2];
    ref=_state->ref_frame_data[refi]+frag_buf_off;
    if(oc_state_get_mv_offsets(_state,mvoffsets,_pli,
     _state->frag_mvs[_fragi])>1){
      oc_frag_recon_inter2_neon(dst,ref+mvoffsets[0],ref+mvoffsets[1],ystride,
       _dct_coeffs+64);
    }
    else oc_frag_recon_inter_neon(dst,ref+mvoffsets[0],ystride,_dct_coeffs+64);
  }
}
#  endif
# endif

#endif
