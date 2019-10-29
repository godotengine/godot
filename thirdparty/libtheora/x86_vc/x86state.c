/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: x86state.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/

#include "x86int.h"

#if defined(OC_X86_ASM)

#include "../cpu.c"

/*This table has been modified from OC_FZIG_ZAG by baking a 4x4 transpose into
   each quadrant of the destination.*/
static const unsigned char OC_FZIG_ZAG_MMX[128]={
   0, 8, 1, 2, 9,16,24,17,
  10, 3,32,11,18,25, 4,12,
   5,26,19,40,33,34,41,48,
  27, 6,13,20,28,21,14, 7,
  56,49,42,35,43,50,57,36,
  15,22,29,30,23,44,37,58,
  51,59,38,45,52,31,60,53,
  46,39,47,54,61,62,55,63,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
};

void oc_state_vtable_init_x86(oc_theora_state *_state){
  _state->cpu_flags=oc_cpu_flags_get();
  if(_state->cpu_flags&OC_CPU_X86_MMX){
    _state->opt_vtable.frag_copy=oc_frag_copy_mmx;
    _state->opt_vtable.frag_recon_intra=oc_frag_recon_intra_mmx;
    _state->opt_vtable.frag_recon_inter=oc_frag_recon_inter_mmx;
    _state->opt_vtable.frag_recon_inter2=oc_frag_recon_inter2_mmx;
    _state->opt_vtable.idct8x8=oc_idct8x8_mmx;
    _state->opt_vtable.state_frag_recon=oc_state_frag_recon_mmx;
    _state->opt_vtable.state_frag_copy_list=oc_state_frag_copy_list_mmx;
    _state->opt_vtable.state_loop_filter_frag_rows=
     oc_state_loop_filter_frag_rows_mmx;
    _state->opt_vtable.restore_fpu=oc_restore_fpu_mmx;
    _state->opt_data.dct_fzig_zag=OC_FZIG_ZAG_MMX;
  }
  else oc_state_vtable_init_c(_state);
}
#endif
