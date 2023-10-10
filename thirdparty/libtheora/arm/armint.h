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
#if !defined(_arm_armint_H)
# define _arm_armint_H (1)
# include "../internal.h"

# if defined(OC_ARM_ASM)

#  if defined(__ARMEB__)
#   error "Big-endian configurations are not supported by the ARM asm. " \
 "Reconfigure with --disable-asm or undefine OC_ARM_ASM."
#  endif

#  define oc_state_accel_init oc_state_accel_init_arm
/*This function is implemented entirely in asm, so it's helpful to pull out all
   of the things that depend on structure offsets.
  We reuse the function pointer with the wrong prototype, though.*/
#  define oc_state_loop_filter_frag_rows(_state,_bv,_refi,_pli, \
 _fragy0,_fragy_end) \
  ((oc_loop_filter_frag_rows_arm_func) \
   (_state)->opt_vtable.state_loop_filter_frag_rows)( \
   (_state)->ref_frame_data[(_refi)],(_state)->ref_ystride[(_pli)], \
   (_bv), \
   (_state)->frags, \
   (_state)->fplanes[(_pli)].froffset \
   +(_fragy0)*(ptrdiff_t)(_state)->fplanes[(_pli)].nhfrags, \
   (_state)->fplanes[(_pli)].froffset \
   +(_fragy_end)*(ptrdiff_t)(_state)->fplanes[(_pli)].nhfrags, \
   (_state)->fplanes[(_pli)].froffset, \
   (_state)->fplanes[(_pli)].froffset+(_state)->fplanes[(_pli)].nfrags, \
   (_state)->frag_buf_offs, \
   (_state)->fplanes[(_pli)].nhfrags)
/*For everything else the default vtable macros are fine.*/
#  define OC_STATE_USE_VTABLE (1)
# endif

# include "../state.h"
# include "armcpu.h"

# if defined(OC_ARM_ASM)
typedef void (*oc_loop_filter_frag_rows_arm_func)(
 unsigned char *_ref_frame_data,int _ystride,signed char _bv[256],
 const oc_fragment *_frags,ptrdiff_t _fragi0,ptrdiff_t _fragi0_end,
 ptrdiff_t _fragi_top,ptrdiff_t _fragi_bot,
 const ptrdiff_t *_frag_buf_offs,int _nhfrags);

void oc_state_accel_init_arm(oc_theora_state *_state);
void oc_frag_copy_list_arm(unsigned char *_dst_frame,
 const unsigned char *_src_frame,int _ystride,
 const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs);
void oc_frag_recon_intra_arm(unsigned char *_dst,int _ystride,
 const ogg_int16_t *_residue);
void oc_frag_recon_inter_arm(unsigned char *_dst,const unsigned char *_src,
 int _ystride,const ogg_int16_t *_residue);
void oc_frag_recon_inter2_arm(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t *_residue);
void oc_idct8x8_1_arm(ogg_int16_t _y[64],ogg_uint16_t _dc);
void oc_idct8x8_arm(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi);
void oc_state_frag_recon_arm(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant);
void oc_loop_filter_frag_rows_arm(unsigned char *_ref_frame_data,
 int _ystride,signed char *_bv,const oc_fragment *_frags,ptrdiff_t _fragi0,
 ptrdiff_t _fragi0_end,ptrdiff_t _fragi_top,ptrdiff_t _fragi_bot,
 const ptrdiff_t *_frag_buf_offs,int _nhfrags);

#  if defined(OC_ARM_ASM_EDSP)
void oc_frag_copy_list_edsp(unsigned char *_dst_frame,
 const unsigned char *_src_frame,int _ystride,
 const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs);

#   if defined(OC_ARM_ASM_MEDIA)
void oc_frag_recon_intra_v6(unsigned char *_dst,int _ystride,
 const ogg_int16_t *_residue);
void oc_frag_recon_inter_v6(unsigned char *_dst,const unsigned char *_src,
 int _ystride,const ogg_int16_t *_residue);
void oc_frag_recon_inter2_v6(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t *_residue);
void oc_idct8x8_1_v6(ogg_int16_t _y[64],ogg_uint16_t _dc);
void oc_idct8x8_v6(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi);
void oc_state_frag_recon_v6(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant);
void oc_loop_filter_init_v6(signed char *_bv,int _flimit);
void oc_loop_filter_frag_rows_v6(unsigned char *_ref_frame_data,
 int _ystride,signed char *_bv,const oc_fragment *_frags,ptrdiff_t _fragi0,
 ptrdiff_t _fragi0_end,ptrdiff_t _fragi_top,ptrdiff_t _fragi_bot,
 const ptrdiff_t *_frag_buf_offs,int _nhfrags);

#    if defined(OC_ARM_ASM_NEON)
void oc_frag_copy_list_neon(unsigned char *_dst_frame,
 const unsigned char *_src_frame,int _ystride,
 const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs);
void oc_frag_recon_intra_neon(unsigned char *_dst,int _ystride,
 const ogg_int16_t *_residue);
void oc_frag_recon_inter_neon(unsigned char *_dst,const unsigned char *_src,
 int _ystride,const ogg_int16_t *_residue);
void oc_frag_recon_inter2_neon(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t *_residue);
void oc_idct8x8_1_neon(ogg_int16_t _y[64],ogg_uint16_t _dc);
void oc_idct8x8_neon(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi);
void oc_state_frag_recon_neon(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant);
void oc_loop_filter_init_neon(signed char *_bv,int _flimit);
void oc_loop_filter_frag_rows_neon(unsigned char *_ref_frame_data,
 int _ystride,signed char *_bv,const oc_fragment *_frags,ptrdiff_t _fragi0,
 ptrdiff_t _fragi0_end,ptrdiff_t _fragi_top,ptrdiff_t _fragi_bot,
 const ptrdiff_t *_frag_buf_offs,int _nhfrags);
#    endif
#   endif
#  endif
# endif

#endif
