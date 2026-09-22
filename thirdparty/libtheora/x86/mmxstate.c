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

/*MMX acceleration of complete fragment reconstruction algorithm.
  Originally written by Rudolf Marek.*/
#include <string.h>
#include "x86int.h"
#include "mmxloop.h"

#if defined(OC_X86_ASM)

void oc_state_frag_recon_mmx(const oc_theora_state *_state,ptrdiff_t _fragi,
 int _pli,ogg_int16_t _dct_coeffs[128],int _last_zzi,ogg_uint16_t _dc_quant){
  unsigned char *dst;
  ptrdiff_t      frag_buf_off;
  int            ystride;
  int            refi;
  /*Apply the inverse transform.*/
  /*Special case only having a DC component.*/
  if(_last_zzi<2){
    /*Note that this value must be unsigned, to keep the __asm__ block from
       sign-extending it when it puts it in a register.*/
    ogg_uint16_t p;
    int          i;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_int16_t)(_dct_coeffs[0]*(ogg_int32_t)_dc_quant+15>>5);
    /*Fill _dct_coeffs with p.*/
    __asm__ __volatile__(
      /*mm0=0000 0000 0000 AAAA*/
      "movd %[p],%%mm0\n\t"
      /*mm0=0000 0000 AAAA AAAA*/
      "punpcklwd %%mm0,%%mm0\n\t"
      /*mm0=AAAA AAAA AAAA AAAA*/
      "punpckldq %%mm0,%%mm0\n\t"
      :
      :[p]"r"((unsigned)p)
    );
    for(i=0;i<4;i++){
      __asm__ __volatile__(
        "movq %%mm0,"OC_MEM_OFFS(0x00,y)"\n\t"
        "movq %%mm0,"OC_MEM_OFFS(0x08,y)"\n\t"
        "movq %%mm0,"OC_MEM_OFFS(0x10,y)"\n\t"
        "movq %%mm0,"OC_MEM_OFFS(0x18,y)"\n\t"
        :[y]"=m"OC_ARRAY_OPERAND(ogg_int16_t,_dct_coeffs+64+16*i,16)
      );
    }
  }
  else{
    /*Dequantize the DC coefficient.*/
    _dct_coeffs[0]=(ogg_int16_t)(_dct_coeffs[0]*(int)_dc_quant);
    oc_idct8x8(_state,_dct_coeffs+64,_dct_coeffs,_last_zzi);
  }
  /*Fill in the target buffer.*/
  frag_buf_off=_state->frag_buf_offs[_fragi];
  refi=_state->frags[_fragi].refi;
  ystride=_state->ref_ystride[_pli];
  dst=_state->ref_frame_data[OC_FRAME_SELF]+frag_buf_off;
  if(refi==OC_FRAME_SELF)oc_frag_recon_intra_mmx(dst,ystride,_dct_coeffs+64);
  else{
    const unsigned char *ref;
    int                  mvoffsets[2];
    ref=_state->ref_frame_data[refi]+frag_buf_off;
    if(oc_state_get_mv_offsets(_state,mvoffsets,_pli,
     _state->frag_mvs[_fragi])>1){
      oc_frag_recon_inter2_mmx(dst,ref+mvoffsets[0],ref+mvoffsets[1],ystride,
       _dct_coeffs+64);
    }
    else oc_frag_recon_inter_mmx(dst,ref+mvoffsets[0],ystride,_dct_coeffs+64);
  }
}

/*We copy these entire function to inline the actual MMX routines so that we
   use only a single indirect call.*/

void oc_loop_filter_init_mmx(signed char _bv[256],int _flimit){
  memset(_bv,_flimit,8);
}

/*Apply the loop filter to a given set of fragment rows in the given plane.
  The filter may be run on the bottom edge, affecting pixels in the next row of
   fragments, so this row also needs to be available.
  _bv:        The bounding values array.
  _refi:      The index of the frame buffer to filter.
  _pli:       The color plane to filter.
  _fragy0:    The Y coordinate of the first fragment row to filter.
  _fragy_end: The Y coordinate of the fragment row to stop filtering at.*/
void oc_state_loop_filter_frag_rows_mmx(const oc_theora_state *_state,
 signed char _bv[256],int _refi,int _pli,int _fragy0,int _fragy_end){
  OC_ALIGN8(unsigned char   ll[8]);
  const oc_fragment_plane *fplane;
  const oc_fragment       *frags;
  const ptrdiff_t         *frag_buf_offs;
  unsigned char           *ref_frame_data;
  ptrdiff_t                fragi_top;
  ptrdiff_t                fragi_bot;
  ptrdiff_t                fragi0;
  ptrdiff_t                fragi0_end;
  int                      ystride;
  int                      nhfrags;
  memset(ll,_state->loop_filter_limits[_state->qis[0]],sizeof(ll));
  fplane=_state->fplanes+_pli;
  nhfrags=fplane->nhfrags;
  fragi_top=fplane->froffset;
  fragi_bot=fragi_top+fplane->nfrags;
  fragi0=fragi_top+_fragy0*(ptrdiff_t)nhfrags;
  fragi0_end=fragi0+(_fragy_end-_fragy0)*(ptrdiff_t)nhfrags;
  ystride=_state->ref_ystride[_pli];
  frags=_state->frags;
  frag_buf_offs=_state->frag_buf_offs;
  ref_frame_data=_state->ref_frame_data[_refi];
  /*The following loops are constructed somewhat non-intuitively on purpose.
    The main idea is: if a block boundary has at least one coded fragment on
     it, the filter is applied to it.
    However, the order that the filters are applied in matters, and VP3 chose
     the somewhat strange ordering used below.*/
  while(fragi0<fragi0_end){
    ptrdiff_t fragi;
    ptrdiff_t fragi_end;
    fragi=fragi0;
    fragi_end=fragi+nhfrags;
    while(fragi<fragi_end){
      if(frags[fragi].coded){
        unsigned char *ref;
        ref=ref_frame_data+frag_buf_offs[fragi];
        if(fragi>fragi0){
          OC_LOOP_FILTER_H(OC_LOOP_FILTER8_MMX,ref,ystride,ll);
        }
        if(fragi0>fragi_top){
          OC_LOOP_FILTER_V(OC_LOOP_FILTER8_MMX,ref,ystride,ll);
        }
        if(fragi+1<fragi_end&&!frags[fragi+1].coded){
          OC_LOOP_FILTER_H(OC_LOOP_FILTER8_MMX,ref+8,ystride,ll);
        }
        if(fragi+nhfrags<fragi_bot&&!frags[fragi+nhfrags].coded){
          OC_LOOP_FILTER_V(OC_LOOP_FILTER8_MMX,ref+(ystride*8),ystride,ll);
        }
      }
      fragi++;
    }
    fragi0+=nhfrags;
  }
}

void oc_loop_filter_init_mmxext(signed char _bv[256],int _flimit){
  memset(_bv,~(_flimit<<1),8);
}

/*Apply the loop filter to a given set of fragment rows in the given plane.
  The filter may be run on the bottom edge, affecting pixels in the next row of
   fragments, so this row also needs to be available.
  _bv:        The bounding values array.
  _refi:      The index of the frame buffer to filter.
  _pli:       The color plane to filter.
  _fragy0:    The Y coordinate of the first fragment row to filter.
  _fragy_end: The Y coordinate of the fragment row to stop filtering at.*/
void oc_state_loop_filter_frag_rows_mmxext(const oc_theora_state *_state,
 signed char _bv[256],int _refi,int _pli,int _fragy0,int _fragy_end){
  const oc_fragment_plane *fplane;
  const oc_fragment       *frags;
  const ptrdiff_t         *frag_buf_offs;
  unsigned char           *ref_frame_data;
  ptrdiff_t                fragi_top;
  ptrdiff_t                fragi_bot;
  ptrdiff_t                fragi0;
  ptrdiff_t                fragi0_end;
  int                      ystride;
  int                      nhfrags;
  fplane=_state->fplanes+_pli;
  nhfrags=fplane->nhfrags;
  fragi_top=fplane->froffset;
  fragi_bot=fragi_top+fplane->nfrags;
  fragi0=fragi_top+_fragy0*(ptrdiff_t)nhfrags;
  fragi0_end=fragi_top+_fragy_end*(ptrdiff_t)nhfrags;
  ystride=_state->ref_ystride[_pli];
  frags=_state->frags;
  frag_buf_offs=_state->frag_buf_offs;
  ref_frame_data=_state->ref_frame_data[_refi];
  /*The following loops are constructed somewhat non-intuitively on purpose.
    The main idea is: if a block boundary has at least one coded fragment on
     it, the filter is applied to it.
    However, the order that the filters are applied in matters, and VP3 chose
     the somewhat strange ordering used below.*/
  while(fragi0<fragi0_end){
    ptrdiff_t fragi;
    ptrdiff_t fragi_end;
    fragi=fragi0;
    fragi_end=fragi+nhfrags;
    while(fragi<fragi_end){
      if(frags[fragi].coded){
        unsigned char *ref;
        ref=ref_frame_data+frag_buf_offs[fragi];
        if(fragi>fragi0){
          OC_LOOP_FILTER_H(OC_LOOP_FILTER8_MMXEXT,ref,ystride,_bv);
        }
        if(fragi0>fragi_top){
          OC_LOOP_FILTER_V(OC_LOOP_FILTER8_MMXEXT,ref,ystride,_bv);
        }
        if(fragi+1<fragi_end&&!frags[fragi+1].coded){
          OC_LOOP_FILTER_H(OC_LOOP_FILTER8_MMXEXT,ref+8,ystride,_bv);
        }
        if(fragi+nhfrags<fragi_bot&&!frags[fragi+nhfrags].coded){
          OC_LOOP_FILTER_V(OC_LOOP_FILTER8_MMXEXT,ref+(ystride*8),ystride,_bv);
        }
      }
      fragi++;
    }
    fragi0+=nhfrags;
  }
}

#endif
