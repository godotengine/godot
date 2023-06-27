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
    last mod: $Id$

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ogg/ogg.h>
#include "decint.h"
#if defined(OC_DUMP_IMAGES)
# include <stdio.h>
# include "png.h"
#endif
#if defined(HAVE_CAIRO)
# include <cairo.h>
#endif


/*No post-processing.*/
#define OC_PP_LEVEL_DISABLED  (0)
/*Keep track of DC qi for each block only.*/
#define OC_PP_LEVEL_TRACKDCQI (1)
/*Deblock the luma plane.*/
#define OC_PP_LEVEL_DEBLOCKY  (2)
/*Dering the luma plane.*/
#define OC_PP_LEVEL_DERINGY   (3)
/*Stronger luma plane deringing.*/
#define OC_PP_LEVEL_SDERINGY  (4)
/*Deblock the chroma planes.*/
#define OC_PP_LEVEL_DEBLOCKC  (5)
/*Dering the chroma planes.*/
#define OC_PP_LEVEL_DERINGC   (6)
/*Stronger chroma plane deringing.*/
#define OC_PP_LEVEL_SDERINGC  (7)
/*Maximum valid post-processing level.*/
#define OC_PP_LEVEL_MAX       (7)



/*The mode alphabets for the various mode coding schemes.
  Scheme 0 uses a custom alphabet, which is not stored in this table.*/
static const unsigned char OC_MODE_ALPHABETS[7][OC_NMODES]={
  /*Last MV dominates */
  {
    OC_MODE_INTER_MV_LAST,OC_MODE_INTER_MV_LAST2,OC_MODE_INTER_MV,
    OC_MODE_INTER_NOMV,OC_MODE_INTRA,OC_MODE_GOLDEN_NOMV,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  },
  {
    OC_MODE_INTER_MV_LAST,OC_MODE_INTER_MV_LAST2,OC_MODE_INTER_NOMV,
    OC_MODE_INTER_MV,OC_MODE_INTRA,OC_MODE_GOLDEN_NOMV,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  },
  {
    OC_MODE_INTER_MV_LAST,OC_MODE_INTER_MV,OC_MODE_INTER_MV_LAST2,
    OC_MODE_INTER_NOMV,OC_MODE_INTRA,OC_MODE_GOLDEN_NOMV,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  },
  {
    OC_MODE_INTER_MV_LAST,OC_MODE_INTER_MV,OC_MODE_INTER_NOMV,
    OC_MODE_INTER_MV_LAST2,OC_MODE_INTRA,OC_MODE_GOLDEN_NOMV,
    OC_MODE_GOLDEN_MV,OC_MODE_INTER_MV_FOUR
  },
  /*No MV dominates.*/
  {
    OC_MODE_INTER_NOMV,OC_MODE_INTER_MV_LAST,OC_MODE_INTER_MV_LAST2,
    OC_MODE_INTER_MV,OC_MODE_INTRA,OC_MODE_GOLDEN_NOMV,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  },
  {
    OC_MODE_INTER_NOMV,OC_MODE_GOLDEN_NOMV,OC_MODE_INTER_MV_LAST,
    OC_MODE_INTER_MV_LAST2,OC_MODE_INTER_MV,OC_MODE_INTRA,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  },
  /*Default ordering.*/
  {
    OC_MODE_INTER_NOMV,OC_MODE_INTRA,OC_MODE_INTER_MV,OC_MODE_INTER_MV_LAST,
    OC_MODE_INTER_MV_LAST2,OC_MODE_GOLDEN_NOMV,OC_MODE_GOLDEN_MV,
    OC_MODE_INTER_MV_FOUR
  }
};


/*The original DCT tokens are extended and reordered during the construction of
   the Huffman tables.
  The extension means more bits can be read with fewer calls to the bitpacker
   during the Huffman decoding process (at the cost of larger Huffman tables),
   and fewer tokens require additional extra bits (reducing the average storage
   per decoded token).
  The revised ordering reveals essential information in the token value
   itself; specifically, whether or not there are additional extra bits to read
   and the parameter to which those extra bits are applied.
  The token is used to fetch a code word from the OC_DCT_CODE_WORD table below.
  The extra bits are added into code word at the bit position inferred from the
   token value, giving the final code word from which all required parameters
   are derived.
  The number of EOBs and the leading zero run length can be extracted directly.
  The coefficient magnitude is optionally negated before extraction, according
   to a 'flip' bit.*/

/*The number of additional extra bits that are decoded with each of the
   internal DCT tokens.*/
static const unsigned char OC_INTERNAL_DCT_TOKEN_EXTRA_BITS[15]={
  12,4,3,3,4,4,5,5,8,8,8,8,3,3,6
};

/*Whether or not an internal token needs any additional extra bits.*/
#define OC_DCT_TOKEN_NEEDS_MORE(token) \
 (token<(int)(sizeof(OC_INTERNAL_DCT_TOKEN_EXTRA_BITS)/ \
  sizeof(*OC_INTERNAL_DCT_TOKEN_EXTRA_BITS)))

/*This token (OC_DCT_REPEAT_RUN3_TOKEN) requires more than 8 extra bits.*/
#define OC_DCT_TOKEN_FAT_EOB (0)

/*The number of EOBs to use for an end-of-frame token.
  Note: We want to set eobs to PTRDIFF_MAX here, but that requires C99, which
   is not yet available everywhere; this should be equivalent.*/
#define OC_DCT_EOB_FINISH (~(size_t)0>>1)

/*The location of the (6) run length bits in the code word.
  These are placed at index 0 and given 8 bits (even though 6 would suffice)
   because it may be faster to extract the lower byte on some platforms.*/
#define OC_DCT_CW_RLEN_SHIFT (0)
/*The location of the (12) EOB bits in the code word.*/
#define OC_DCT_CW_EOB_SHIFT  (8)
/*The location of the (1) flip bit in the code word.
  This must be right under the magnitude bits.*/
#define OC_DCT_CW_FLIP_BIT   (20)
/*The location of the (11) token magnitude bits in the code word.
  These must be last, and rely on a sign-extending right shift.*/
#define OC_DCT_CW_MAG_SHIFT  (21)

/*Pack the given fields into a code word.*/
#define OC_DCT_CW_PACK(_eobs,_rlen,_mag,_flip) \
 ((_eobs)<<OC_DCT_CW_EOB_SHIFT| \
 (_rlen)<<OC_DCT_CW_RLEN_SHIFT| \
 (_flip)<<OC_DCT_CW_FLIP_BIT| \
 (_mag)-(_flip)<<OC_DCT_CW_MAG_SHIFT)

/*A special code word value that signals the end of the frame (a long EOB run
   of zero).*/
#define OC_DCT_CW_FINISH (0)

/*The position at which to insert the extra bits in the code word.
  We use this formulation because Intel has no useful cmov.
  A real architecture would probably do better with two of those.
  This translates to 11 instructions(!), and is _still_ faster than either a
   table lookup (just barely) or the naive double-ternary implementation (which
   gcc translates to a jump and a cmov).
  This assumes OC_DCT_CW_RLEN_SHIFT is zero, but could easily be reworked if
   you want to make one of the other shifts zero.*/
#define OC_DCT_TOKEN_EB_POS(_token) \
 ((OC_DCT_CW_EOB_SHIFT-OC_DCT_CW_MAG_SHIFT&-((_token)<2)) \
 +(OC_DCT_CW_MAG_SHIFT&-((_token)<12)))

/*The code words for each internal token.
  See the notes at OC_DCT_TOKEN_MAP for the reasons why things are out of
   order.*/
static const ogg_int32_t OC_DCT_CODE_WORD[92]={
  /*These tokens require additional extra bits for the EOB count.*/
  /*OC_DCT_REPEAT_RUN3_TOKEN (12 extra bits)*/
  OC_DCT_CW_FINISH,
  /*OC_DCT_REPEAT_RUN2_TOKEN (4 extra bits)*/
  OC_DCT_CW_PACK(16, 0,  0,0),
  /*These tokens require additional extra bits for the magnitude.*/
  /*OC_DCT_VAL_CAT5 (4 extra bits-1 already read)*/
  OC_DCT_CW_PACK( 0, 0, 13,0),
  OC_DCT_CW_PACK( 0, 0, 13,1),
  /*OC_DCT_VAL_CAT6 (5 extra bits-1 already read)*/
  OC_DCT_CW_PACK( 0, 0, 21,0),
  OC_DCT_CW_PACK( 0, 0, 21,1),
  /*OC_DCT_VAL_CAT7 (6 extra bits-1 already read)*/
  OC_DCT_CW_PACK( 0, 0, 37,0),
  OC_DCT_CW_PACK( 0, 0, 37,1),
  /*OC_DCT_VAL_CAT8 (10 extra bits-2 already read)*/
  OC_DCT_CW_PACK( 0, 0, 69,0),
  OC_DCT_CW_PACK( 0, 0,325,0),
  OC_DCT_CW_PACK( 0, 0, 69,1),
  OC_DCT_CW_PACK( 0, 0,325,1),
  /*These tokens require additional extra bits for the run length.*/
  /*OC_DCT_RUN_CAT1C (4 extra bits-1 already read)*/
  OC_DCT_CW_PACK( 0,10, +1,0),
  OC_DCT_CW_PACK( 0,10, -1,0),
  /*OC_DCT_ZRL_TOKEN (6 extra bits)
    Flip is set to distinguish this from OC_DCT_CW_FINISH.*/
  OC_DCT_CW_PACK( 0, 0,  0,1),
  /*The remaining tokens require no additional extra bits.*/
  /*OC_DCT_EOB1_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 1, 0,  0,0),
  /*OC_DCT_EOB2_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 2, 0,  0,0),
  /*OC_DCT_EOB3_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 3, 0,  0,0),
  /*OC_DCT_RUN_CAT1A (1 extra bit-1 already read)x5*/
  OC_DCT_CW_PACK( 0, 1, +1,0),
  OC_DCT_CW_PACK( 0, 1, -1,0),
  OC_DCT_CW_PACK( 0, 2, +1,0),
  OC_DCT_CW_PACK( 0, 2, -1,0),
  OC_DCT_CW_PACK( 0, 3, +1,0),
  OC_DCT_CW_PACK( 0, 3, -1,0),
  OC_DCT_CW_PACK( 0, 4, +1,0),
  OC_DCT_CW_PACK( 0, 4, -1,0),
  OC_DCT_CW_PACK( 0, 5, +1,0),
  OC_DCT_CW_PACK( 0, 5, -1,0),
  /*OC_DCT_RUN_CAT2A (2 extra bits-2 already read)*/
  OC_DCT_CW_PACK( 0, 1, +2,0),
  OC_DCT_CW_PACK( 0, 1, +3,0),
  OC_DCT_CW_PACK( 0, 1, -2,0),
  OC_DCT_CW_PACK( 0, 1, -3,0),
  /*OC_DCT_RUN_CAT1B (3 extra bits-3 already read)*/
  OC_DCT_CW_PACK( 0, 6, +1,0),
  OC_DCT_CW_PACK( 0, 7, +1,0),
  OC_DCT_CW_PACK( 0, 8, +1,0),
  OC_DCT_CW_PACK( 0, 9, +1,0),
  OC_DCT_CW_PACK( 0, 6, -1,0),
  OC_DCT_CW_PACK( 0, 7, -1,0),
  OC_DCT_CW_PACK( 0, 8, -1,0),
  OC_DCT_CW_PACK( 0, 9, -1,0),
  /*OC_DCT_RUN_CAT2B (3 extra bits-3 already read)*/
  OC_DCT_CW_PACK( 0, 2, +2,0),
  OC_DCT_CW_PACK( 0, 3, +2,0),
  OC_DCT_CW_PACK( 0, 2, +3,0),
  OC_DCT_CW_PACK( 0, 3, +3,0),
  OC_DCT_CW_PACK( 0, 2, -2,0),
  OC_DCT_CW_PACK( 0, 3, -2,0),
  OC_DCT_CW_PACK( 0, 2, -3,0),
  OC_DCT_CW_PACK( 0, 3, -3,0),
  /*OC_DCT_SHORT_ZRL_TOKEN (3 extra bits-3 already read)
    Flip is set on the first one to distinguish it from OC_DCT_CW_FINISH.*/
  OC_DCT_CW_PACK( 0, 0,  0,1),
  OC_DCT_CW_PACK( 0, 1,  0,0),
  OC_DCT_CW_PACK( 0, 2,  0,0),
  OC_DCT_CW_PACK( 0, 3,  0,0),
  OC_DCT_CW_PACK( 0, 4,  0,0),
  OC_DCT_CW_PACK( 0, 5,  0,0),
  OC_DCT_CW_PACK( 0, 6,  0,0),
  OC_DCT_CW_PACK( 0, 7,  0,0),
  /*OC_ONE_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 0, 0, +1,0),
  /*OC_MINUS_ONE_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 0, 0, -1,0),
  /*OC_TWO_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 0, 0, +2,0),
  /*OC_MINUS_TWO_TOKEN (0 extra bits)*/
  OC_DCT_CW_PACK( 0, 0, -2,0),
  /*OC_DCT_VAL_CAT2 (1 extra bit-1 already read)x4*/
  OC_DCT_CW_PACK( 0, 0, +3,0),
  OC_DCT_CW_PACK( 0, 0, -3,0),
  OC_DCT_CW_PACK( 0, 0, +4,0),
  OC_DCT_CW_PACK( 0, 0, -4,0),
  OC_DCT_CW_PACK( 0, 0, +5,0),
  OC_DCT_CW_PACK( 0, 0, -5,0),
  OC_DCT_CW_PACK( 0, 0, +6,0),
  OC_DCT_CW_PACK( 0, 0, -6,0),
  /*OC_DCT_VAL_CAT3 (2 extra bits-2 already read)*/
  OC_DCT_CW_PACK( 0, 0, +7,0),
  OC_DCT_CW_PACK( 0, 0, +8,0),
  OC_DCT_CW_PACK( 0, 0, -7,0),
  OC_DCT_CW_PACK( 0, 0, -8,0),
  /*OC_DCT_VAL_CAT4 (3 extra bits-3 already read)*/
  OC_DCT_CW_PACK( 0, 0, +9,0),
  OC_DCT_CW_PACK( 0, 0,+10,0),
  OC_DCT_CW_PACK( 0, 0,+11,0),
  OC_DCT_CW_PACK( 0, 0,+12,0),
  OC_DCT_CW_PACK( 0, 0, -9,0),
  OC_DCT_CW_PACK( 0, 0,-10,0),
  OC_DCT_CW_PACK( 0, 0,-11,0),
  OC_DCT_CW_PACK( 0, 0,-12,0),
  /*OC_DCT_REPEAT_RUN1_TOKEN (3 extra bits-3 already read)*/
  OC_DCT_CW_PACK( 8, 0,  0,0),
  OC_DCT_CW_PACK( 9, 0,  0,0),
  OC_DCT_CW_PACK(10, 0,  0,0),
  OC_DCT_CW_PACK(11, 0,  0,0),
  OC_DCT_CW_PACK(12, 0,  0,0),
  OC_DCT_CW_PACK(13, 0,  0,0),
  OC_DCT_CW_PACK(14, 0,  0,0),
  OC_DCT_CW_PACK(15, 0,  0,0),
  /*OC_DCT_REPEAT_RUN0_TOKEN (2 extra bits-2 already read)*/
  OC_DCT_CW_PACK( 4, 0,  0,0),
  OC_DCT_CW_PACK( 5, 0,  0,0),
  OC_DCT_CW_PACK( 6, 0,  0,0),
  OC_DCT_CW_PACK( 7, 0,  0,0),
};



static int oc_sb_run_unpack(oc_pack_buf *_opb){
  /*Coding scheme:
       Codeword            Run Length
     0                       1
     10x                     2-3
     110x                    4-5
     1110xx                  6-9
     11110xxx                10-17
     111110xxxx              18-33
     111111xxxxxxxxxxxx      34-4129*/
  static const ogg_int16_t OC_SB_RUN_TREE[22]={
    4,
     -(1<<8|1),-(1<<8|1),-(1<<8|1),-(1<<8|1),
     -(1<<8|1),-(1<<8|1),-(1<<8|1),-(1<<8|1),
     -(3<<8|2),-(3<<8|2),-(3<<8|3),-(3<<8|3),
     -(4<<8|4),-(4<<8|5),-(4<<8|2<<4|6-6),17,
      2,
       -(2<<8|2<<4|10-6),-(2<<8|2<<4|14-6),-(2<<8|4<<4|18-6),-(2<<8|12<<4|34-6)
  };
  int ret;
  ret=oc_huff_token_decode(_opb,OC_SB_RUN_TREE);
  if(ret>=0x10){
    int offs;
    offs=ret&0x1F;
    ret=6+offs+(int)oc_pack_read(_opb,ret-offs>>4);
  }
  return ret;
}

static int oc_block_run_unpack(oc_pack_buf *_opb){
  /*Coding scheme:
     Codeword             Run Length
     0x                      1-2
     10x                     3-4
     110x                    5-6
     1110xx                  7-10
     11110xx                 11-14
     11111xxxx               15-30*/
  static const ogg_int16_t OC_BLOCK_RUN_TREE[61]={
    5,
     -(2<<8|1),-(2<<8|1),-(2<<8|1),-(2<<8|1),
     -(2<<8|1),-(2<<8|1),-(2<<8|1),-(2<<8|1),
     -(2<<8|2),-(2<<8|2),-(2<<8|2),-(2<<8|2),
     -(2<<8|2),-(2<<8|2),-(2<<8|2),-(2<<8|2),
     -(3<<8|3),-(3<<8|3),-(3<<8|3),-(3<<8|3),
     -(3<<8|4),-(3<<8|4),-(3<<8|4),-(3<<8|4),
     -(4<<8|5),-(4<<8|5),-(4<<8|6),-(4<<8|6),
     33,       36,       39,       44,
      1,-(1<<8|7),-(1<<8|8),
      1,-(1<<8|9),-(1<<8|10),
      2,-(2<<8|11),-(2<<8|12),-(2<<8|13),-(2<<8|14),
      4,
       -(4<<8|15),-(4<<8|16),-(4<<8|17),-(4<<8|18),
       -(4<<8|19),-(4<<8|20),-(4<<8|21),-(4<<8|22),
       -(4<<8|23),-(4<<8|24),-(4<<8|25),-(4<<8|26),
       -(4<<8|27),-(4<<8|28),-(4<<8|29),-(4<<8|30)
  };
  return oc_huff_token_decode(_opb,OC_BLOCK_RUN_TREE);
}



void oc_dec_accel_init_c(oc_dec_ctx *_dec){
# if defined(OC_DEC_USE_VTABLE)
  _dec->opt_vtable.dc_unpredict_mcu_plane=
   oc_dec_dc_unpredict_mcu_plane_c;
# endif
}

static int oc_dec_init(oc_dec_ctx *_dec,const th_info *_info,
 const th_setup_info *_setup){
  int qti;
  int pli;
  int qi;
  int ret;
  ret=oc_state_init(&_dec->state,_info,3);
  if(ret<0)return ret;
  ret=oc_huff_trees_copy(_dec->huff_tables,
   (const ogg_int16_t *const *)_setup->huff_tables);
  if(ret<0){
    oc_state_clear(&_dec->state);
    return ret;
  }
  /*For each fragment, allocate one byte for every DCT coefficient token, plus
     one byte for extra-bits for each token, plus one more byte for the long
     EOB run, just in case it's the very last token and has a run length of
     one.*/
  _dec->dct_tokens=(unsigned char *)_ogg_malloc((64+64+1)*
   _dec->state.nfrags*sizeof(_dec->dct_tokens[0]));
  if(_dec->dct_tokens==NULL){
    oc_huff_trees_clear(_dec->huff_tables);
    oc_state_clear(&_dec->state);
    return TH_EFAULT;
  }
  for(qi=0;qi<64;qi++)for(pli=0;pli<3;pli++)for(qti=0;qti<2;qti++){
    _dec->state.dequant_tables[qi][pli][qti]=
     _dec->state.dequant_table_data[qi][pli][qti];
  }
  oc_dequant_tables_init(_dec->state.dequant_tables,_dec->pp_dc_scale,
   &_setup->qinfo);
  for(qi=0;qi<64;qi++){
    int qsum;
    qsum=0;
    for(qti=0;qti<2;qti++)for(pli=0;pli<3;pli++){
      qsum+=_dec->state.dequant_tables[qi][pli][qti][12]+
       _dec->state.dequant_tables[qi][pli][qti][17]+
       _dec->state.dequant_tables[qi][pli][qti][18]+
       _dec->state.dequant_tables[qi][pli][qti][24]<<(pli==0);
    }
    _dec->pp_sharp_mod[qi]=-(qsum>>11);
  }
  memcpy(_dec->state.loop_filter_limits,_setup->qinfo.loop_filter_limits,
   sizeof(_dec->state.loop_filter_limits));
  oc_dec_accel_init(_dec);
  _dec->pp_level=OC_PP_LEVEL_DISABLED;
  _dec->dc_qis=NULL;
  _dec->variances=NULL;
  _dec->pp_frame_data=NULL;
  _dec->stripe_cb.ctx=NULL;
  _dec->stripe_cb.stripe_decoded=NULL;
#if defined(HAVE_CAIRO)
  _dec->telemetry_bits=0;
  _dec->telemetry_qi=0;
  _dec->telemetry_mbmode=0;
  _dec->telemetry_mv=0;
  _dec->telemetry_frame_data=NULL;
#endif
  return 0;
}

static void oc_dec_clear(oc_dec_ctx *_dec){
#if defined(HAVE_CAIRO)
  _ogg_free(_dec->telemetry_frame_data);
#endif
  _ogg_free(_dec->pp_frame_data);
  _ogg_free(_dec->variances);
  _ogg_free(_dec->dc_qis);
  _ogg_free(_dec->dct_tokens);
  oc_huff_trees_clear(_dec->huff_tables);
  oc_state_clear(&_dec->state);
}


static int oc_dec_frame_header_unpack(oc_dec_ctx *_dec){
  long val;
  /*Check to make sure this is a data packet.*/
  val=oc_pack_read1(&_dec->opb);
  if(val!=0)return TH_EBADPACKET;
  /*Read in the frame type (I or P).*/
  val=oc_pack_read1(&_dec->opb);
  _dec->state.frame_type=(int)val;
  /*Read in the qi list.*/
  val=oc_pack_read(&_dec->opb,6);
  _dec->state.qis[0]=(unsigned char)val;
  val=oc_pack_read1(&_dec->opb);
  if(!val)_dec->state.nqis=1;
  else{
    val=oc_pack_read(&_dec->opb,6);
    _dec->state.qis[1]=(unsigned char)val;
    val=oc_pack_read1(&_dec->opb);
    if(!val)_dec->state.nqis=2;
    else{
      val=oc_pack_read(&_dec->opb,6);
      _dec->state.qis[2]=(unsigned char)val;
      _dec->state.nqis=3;
    }
  }
  if(_dec->state.frame_type==OC_INTRA_FRAME){
    /*Keyframes have 3 unused configuration bits, holdovers from VP3 days.
      Most of the other unused bits in the VP3 headers were eliminated.
      I don't know why these remain.*/
    /*I wanted to eliminate wasted bits, but not all config wiggle room
       --Monty.*/
    val=oc_pack_read(&_dec->opb,3);
    if(val!=0)return TH_EIMPL;
  }
  return 0;
}

/*Mark all fragments as coded and in OC_MODE_INTRA.
  This also builds up the coded fragment list (in coded order), and clears the
   uncoded fragment list.
  It does not update the coded macro block list nor the super block flags, as
   those are not used when decoding INTRA frames.*/
static void oc_dec_mark_all_intra(oc_dec_ctx *_dec){
  const oc_sb_map   *sb_maps;
  const oc_sb_flags *sb_flags;
  oc_fragment       *frags;
  ptrdiff_t         *coded_fragis;
  ptrdiff_t          ncoded_fragis;
  ptrdiff_t          prev_ncoded_fragis;
  unsigned           nsbs;
  unsigned           sbi;
  int                pli;
  coded_fragis=_dec->state.coded_fragis;
  prev_ncoded_fragis=ncoded_fragis=0;
  sb_maps=(const oc_sb_map *)_dec->state.sb_maps;
  sb_flags=_dec->state.sb_flags;
  frags=_dec->state.frags;
  sbi=nsbs=0;
  for(pli=0;pli<3;pli++){
    nsbs+=_dec->state.fplanes[pli].nsbs;
    for(;sbi<nsbs;sbi++){
      int quadi;
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        int bi;
        for(bi=0;bi<4;bi++){
          ptrdiff_t fragi;
          fragi=sb_maps[sbi][quadi][bi];
          if(fragi>=0){
            frags[fragi].coded=1;
            frags[fragi].refi=OC_FRAME_SELF;
            frags[fragi].mb_mode=OC_MODE_INTRA;
            coded_fragis[ncoded_fragis++]=fragi;
          }
        }
      }
    }
    _dec->state.ncoded_fragis[pli]=ncoded_fragis-prev_ncoded_fragis;
    prev_ncoded_fragis=ncoded_fragis;
  }
  _dec->state.ntotal_coded_fragis=ncoded_fragis;
}

/*Decodes the bit flags indicating whether each super block is partially coded
   or not.
  Return: The number of partially coded super blocks.*/
static unsigned oc_dec_partial_sb_flags_unpack(oc_dec_ctx *_dec){
  oc_sb_flags *sb_flags;
  unsigned     nsbs;
  unsigned     sbi;
  unsigned     npartial;
  unsigned     run_count;
  long         val;
  int          flag;
  val=oc_pack_read1(&_dec->opb);
  flag=(int)val;
  sb_flags=_dec->state.sb_flags;
  nsbs=_dec->state.nsbs;
  sbi=npartial=0;
  while(sbi<nsbs){
    int full_run;
    run_count=oc_sb_run_unpack(&_dec->opb);
    full_run=run_count>=4129;
    do{
      sb_flags[sbi].coded_partially=flag;
      sb_flags[sbi].coded_fully=0;
      npartial+=flag;
      sbi++;
    }
    while(--run_count>0&&sbi<nsbs);
    if(full_run&&sbi<nsbs){
      val=oc_pack_read1(&_dec->opb);
      flag=(int)val;
    }
    else flag=!flag;
  }
  /*TODO: run_count should be 0 here.
    If it's not, we should issue a warning of some kind.*/
  return npartial;
}

/*Decodes the bit flags for whether or not each non-partially-coded super
   block is fully coded or not.
  This function should only be called if there is at least one
   non-partially-coded super block.
  Return: The number of partially coded super blocks.*/
static void oc_dec_coded_sb_flags_unpack(oc_dec_ctx *_dec){
  oc_sb_flags *sb_flags;
  unsigned     nsbs;
  unsigned     sbi;
  unsigned     run_count;
  long         val;
  int          flag;
  sb_flags=_dec->state.sb_flags;
  nsbs=_dec->state.nsbs;
  /*Skip partially coded super blocks.*/
  for(sbi=0;sb_flags[sbi].coded_partially;sbi++);
  val=oc_pack_read1(&_dec->opb);
  flag=(int)val;
  do{
    int full_run;
    run_count=oc_sb_run_unpack(&_dec->opb);
    full_run=run_count>=4129;
    for(;sbi<nsbs;sbi++){
      if(sb_flags[sbi].coded_partially)continue;
      if(run_count--<=0)break;
      sb_flags[sbi].coded_fully=flag;
    }
    if(full_run&&sbi<nsbs){
      val=oc_pack_read1(&_dec->opb);
      flag=(int)val;
    }
    else flag=!flag;
  }
  while(sbi<nsbs);
  /*TODO: run_count should be 0 here.
    If it's not, we should issue a warning of some kind.*/
}

static void oc_dec_coded_flags_unpack(oc_dec_ctx *_dec){
  const oc_sb_map   *sb_maps;
  const oc_sb_flags *sb_flags;
  signed char       *mb_modes;
  oc_fragment       *frags;
  unsigned           nsbs;
  unsigned           sbi;
  unsigned           npartial;
  long               val;
  int                pli;
  int                flag;
  int                run_count;
  ptrdiff_t         *coded_fragis;
  ptrdiff_t         *uncoded_fragis;
  ptrdiff_t          ncoded_fragis;
  ptrdiff_t          nuncoded_fragis;
  ptrdiff_t          prev_ncoded_fragis;
  npartial=oc_dec_partial_sb_flags_unpack(_dec);
  if(npartial<_dec->state.nsbs)oc_dec_coded_sb_flags_unpack(_dec);
  if(npartial>0){
    val=oc_pack_read1(&_dec->opb);
    flag=!(int)val;
  }
  else flag=0;
  sb_maps=(const oc_sb_map *)_dec->state.sb_maps;
  sb_flags=_dec->state.sb_flags;
  mb_modes=_dec->state.mb_modes;
  frags=_dec->state.frags;
  sbi=nsbs=run_count=0;
  coded_fragis=_dec->state.coded_fragis;
  uncoded_fragis=coded_fragis+_dec->state.nfrags;
  prev_ncoded_fragis=ncoded_fragis=nuncoded_fragis=0;
  for(pli=0;pli<3;pli++){
    nsbs+=_dec->state.fplanes[pli].nsbs;
    for(;sbi<nsbs;sbi++){
      int quadi;
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        int quad_coded;
        int bi;
        quad_coded=0;
        for(bi=0;bi<4;bi++){
          ptrdiff_t fragi;
          fragi=sb_maps[sbi][quadi][bi];
          if(fragi>=0){
            int coded;
            if(sb_flags[sbi].coded_fully)coded=1;
            else if(!sb_flags[sbi].coded_partially)coded=0;
            else{
              if(run_count<=0){
                run_count=oc_block_run_unpack(&_dec->opb);
                flag=!flag;
              }
              run_count--;
              coded=flag;
            }
            if(coded)coded_fragis[ncoded_fragis++]=fragi;
            else *(uncoded_fragis-++nuncoded_fragis)=fragi;
            quad_coded|=coded;
            frags[fragi].coded=coded;
            frags[fragi].refi=OC_FRAME_NONE;
          }
        }
        /*Remember if there's a coded luma block in this macro block.*/
        if(!pli)mb_modes[sbi<<2|quadi]=quad_coded;
      }
    }
    _dec->state.ncoded_fragis[pli]=ncoded_fragis-prev_ncoded_fragis;
    prev_ncoded_fragis=ncoded_fragis;
  }
  _dec->state.ntotal_coded_fragis=ncoded_fragis;
  /*TODO: run_count should be 0 here.
    If it's not, we should issue a warning of some kind.*/
}


/*Coding scheme:
   Codeword            Mode Index
   0                       0
   10                      1
   110                     2
   1110                    3
   11110                   4
   111110                  5
   1111110                 6
   1111111                 7*/
static const ogg_int16_t OC_VLC_MODE_TREE[26]={
  4,
   -(1<<8|0),-(1<<8|0),-(1<<8|0),-(1<<8|0),
   -(1<<8|0),-(1<<8|0),-(1<<8|0),-(1<<8|0),
   -(2<<8|1),-(2<<8|1),-(2<<8|1),-(2<<8|1),
   -(3<<8|2),-(3<<8|2),-(4<<8|3),17,
    3,
     -(1<<8|4),-(1<<8|4),-(1<<8|4),-(1<<8|4),
     -(2<<8|5),-(2<<8|5),-(3<<8|6),-(3<<8|7)
};

static const ogg_int16_t OC_CLC_MODE_TREE[9]={
  3,
   -(3<<8|0),-(3<<8|1),-(3<<8|2),-(3<<8|3),
   -(3<<8|4),-(3<<8|5),-(3<<8|6),-(3<<8|7)
};

/*Unpacks the list of macro block modes for INTER frames.*/
static void oc_dec_mb_modes_unpack(oc_dec_ctx *_dec){
  signed char         *mb_modes;
  const unsigned char *alphabet;
  unsigned char        scheme0_alphabet[8];
  const ogg_int16_t   *mode_tree;
  size_t               nmbs;
  size_t               mbi;
  long                 val;
  int                  mode_scheme;
  val=oc_pack_read(&_dec->opb,3);
  mode_scheme=(int)val;
  if(mode_scheme==0){
    int mi;
    /*Just in case, initialize the modes to something.
      If the bitstream doesn't contain each index exactly once, it's likely
       corrupt and the rest of the packet is garbage anyway, but this way we
       won't crash, and we'll decode SOMETHING.*/
    /*LOOP VECTORIZES*/
    for(mi=0;mi<OC_NMODES;mi++)scheme0_alphabet[mi]=OC_MODE_INTER_NOMV;
    for(mi=0;mi<OC_NMODES;mi++){
      val=oc_pack_read(&_dec->opb,3);
      scheme0_alphabet[val]=OC_MODE_ALPHABETS[6][mi];
    }
    alphabet=scheme0_alphabet;
  }
  else alphabet=OC_MODE_ALPHABETS[mode_scheme-1];
  mode_tree=mode_scheme==7?OC_CLC_MODE_TREE:OC_VLC_MODE_TREE;
  mb_modes=_dec->state.mb_modes;
  nmbs=_dec->state.nmbs;
  for(mbi=0;mbi<nmbs;mbi++){
    if(mb_modes[mbi]>0){
      /*We have a coded luma block; decode a mode.*/
      mb_modes[mbi]=alphabet[oc_huff_token_decode(&_dec->opb,mode_tree)];
    }
    /*For other valid macro blocks, INTER_NOMV is forced, but we rely on the
       fact that OC_MODE_INTER_NOMV is already 0.*/
  }
}



static const ogg_int16_t OC_VLC_MV_COMP_TREE[101]={
  5,
   -(3<<8|32+0),-(3<<8|32+0),-(3<<8|32+0),-(3<<8|32+0),
   -(3<<8|32+1),-(3<<8|32+1),-(3<<8|32+1),-(3<<8|32+1),
   -(3<<8|32-1),-(3<<8|32-1),-(3<<8|32-1),-(3<<8|32-1),
   -(4<<8|32+2),-(4<<8|32+2),-(4<<8|32-2),-(4<<8|32-2),
   -(4<<8|32+3),-(4<<8|32+3),-(4<<8|32-3),-(4<<8|32-3),
   33,          36,          39,          42,
   45,          50,          55,          60,
   65,          74,          83,          92,
    1,-(1<<8|32+4),-(1<<8|32-4),
    1,-(1<<8|32+5),-(1<<8|32-5),
    1,-(1<<8|32+6),-(1<<8|32-6),
    1,-(1<<8|32+7),-(1<<8|32-7),
    2,-(2<<8|32+8),-(2<<8|32-8),-(2<<8|32+9),-(2<<8|32-9),
    2,-(2<<8|32+10),-(2<<8|32-10),-(2<<8|32+11),-(2<<8|32-11),
    2,-(2<<8|32+12),-(2<<8|32-12),-(2<<8|32+13),-(2<<8|32-13),
    2,-(2<<8|32+14),-(2<<8|32-14),-(2<<8|32+15),-(2<<8|32-15),
    3,
     -(3<<8|32+16),-(3<<8|32-16),-(3<<8|32+17),-(3<<8|32-17),
     -(3<<8|32+18),-(3<<8|32-18),-(3<<8|32+19),-(3<<8|32-19),
    3,
     -(3<<8|32+20),-(3<<8|32-20),-(3<<8|32+21),-(3<<8|32-21),
     -(3<<8|32+22),-(3<<8|32-22),-(3<<8|32+23),-(3<<8|32-23),
    3,
     -(3<<8|32+24),-(3<<8|32-24),-(3<<8|32+25),-(3<<8|32-25),
     -(3<<8|32+26),-(3<<8|32-26),-(3<<8|32+27),-(3<<8|32-27),
    3,
     -(3<<8|32+28),-(3<<8|32-28),-(3<<8|32+29),-(3<<8|32-29),
     -(3<<8|32+30),-(3<<8|32-30),-(3<<8|32+31),-(3<<8|32-31)
};

static const ogg_int16_t OC_CLC_MV_COMP_TREE[65]={
  6,
   -(6<<8|32 +0),-(6<<8|32 -0),-(6<<8|32 +1),-(6<<8|32 -1),
   -(6<<8|32 +2),-(6<<8|32 -2),-(6<<8|32 +3),-(6<<8|32 -3),
   -(6<<8|32 +4),-(6<<8|32 -4),-(6<<8|32 +5),-(6<<8|32 -5),
   -(6<<8|32 +6),-(6<<8|32 -6),-(6<<8|32 +7),-(6<<8|32 -7),
   -(6<<8|32 +8),-(6<<8|32 -8),-(6<<8|32 +9),-(6<<8|32 -9),
   -(6<<8|32+10),-(6<<8|32-10),-(6<<8|32+11),-(6<<8|32-11),
   -(6<<8|32+12),-(6<<8|32-12),-(6<<8|32+13),-(6<<8|32-13),
   -(6<<8|32+14),-(6<<8|32-14),-(6<<8|32+15),-(6<<8|32-15),
   -(6<<8|32+16),-(6<<8|32-16),-(6<<8|32+17),-(6<<8|32-17),
   -(6<<8|32+18),-(6<<8|32-18),-(6<<8|32+19),-(6<<8|32-19),
   -(6<<8|32+20),-(6<<8|32-20),-(6<<8|32+21),-(6<<8|32-21),
   -(6<<8|32+22),-(6<<8|32-22),-(6<<8|32+23),-(6<<8|32-23),
   -(6<<8|32+24),-(6<<8|32-24),-(6<<8|32+25),-(6<<8|32-25),
   -(6<<8|32+26),-(6<<8|32-26),-(6<<8|32+27),-(6<<8|32-27),
   -(6<<8|32+28),-(6<<8|32-28),-(6<<8|32+29),-(6<<8|32-29),
   -(6<<8|32+30),-(6<<8|32-30),-(6<<8|32+31),-(6<<8|32-31)
};


static oc_mv oc_mv_unpack(oc_pack_buf *_opb,const ogg_int16_t *_tree){
  int dx;
  int dy;
  dx=oc_huff_token_decode(_opb,_tree)-32;
  dy=oc_huff_token_decode(_opb,_tree)-32;
  return OC_MV(dx,dy);
}

/*Unpacks the list of motion vectors for INTER frames, and propagtes the macro
   block modes and motion vectors to the individual fragments.*/
static void oc_dec_mv_unpack_and_frag_modes_fill(oc_dec_ctx *_dec){
  const oc_mb_map        *mb_maps;
  const signed char      *mb_modes;
  oc_set_chroma_mvs_func  set_chroma_mvs;
  const ogg_int16_t      *mv_comp_tree;
  oc_fragment            *frags;
  oc_mv                  *frag_mvs;
  const unsigned char    *map_idxs;
  int                     map_nidxs;
  oc_mv                   last_mv;
  oc_mv                   prior_mv;
  oc_mv                   cbmvs[4];
  size_t                  nmbs;
  size_t                  mbi;
  long                    val;
  set_chroma_mvs=OC_SET_CHROMA_MVS_TABLE[_dec->state.info.pixel_fmt];
  val=oc_pack_read1(&_dec->opb);
  mv_comp_tree=val?OC_CLC_MV_COMP_TREE:OC_VLC_MV_COMP_TREE;
  map_idxs=OC_MB_MAP_IDXS[_dec->state.info.pixel_fmt];
  map_nidxs=OC_MB_MAP_NIDXS[_dec->state.info.pixel_fmt];
  prior_mv=last_mv=0;
  frags=_dec->state.frags;
  frag_mvs=_dec->state.frag_mvs;
  mb_maps=(const oc_mb_map *)_dec->state.mb_maps;
  mb_modes=_dec->state.mb_modes;
  nmbs=_dec->state.nmbs;
  for(mbi=0;mbi<nmbs;mbi++){
    int mb_mode;
    mb_mode=mb_modes[mbi];
    if(mb_mode!=OC_MODE_INVALID){
      oc_mv     mbmv;
      ptrdiff_t fragi;
      int       mapi;
      int       mapii;
      int       refi;
      if(mb_mode==OC_MODE_INTER_MV_FOUR){
        oc_mv lbmvs[4];
        int   bi;
        prior_mv=last_mv;
        for(bi=0;bi<4;bi++){
          fragi=mb_maps[mbi][0][bi];
          if(frags[fragi].coded){
            frags[fragi].refi=OC_FRAME_PREV;
            frags[fragi].mb_mode=OC_MODE_INTER_MV_FOUR;
            lbmvs[bi]=last_mv=oc_mv_unpack(&_dec->opb,mv_comp_tree);
            frag_mvs[fragi]=lbmvs[bi];
          }
          else lbmvs[bi]=0;
        }
        (*set_chroma_mvs)(cbmvs,lbmvs);
        for(mapii=4;mapii<map_nidxs;mapii++){
          mapi=map_idxs[mapii];
          bi=mapi&3;
          fragi=mb_maps[mbi][mapi>>2][bi];
          if(frags[fragi].coded){
            frags[fragi].refi=OC_FRAME_PREV;
            frags[fragi].mb_mode=OC_MODE_INTER_MV_FOUR;
            frag_mvs[fragi]=cbmvs[bi];
          }
        }
      }
      else{
        switch(mb_mode){
          case OC_MODE_INTER_MV:{
            prior_mv=last_mv;
            last_mv=mbmv=oc_mv_unpack(&_dec->opb,mv_comp_tree);
          }break;
          case OC_MODE_INTER_MV_LAST:mbmv=last_mv;break;
          case OC_MODE_INTER_MV_LAST2:{
            mbmv=prior_mv;
            prior_mv=last_mv;
            last_mv=mbmv;
          }break;
          case OC_MODE_GOLDEN_MV:{
            mbmv=oc_mv_unpack(&_dec->opb,mv_comp_tree);
          }break;
          default:mbmv=0;break;
        }
        /*Fill in the MVs for the fragments.*/
        refi=OC_FRAME_FOR_MODE(mb_mode);
        mapii=0;
        do{
          mapi=map_idxs[mapii];
          fragi=mb_maps[mbi][mapi>>2][mapi&3];
          if(frags[fragi].coded){
            frags[fragi].refi=refi;
            frags[fragi].mb_mode=mb_mode;
            frag_mvs[fragi]=mbmv;
          }
        }
        while(++mapii<map_nidxs);
      }
    }
  }
}

static void oc_dec_block_qis_unpack(oc_dec_ctx *_dec){
  oc_fragment     *frags;
  const ptrdiff_t *coded_fragis;
  ptrdiff_t        ncoded_fragis;
  ptrdiff_t        fragii;
  ptrdiff_t        fragi;
  ncoded_fragis=_dec->state.ntotal_coded_fragis;
  if(ncoded_fragis<=0)return;
  frags=_dec->state.frags;
  coded_fragis=_dec->state.coded_fragis;
  if(_dec->state.nqis==1){
    /*If this frame has only a single qi value, then just use it for all coded
       fragments.*/
    for(fragii=0;fragii<ncoded_fragis;fragii++){
      frags[coded_fragis[fragii]].qii=0;
    }
  }
  else{
    long val;
    int  flag;
    int  nqi1;
    int  run_count;
    /*Otherwise, we decode a qi index for each fragment, using two passes of
      the same binary RLE scheme used for super-block coded bits.
     The first pass marks each fragment as having a qii of 0 or greater than
      0, and the second pass (if necessary), distinguishes between a qii of
      1 and 2.
     At first we just store the qii in the fragment.
     After all the qii's are decoded, we make a final pass to replace them
      with the corresponding qi's for this frame.*/
    val=oc_pack_read1(&_dec->opb);
    flag=(int)val;
    nqi1=0;
    fragii=0;
    while(fragii<ncoded_fragis){
      int full_run;
      run_count=oc_sb_run_unpack(&_dec->opb);
      full_run=run_count>=4129;
      do{
        frags[coded_fragis[fragii++]].qii=flag;
        nqi1+=flag;
      }
      while(--run_count>0&&fragii<ncoded_fragis);
      if(full_run&&fragii<ncoded_fragis){
        val=oc_pack_read1(&_dec->opb);
        flag=(int)val;
      }
      else flag=!flag;
    }
    /*TODO: run_count should be 0 here.
      If it's not, we should issue a warning of some kind.*/
    /*If we have 3 different qi's for this frame, and there was at least one
       fragment with a non-zero qi, make the second pass.*/
    if(_dec->state.nqis==3&&nqi1>0){
      /*Skip qii==0 fragments.*/
      for(fragii=0;frags[coded_fragis[fragii]].qii==0;fragii++);
      val=oc_pack_read1(&_dec->opb);
      flag=(int)val;
      do{
        int full_run;
        run_count=oc_sb_run_unpack(&_dec->opb);
        full_run=run_count>=4129;
        for(;fragii<ncoded_fragis;fragii++){
          fragi=coded_fragis[fragii];
          if(frags[fragi].qii==0)continue;
          if(run_count--<=0)break;
          frags[fragi].qii+=flag;
        }
        if(full_run&&fragii<ncoded_fragis){
          val=oc_pack_read1(&_dec->opb);
          flag=(int)val;
        }
        else flag=!flag;
      }
      while(fragii<ncoded_fragis);
      /*TODO: run_count should be 0 here.
        If it's not, we should issue a warning of some kind.*/
    }
  }
}



/*Unpacks the DC coefficient tokens.
  Unlike when unpacking the AC coefficient tokens, we actually need to decode
   the DC coefficient values now so that we can do DC prediction.
  _huff_idx:   The index of the Huffman table to use for each color plane.
  _ntoks_left: The number of tokens left to be decoded in each color plane for
                each coefficient.
               This is updated as EOB tokens and zero run tokens are decoded.
  Return: The length of any outstanding EOB run.*/
static ptrdiff_t oc_dec_dc_coeff_unpack(oc_dec_ctx *_dec,int _huff_idxs[2],
 ptrdiff_t _ntoks_left[3][64]){
  unsigned char   *dct_tokens;
  oc_fragment     *frags;
  const ptrdiff_t *coded_fragis;
  ptrdiff_t        ncoded_fragis;
  ptrdiff_t        fragii;
  ptrdiff_t        eobs;
  ptrdiff_t        ti;
  int              pli;
  dct_tokens=_dec->dct_tokens;
  frags=_dec->state.frags;
  coded_fragis=_dec->state.coded_fragis;
  ncoded_fragis=fragii=eobs=ti=0;
  for(pli=0;pli<3;pli++){
    ptrdiff_t run_counts[64];
    ptrdiff_t eob_count;
    ptrdiff_t eobi;
    int       rli;
    ncoded_fragis+=_dec->state.ncoded_fragis[pli];
    memset(run_counts,0,sizeof(run_counts));
    _dec->eob_runs[pli][0]=eobs;
    _dec->ti0[pli][0]=ti;
    /*Continue any previous EOB run, if there was one.*/
    eobi=eobs;
    if(ncoded_fragis-fragii<eobi)eobi=ncoded_fragis-fragii;
    eob_count=eobi;
    eobs-=eobi;
    while(eobi-->0)frags[coded_fragis[fragii++]].dc=0;
    while(fragii<ncoded_fragis){
      int token;
      int cw;
      int eb;
      int skip;
      token=oc_huff_token_decode(&_dec->opb,
       _dec->huff_tables[_huff_idxs[pli+1>>1]]);
      dct_tokens[ti++]=(unsigned char)token;
      if(OC_DCT_TOKEN_NEEDS_MORE(token)){
        eb=(int)oc_pack_read(&_dec->opb,
         OC_INTERNAL_DCT_TOKEN_EXTRA_BITS[token]);
        dct_tokens[ti++]=(unsigned char)eb;
        if(token==OC_DCT_TOKEN_FAT_EOB)dct_tokens[ti++]=(unsigned char)(eb>>8);
        eb<<=OC_DCT_TOKEN_EB_POS(token);
      }
      else eb=0;
      cw=OC_DCT_CODE_WORD[token]+eb;
      eobs=cw>>OC_DCT_CW_EOB_SHIFT&0xFFF;
      if(cw==OC_DCT_CW_FINISH)eobs=OC_DCT_EOB_FINISH;
      if(eobs){
        eobi=OC_MINI(eobs,ncoded_fragis-fragii);
        eob_count+=eobi;
        eobs-=eobi;
        while(eobi-->0)frags[coded_fragis[fragii++]].dc=0;
      }
      else{
        int coeff;
        skip=(unsigned char)(cw>>OC_DCT_CW_RLEN_SHIFT);
        cw^=-(cw&1<<OC_DCT_CW_FLIP_BIT);
        coeff=cw>>OC_DCT_CW_MAG_SHIFT;
        if(skip)coeff=0;
        run_counts[skip]++;
        frags[coded_fragis[fragii++]].dc=coeff;
      }
    }
    /*Add the total EOB count to the longest run length.*/
    run_counts[63]+=eob_count;
    /*And convert the run_counts array to a moment table.*/
    for(rli=63;rli-->0;)run_counts[rli]+=run_counts[rli+1];
    /*Finally, subtract off the number of coefficients that have been
       accounted for by runs started in this coefficient.*/
    for(rli=64;rli-->0;)_ntoks_left[pli][rli]-=run_counts[rli];
  }
  _dec->dct_tokens_count=ti;
  return eobs;
}

/*Unpacks the AC coefficient tokens.
  This can completely discard coefficient values while unpacking, and so is
   somewhat simpler than unpacking the DC coefficient tokens.
  _huff_idx:   The index of the Huffman table to use for each color plane.
  _ntoks_left: The number of tokens left to be decoded in each color plane for
                each coefficient.
               This is updated as EOB tokens and zero run tokens are decoded.
  _eobs:       The length of any outstanding EOB run from previous
                coefficients.
  Return: The length of any outstanding EOB run.*/
static int oc_dec_ac_coeff_unpack(oc_dec_ctx *_dec,int _zzi,int _huff_idxs[2],
 ptrdiff_t _ntoks_left[3][64],ptrdiff_t _eobs){
  unsigned char *dct_tokens;
  ptrdiff_t      ti;
  int            pli;
  dct_tokens=_dec->dct_tokens;
  ti=_dec->dct_tokens_count;
  for(pli=0;pli<3;pli++){
    ptrdiff_t run_counts[64];
    ptrdiff_t eob_count;
    size_t    ntoks_left;
    size_t    ntoks;
    int       rli;
    _dec->eob_runs[pli][_zzi]=_eobs;
    _dec->ti0[pli][_zzi]=ti;
    ntoks_left=_ntoks_left[pli][_zzi];
    memset(run_counts,0,sizeof(run_counts));
    eob_count=0;
    ntoks=0;
    while(ntoks+_eobs<ntoks_left){
      int token;
      int cw;
      int eb;
      int skip;
      ntoks+=_eobs;
      eob_count+=_eobs;
      token=oc_huff_token_decode(&_dec->opb,
       _dec->huff_tables[_huff_idxs[pli+1>>1]]);
      dct_tokens[ti++]=(unsigned char)token;
      if(OC_DCT_TOKEN_NEEDS_MORE(token)){
        eb=(int)oc_pack_read(&_dec->opb,
         OC_INTERNAL_DCT_TOKEN_EXTRA_BITS[token]);
        dct_tokens[ti++]=(unsigned char)eb;
        if(token==OC_DCT_TOKEN_FAT_EOB)dct_tokens[ti++]=(unsigned char)(eb>>8);
        eb<<=OC_DCT_TOKEN_EB_POS(token);
      }
      else eb=0;
      cw=OC_DCT_CODE_WORD[token]+eb;
      skip=(unsigned char)(cw>>OC_DCT_CW_RLEN_SHIFT);
      _eobs=cw>>OC_DCT_CW_EOB_SHIFT&0xFFF;
      if(cw==OC_DCT_CW_FINISH)_eobs=OC_DCT_EOB_FINISH;
      if(_eobs==0){
        run_counts[skip]++;
        ntoks++;
      }
    }
    /*Add the portion of the last EOB run actually used by this coefficient.*/
    eob_count+=ntoks_left-ntoks;
    /*And remove it from the remaining EOB count.*/
    _eobs-=ntoks_left-ntoks;
    /*Add the total EOB count to the longest run length.*/
    run_counts[63]+=eob_count;
    /*And convert the run_counts array to a moment table.*/
    for(rli=63;rli-->0;)run_counts[rli]+=run_counts[rli+1];
    /*Finally, subtract off the number of coefficients that have been
       accounted for by runs started in this coefficient.*/
    for(rli=64-_zzi;rli-->0;)_ntoks_left[pli][_zzi+rli]-=run_counts[rli];
  }
  _dec->dct_tokens_count=ti;
  return _eobs;
}

/*Tokens describing the DCT coefficients that belong to each fragment are
   stored in the bitstream grouped by coefficient, not by fragment.

  This means that we either decode all the tokens in order, building up a
   separate coefficient list for each fragment as we go, and then go back and
   do the iDCT on each fragment, or we have to create separate lists of tokens
   for each coefficient, so that we can pull the next token required off the
   head of the appropriate list when decoding a specific fragment.

  The former was VP3's choice, and it meant 2*w*h extra storage for all the
   decoded coefficient values.

  We take the second option, which lets us store just one to three bytes per
   token (generally far fewer than the number of coefficients, due to EOB
   tokens and zero runs), and which requires us to only maintain a counter for
   each of the 64 coefficients, instead of a counter for every fragment to
   determine where the next token goes.

  We actually use 3 counters per coefficient, one for each color plane, so we
   can decode all color planes simultaneously.
  This lets color conversion, etc., be done as soon as a full MCU (one or
   two super block rows) is decoded, while the image data is still in cache.*/

static void oc_dec_residual_tokens_unpack(oc_dec_ctx *_dec){
  static const unsigned char OC_HUFF_LIST_MAX[5]={1,6,15,28,64};
  ptrdiff_t  ntoks_left[3][64];
  int        huff_idxs[2];
  ptrdiff_t  eobs;
  long       val;
  int        pli;
  int        zzi;
  int        hgi;
  for(pli=0;pli<3;pli++)for(zzi=0;zzi<64;zzi++){
    ntoks_left[pli][zzi]=_dec->state.ncoded_fragis[pli];
  }
  val=oc_pack_read(&_dec->opb,4);
  huff_idxs[0]=(int)val;
  val=oc_pack_read(&_dec->opb,4);
  huff_idxs[1]=(int)val;
  _dec->eob_runs[0][0]=0;
  eobs=oc_dec_dc_coeff_unpack(_dec,huff_idxs,ntoks_left);
#if defined(HAVE_CAIRO)
  _dec->telemetry_dc_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
  val=oc_pack_read(&_dec->opb,4);
  huff_idxs[0]=(int)val;
  val=oc_pack_read(&_dec->opb,4);
  huff_idxs[1]=(int)val;
  zzi=1;
  for(hgi=1;hgi<5;hgi++){
    huff_idxs[0]+=16;
    huff_idxs[1]+=16;
    for(;zzi<OC_HUFF_LIST_MAX[hgi];zzi++){
      eobs=oc_dec_ac_coeff_unpack(_dec,zzi,huff_idxs,ntoks_left,eobs);
    }
  }
  /*TODO: eobs should be exactly zero, or 4096 or greater.
    The second case occurs when an EOB run of size zero is encountered, which
     gets treated as an infinite EOB run (where infinity is PTRDIFF_MAX).
    If neither of these conditions holds, then a warning should be issued.*/
}


static int oc_dec_postprocess_init(oc_dec_ctx *_dec){
  /*musl libc malloc()/realloc() calls might use floating point, so make sure
     we've cleared the MMX state for them.*/
  oc_restore_fpu(&_dec->state);
  /*pp_level 0: disabled; free any memory used and return*/
  if(_dec->pp_level<=OC_PP_LEVEL_DISABLED){
    if(_dec->dc_qis!=NULL){
      _ogg_free(_dec->dc_qis);
      _dec->dc_qis=NULL;
      _ogg_free(_dec->variances);
      _dec->variances=NULL;
      _ogg_free(_dec->pp_frame_data);
      _dec->pp_frame_data=NULL;
    }
    return 1;
  }
  if(_dec->dc_qis==NULL){
    /*If we haven't been tracking DC quantization indices, there's no point in
       starting now.*/
    if(_dec->state.frame_type!=OC_INTRA_FRAME)return 1;
    _dec->dc_qis=(unsigned char *)_ogg_malloc(
     _dec->state.nfrags*sizeof(_dec->dc_qis[0]));
    if(_dec->dc_qis==NULL)return 1;
    memset(_dec->dc_qis,_dec->state.qis[0],_dec->state.nfrags);
  }
  else{
    unsigned char   *dc_qis;
    const ptrdiff_t *coded_fragis;
    ptrdiff_t        ncoded_fragis;
    ptrdiff_t        fragii;
    unsigned char    qi0;
    /*Update the DC quantization index of each coded block.*/
    dc_qis=_dec->dc_qis;
    coded_fragis=_dec->state.coded_fragis;
    ncoded_fragis=_dec->state.ncoded_fragis[0]+
     _dec->state.ncoded_fragis[1]+_dec->state.ncoded_fragis[2];
    qi0=(unsigned char)_dec->state.qis[0];
    for(fragii=0;fragii<ncoded_fragis;fragii++){
      dc_qis[coded_fragis[fragii]]=qi0;
    }
  }
  /*pp_level 1: Stop after updating DC quantization indices.*/
  if(_dec->pp_level<=OC_PP_LEVEL_TRACKDCQI){
    if(_dec->variances!=NULL){
      _ogg_free(_dec->variances);
      _dec->variances=NULL;
      _ogg_free(_dec->pp_frame_data);
      _dec->pp_frame_data=NULL;
    }
    return 1;
  }
  if(_dec->variances==NULL){
    size_t frame_sz;
    size_t c_sz;
    int    c_w;
    int    c_h;
    frame_sz=_dec->state.info.frame_width*(size_t)_dec->state.info.frame_height;
    c_w=_dec->state.info.frame_width>>!(_dec->state.info.pixel_fmt&1);
    c_h=_dec->state.info.frame_height>>!(_dec->state.info.pixel_fmt&2);
    c_sz=c_w*(size_t)c_h;
    /*Allocate space for the chroma planes, even if we're not going to use
       them; this simplifies allocation state management, though it may waste
       memory on the few systems that don't overcommit pages.*/
    frame_sz+=c_sz<<1;
    _dec->pp_frame_data=(unsigned char *)_ogg_malloc(
     frame_sz*sizeof(_dec->pp_frame_data[0]));
    _dec->variances=(int *)_ogg_malloc(
     _dec->state.nfrags*sizeof(_dec->variances[0]));
    if(_dec->variances==NULL||_dec->pp_frame_data==NULL){
      _ogg_free(_dec->pp_frame_data);
      _dec->pp_frame_data=NULL;
      _ogg_free(_dec->variances);
      _dec->variances=NULL;
      return 1;
    }
    /*Force an update of the PP buffer pointers.*/
    _dec->pp_frame_state=0;
  }
  /*Update the PP buffer pointers if necessary.*/
  if(_dec->pp_frame_state!=1+(_dec->pp_level>=OC_PP_LEVEL_DEBLOCKC)){
    if(_dec->pp_level<OC_PP_LEVEL_DEBLOCKC){
      /*If chroma processing is disabled, just use the PP luma plane.*/
      _dec->pp_frame_buf[0].width=_dec->state.info.frame_width;
      _dec->pp_frame_buf[0].height=_dec->state.info.frame_height;
      _dec->pp_frame_buf[0].stride=-_dec->pp_frame_buf[0].width;
      _dec->pp_frame_buf[0].data=_dec->pp_frame_data+
       (1-_dec->pp_frame_buf[0].height)*(ptrdiff_t)_dec->pp_frame_buf[0].stride;
    }
    else{
      size_t y_sz;
      size_t c_sz;
      int    c_w;
      int    c_h;
      /*Otherwise, set up pointers to all three PP planes.*/
      y_sz=_dec->state.info.frame_width*(size_t)_dec->state.info.frame_height;
      c_w=_dec->state.info.frame_width>>!(_dec->state.info.pixel_fmt&1);
      c_h=_dec->state.info.frame_height>>!(_dec->state.info.pixel_fmt&2);
      c_sz=c_w*(size_t)c_h;
      _dec->pp_frame_buf[0].width=_dec->state.info.frame_width;
      _dec->pp_frame_buf[0].height=_dec->state.info.frame_height;
      _dec->pp_frame_buf[0].stride=_dec->pp_frame_buf[0].width;
      _dec->pp_frame_buf[0].data=_dec->pp_frame_data;
      _dec->pp_frame_buf[1].width=c_w;
      _dec->pp_frame_buf[1].height=c_h;
      _dec->pp_frame_buf[1].stride=_dec->pp_frame_buf[1].width;
      _dec->pp_frame_buf[1].data=_dec->pp_frame_buf[0].data+y_sz;
      _dec->pp_frame_buf[2].width=c_w;
      _dec->pp_frame_buf[2].height=c_h;
      _dec->pp_frame_buf[2].stride=_dec->pp_frame_buf[2].width;
      _dec->pp_frame_buf[2].data=_dec->pp_frame_buf[1].data+c_sz;
      oc_ycbcr_buffer_flip(_dec->pp_frame_buf,_dec->pp_frame_buf);
    }
    _dec->pp_frame_state=1+(_dec->pp_level>=OC_PP_LEVEL_DEBLOCKC);
  }
  /*If we're not processing chroma, copy the reference frame's chroma planes.*/
  if(_dec->pp_level<OC_PP_LEVEL_DEBLOCKC){
    memcpy(_dec->pp_frame_buf+1,
     _dec->state.ref_frame_bufs[_dec->state.ref_frame_idx[OC_FRAME_SELF]]+1,
     sizeof(_dec->pp_frame_buf[1])*2);
  }
  return 0;
}


/*Initialize the main decoding pipeline.*/
static void oc_dec_pipeline_init(oc_dec_ctx *_dec,
 oc_dec_pipeline_state *_pipe){
  const ptrdiff_t *coded_fragis;
  const ptrdiff_t *uncoded_fragis;
  int              flimit;
  int              pli;
  int              qii;
  int              qti;
  int              zzi;
  /*If chroma is sub-sampled in the vertical direction, we have to decode two
     super block rows of Y' for each super block row of Cb and Cr.*/
  _pipe->mcu_nvfrags=4<<!(_dec->state.info.pixel_fmt&2);
  /*Initialize the token and extra bits indices for each plane and
     coefficient.*/
  memcpy(_pipe->ti,_dec->ti0,sizeof(_pipe->ti));
  /*Also copy over the initial the EOB run counts.*/
  memcpy(_pipe->eob_runs,_dec->eob_runs,sizeof(_pipe->eob_runs));
  /*Set up per-plane pointers to the coded and uncoded fragments lists.*/
  coded_fragis=_dec->state.coded_fragis;
  uncoded_fragis=coded_fragis+_dec->state.nfrags;
  for(pli=0;pli<3;pli++){
    ptrdiff_t ncoded_fragis;
    _pipe->coded_fragis[pli]=coded_fragis;
    _pipe->uncoded_fragis[pli]=uncoded_fragis;
    ncoded_fragis=_dec->state.ncoded_fragis[pli];
    coded_fragis+=ncoded_fragis;
    uncoded_fragis+=ncoded_fragis-_dec->state.fplanes[pli].nfrags;
  }
  /*Set up condensed quantizer tables.*/
  for(pli=0;pli<3;pli++){
    for(qii=0;qii<_dec->state.nqis;qii++){
      for(qti=0;qti<2;qti++){
        _pipe->dequant[pli][qii][qti]=
         _dec->state.dequant_tables[_dec->state.qis[qii]][pli][qti];
      }
    }
  }
  /*Set the previous DC predictor to 0 for all color planes and frame types.*/
  memset(_pipe->pred_last,0,sizeof(_pipe->pred_last));
  /*Initialize the bounding value array for the loop filter.*/
  flimit=_dec->state.loop_filter_limits[_dec->state.qis[0]];
  _pipe->loop_filter=flimit!=0;
  if(flimit!=0)oc_loop_filter_init(&_dec->state,_pipe->bounding_values,flimit);
  /*Initialize any buffers needed for post-processing.
    We also save the current post-processing level, to guard against the user
     changing it from a callback.*/
  if(!oc_dec_postprocess_init(_dec))_pipe->pp_level=_dec->pp_level;
  /*If we don't have enough information to post-process, disable it, regardless
     of the user-requested level.*/
  else{
    _pipe->pp_level=OC_PP_LEVEL_DISABLED;
    memcpy(_dec->pp_frame_buf,
     _dec->state.ref_frame_bufs[_dec->state.ref_frame_idx[OC_FRAME_SELF]],
     sizeof(_dec->pp_frame_buf[0])*3);
  }
  /*Clear down the DCT coefficient buffer for the first block.*/
  for(zzi=0;zzi<64;zzi++)_pipe->dct_coeffs[zzi]=0;
}

/*Undo the DC prediction in a single plane of an MCU (one or two super block
   rows).
  As a side effect, the number of coded and uncoded fragments in this plane of
   the MCU is also computed.*/
void oc_dec_dc_unpredict_mcu_plane_c(oc_dec_ctx *_dec,
 oc_dec_pipeline_state *_pipe,int _pli){
  const oc_fragment_plane *fplane;
  oc_fragment             *frags;
  int                     *pred_last;
  ptrdiff_t                ncoded_fragis;
  ptrdiff_t                fragi;
  int                      fragx;
  int                      fragy;
  int                      fragy0;
  int                      fragy_end;
  int                      nhfrags;
  /*Compute the first and last fragment row of the current MCU for this
     plane.*/
  fplane=_dec->state.fplanes+_pli;
  fragy0=_pipe->fragy0[_pli];
  fragy_end=_pipe->fragy_end[_pli];
  nhfrags=fplane->nhfrags;
  pred_last=_pipe->pred_last[_pli];
  frags=_dec->state.frags;
  ncoded_fragis=0;
  fragi=fplane->froffset+fragy0*(ptrdiff_t)nhfrags;
  for(fragy=fragy0;fragy<fragy_end;fragy++){
    if(fragy==0){
      /*For the first row, all of the cases reduce to just using the previous
         predictor for the same reference frame.*/
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        if(frags[fragi].coded){
          int refi;
          refi=frags[fragi].refi;
          pred_last[refi]=frags[fragi].dc+=pred_last[refi];
          ncoded_fragis++;
        }
      }
    }
    else{
      oc_fragment *u_frags;
      int          l_ref;
      int          ul_ref;
      int          u_ref;
      u_frags=frags-nhfrags;
      l_ref=-1;
      ul_ref=-1;
      u_ref=u_frags[fragi].refi;
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        int ur_ref;
        if(fragx+1>=nhfrags)ur_ref=-1;
        else ur_ref=u_frags[fragi+1].refi;
        if(frags[fragi].coded){
          int pred;
          int refi;
          refi=frags[fragi].refi;
          /*We break out a separate case based on which of our neighbors use
             the same reference frames.
            This is somewhat faster than trying to make a generic case which
             handles all of them, since it reduces lots of poorly predicted
             jumps to one switch statement, and also lets a number of the
             multiplications be optimized out by strength reduction.*/
          switch((l_ref==refi)|(ul_ref==refi)<<1|
           (u_ref==refi)<<2|(ur_ref==refi)<<3){
            default:pred=pred_last[refi];break;
            case  1:
            case  3:pred=frags[fragi-1].dc;break;
            case  2:pred=u_frags[fragi-1].dc;break;
            case  4:
            case  6:
            case 12:pred=u_frags[fragi].dc;break;
            case  5:pred=(frags[fragi-1].dc+u_frags[fragi].dc)/2;break;
            case  8:pred=u_frags[fragi+1].dc;break;
            case  9:
            case 11:
            case 13:{
              /*The TI compiler mis-compiles this line.*/
              pred=(75*frags[fragi-1].dc+53*u_frags[fragi+1].dc)/128;
            }break;
            case 10:pred=(u_frags[fragi-1].dc+u_frags[fragi+1].dc)/2;break;
            case 14:{
              pred=(3*(u_frags[fragi-1].dc+u_frags[fragi+1].dc)
               +10*u_frags[fragi].dc)/16;
            }break;
            case  7:
            case 15:{
              int p0;
              int p1;
              int p2;
              p0=frags[fragi-1].dc;
              p1=u_frags[fragi-1].dc;
              p2=u_frags[fragi].dc;
              pred=(29*(p0+p2)-26*p1)/32;
              if(abs(pred-p2)>128)pred=p2;
              else if(abs(pred-p0)>128)pred=p0;
              else if(abs(pred-p1)>128)pred=p1;
            }break;
          }
          pred_last[refi]=frags[fragi].dc+=pred;
          ncoded_fragis++;
          l_ref=refi;
        }
        else l_ref=-1;
        ul_ref=u_ref;
        u_ref=ur_ref;
      }
    }
  }
  _pipe->ncoded_fragis[_pli]=ncoded_fragis;
  /*Also save the number of uncoded fragments so we know how many to copy.*/
  _pipe->nuncoded_fragis[_pli]=
   (fragy_end-fragy0)*(ptrdiff_t)nhfrags-ncoded_fragis;
}

/*Reconstructs all coded fragments in a single MCU (one or two super block
   rows).
  This requires that each coded fragment have a proper macro block mode and
   motion vector (if not in INTRA mode), and have its DC value decoded, with
   the DC prediction process reversed, and the number of coded and uncoded
   fragments in this plane of the MCU be counted.
  The token lists for each color plane and coefficient should also be filled
   in, along with initial token offsets, extra bits offsets, and EOB run
   counts.*/
static void oc_dec_frags_recon_mcu_plane(oc_dec_ctx *_dec,
 oc_dec_pipeline_state *_pipe,int _pli){
  unsigned char       *dct_tokens;
  const unsigned char *dct_fzig_zag;
  ogg_uint16_t         dc_quant[2];
  const oc_fragment   *frags;
  const ptrdiff_t     *coded_fragis;
  ptrdiff_t            ncoded_fragis;
  ptrdiff_t            fragii;
  ptrdiff_t           *ti;
  ptrdiff_t           *eob_runs;
  int                  qti;
  dct_tokens=_dec->dct_tokens;
  dct_fzig_zag=_dec->state.opt_data.dct_fzig_zag;
  frags=_dec->state.frags;
  coded_fragis=_pipe->coded_fragis[_pli];
  ncoded_fragis=_pipe->ncoded_fragis[_pli];
  ti=_pipe->ti[_pli];
  eob_runs=_pipe->eob_runs[_pli];
  for(qti=0;qti<2;qti++)dc_quant[qti]=_pipe->dequant[_pli][0][qti][0];
  for(fragii=0;fragii<ncoded_fragis;fragii++){
    const ogg_uint16_t *ac_quant;
    ptrdiff_t           fragi;
    int                 last_zzi;
    int                 zzi;
    fragi=coded_fragis[fragii];
    qti=frags[fragi].mb_mode!=OC_MODE_INTRA;
    ac_quant=_pipe->dequant[_pli][frags[fragi].qii][qti];
    /*Decode the AC coefficients.*/
    for(zzi=0;zzi<64;){
      int token;
      last_zzi=zzi;
      if(eob_runs[zzi]){
        eob_runs[zzi]--;
        break;
      }
      else{
        ptrdiff_t eob;
        int       cw;
        int       rlen;
        int       coeff;
        int       lti;
        lti=ti[zzi];
        token=dct_tokens[lti++];
        cw=OC_DCT_CODE_WORD[token];
        /*These parts could be done branchless, but the branches are fairly
           predictable and the C code translates into more than a few
           instructions, so it's worth it to avoid them.*/
        if(OC_DCT_TOKEN_NEEDS_MORE(token)){
          cw+=dct_tokens[lti++]<<OC_DCT_TOKEN_EB_POS(token);
        }
        eob=cw>>OC_DCT_CW_EOB_SHIFT&0xFFF;
        if(token==OC_DCT_TOKEN_FAT_EOB){
          eob+=dct_tokens[lti++]<<8;
          if(eob==0)eob=OC_DCT_EOB_FINISH;
        }
        rlen=(unsigned char)(cw>>OC_DCT_CW_RLEN_SHIFT);
        cw^=-(cw&1<<OC_DCT_CW_FLIP_BIT);
        coeff=cw>>OC_DCT_CW_MAG_SHIFT;
        eob_runs[zzi]=eob;
        ti[zzi]=lti;
        zzi+=rlen;
        _pipe->dct_coeffs[dct_fzig_zag[zzi]]=
         (ogg_int16_t)(coeff*(int)ac_quant[zzi]);
        zzi+=!eob;
      }
    }
    /*TODO: zzi should be exactly 64 here.
      If it's not, we should report some kind of warning.*/
    zzi=OC_MINI(zzi,64);
    _pipe->dct_coeffs[0]=(ogg_int16_t)frags[fragi].dc;
    /*last_zzi is always initialized.
      If your compiler thinks otherwise, it is dumb.*/
    oc_state_frag_recon(&_dec->state,fragi,_pli,
     _pipe->dct_coeffs,last_zzi,dc_quant[qti]);
  }
  _pipe->coded_fragis[_pli]+=ncoded_fragis;
  /*Right now the reconstructed MCU has only the coded blocks in it.*/
  /*TODO: We make the decision here to always copy the uncoded blocks into it
     from the reference frame.
    We could also copy the coded blocks back over the reference frame, if we
     wait for an additional MCU to be decoded, which might be faster if only a
     small number of blocks are coded.
    However, this introduces more latency, creating a larger cache footprint.
    It's unknown which decision is better, but this one results in simpler
     code, and the hard case (high bitrate, high resolution) is handled
     correctly.*/
  /*Copy the uncoded blocks from the previous reference frame.*/
  if(_pipe->nuncoded_fragis[_pli]>0){
    _pipe->uncoded_fragis[_pli]-=_pipe->nuncoded_fragis[_pli];
    oc_frag_copy_list(&_dec->state,
     _dec->state.ref_frame_data[OC_FRAME_SELF],
     _dec->state.ref_frame_data[OC_FRAME_PREV],
     _dec->state.ref_ystride[_pli],_pipe->uncoded_fragis[_pli],
     _pipe->nuncoded_fragis[_pli],_dec->state.frag_buf_offs);
  }
}

/*Filter a horizontal block edge.*/
static void oc_filter_hedge(unsigned char *_dst,int _dst_ystride,
 const unsigned char *_src,int _src_ystride,int _qstep,int _flimit,
 int *_variance0,int *_variance1){
  unsigned char       *rdst;
  const unsigned char *rsrc;
  unsigned char       *cdst;
  const unsigned char *csrc;
  int                  r[10];
  int                  sum0;
  int                  sum1;
  int                  bx;
  int                  by;
  rdst=_dst;
  rsrc=_src;
  for(bx=0;bx<8;bx++){
    cdst=rdst;
    csrc=rsrc;
    for(by=0;by<10;by++){
      r[by]=*csrc;
      csrc+=_src_ystride;
    }
    sum0=sum1=0;
    for(by=0;by<4;by++){
      sum0+=abs(r[by+1]-r[by]);
      sum1+=abs(r[by+5]-r[by+6]);
    }
    *_variance0+=OC_MINI(255,sum0);
    *_variance1+=OC_MINI(255,sum1);
    if(sum0<_flimit&&sum1<_flimit&&r[5]-r[4]<_qstep&&r[4]-r[5]<_qstep){
      *cdst=(unsigned char)(r[0]*3+r[1]*2+r[2]+r[3]+r[4]+4>>3);
      cdst+=_dst_ystride;
      *cdst=(unsigned char)(r[0]*2+r[1]+r[2]*2+r[3]+r[4]+r[5]+4>>3);
      cdst+=_dst_ystride;
      for(by=0;by<4;by++){
        *cdst=(unsigned char)(r[by]+r[by+1]+r[by+2]+r[by+3]*2+
         r[by+4]+r[by+5]+r[by+6]+4>>3);
        cdst+=_dst_ystride;
      }
      *cdst=(unsigned char)(r[4]+r[5]+r[6]+r[7]*2+r[8]+r[9]*2+4>>3);
      cdst+=_dst_ystride;
      *cdst=(unsigned char)(r[5]+r[6]+r[7]+r[8]*2+r[9]*3+4>>3);
    }
    else{
      for(by=1;by<=8;by++){
        *cdst=(unsigned char)r[by];
        cdst+=_dst_ystride;
      }
    }
    rdst++;
    rsrc++;
  }
}

/*Filter a vertical block edge.*/
static void oc_filter_vedge(unsigned char *_dst,int _dst_ystride,
 int _qstep,int _flimit,int *_variances){
  unsigned char       *rdst;
  const unsigned char *rsrc;
  unsigned char       *cdst;
  int                  r[10];
  int                  sum0;
  int                  sum1;
  int                  bx;
  int                  by;
  cdst=_dst;
  for(by=0;by<8;by++){
    rsrc=cdst-1;
    rdst=cdst;
    for(bx=0;bx<10;bx++)r[bx]=*rsrc++;
    sum0=sum1=0;
    for(bx=0;bx<4;bx++){
      sum0+=abs(r[bx+1]-r[bx]);
      sum1+=abs(r[bx+5]-r[bx+6]);
    }
    _variances[0]+=OC_MINI(255,sum0);
    _variances[1]+=OC_MINI(255,sum1);
    if(sum0<_flimit&&sum1<_flimit&&r[5]-r[4]<_qstep&&r[4]-r[5]<_qstep){
      *rdst++=(unsigned char)(r[0]*3+r[1]*2+r[2]+r[3]+r[4]+4>>3);
      *rdst++=(unsigned char)(r[0]*2+r[1]+r[2]*2+r[3]+r[4]+r[5]+4>>3);
      for(bx=0;bx<4;bx++){
        *rdst++=(unsigned char)(r[bx]+r[bx+1]+r[bx+2]+r[bx+3]*2+
         r[bx+4]+r[bx+5]+r[bx+6]+4>>3);
      }
      *rdst++=(unsigned char)(r[4]+r[5]+r[6]+r[7]*2+r[8]+r[9]*2+4>>3);
      *rdst=(unsigned char)(r[5]+r[6]+r[7]+r[8]*2+r[9]*3+4>>3);
    }
    cdst+=_dst_ystride;
  }
}

static void oc_dec_deblock_frag_rows(oc_dec_ctx *_dec,
 th_img_plane *_dst,th_img_plane *_src,int _pli,int _fragy0,
 int _fragy_end){
  oc_fragment_plane   *fplane;
  int                 *variance;
  unsigned char       *dc_qi;
  unsigned char       *dst;
  const unsigned char *src;
  ptrdiff_t            froffset;
  int                  dst_ystride;
  int                  src_ystride;
  int                  nhfrags;
  int                  width;
  int                  notstart;
  int                  notdone;
  int                  flimit;
  int                  qstep;
  int                  y_end;
  int                  y;
  int                  x;
  _dst+=_pli;
  _src+=_pli;
  fplane=_dec->state.fplanes+_pli;
  nhfrags=fplane->nhfrags;
  froffset=fplane->froffset+_fragy0*(ptrdiff_t)nhfrags;
  variance=_dec->variances+froffset;
  dc_qi=_dec->dc_qis+froffset;
  notstart=_fragy0>0;
  notdone=_fragy_end<fplane->nvfrags;
  /*We want to clear an extra row of variances, except at the end.*/
  memset(variance+(nhfrags&-notstart),0,
   (_fragy_end+notdone-_fragy0-notstart)*(nhfrags*sizeof(variance[0])));
  /*Except for the first time, we want to point to the middle of the row.*/
  y=(_fragy0<<3)+(notstart<<2);
  dst_ystride=_dst->stride;
  src_ystride=_src->stride;
  dst=_dst->data+y*(ptrdiff_t)dst_ystride;
  src=_src->data+y*(ptrdiff_t)src_ystride;
  width=_dst->width;
  for(;y<4;y++){
    memcpy(dst,src,width*sizeof(dst[0]));
    dst+=dst_ystride;
    src+=src_ystride;
  }
  /*We also want to skip the last row in the frame for this loop.*/
  y_end=_fragy_end-!notdone<<3;
  for(;y<y_end;y+=8){
    qstep=_dec->pp_dc_scale[*dc_qi];
    flimit=(qstep*3)>>2;
    oc_filter_hedge(dst,dst_ystride,src-src_ystride,src_ystride,
     qstep,flimit,variance,variance+nhfrags);
    variance++;
    dc_qi++;
    for(x=8;x<width;x+=8){
      qstep=_dec->pp_dc_scale[*dc_qi];
      flimit=(qstep*3)>>2;
      oc_filter_hedge(dst+x,dst_ystride,src+x-src_ystride,src_ystride,
       qstep,flimit,variance,variance+nhfrags);
      oc_filter_vedge(dst+x-(dst_ystride<<2)-4,dst_ystride,
       qstep,flimit,variance-1);
      variance++;
      dc_qi++;
    }
    dst+=dst_ystride<<3;
    src+=src_ystride<<3;
  }
  /*And finally, handle the last row in the frame, if it's in the range.*/
  if(!notdone){
    int height;
    height=_dst->height;
    for(;y<height;y++){
      memcpy(dst,src,width*sizeof(dst[0]));
      dst+=dst_ystride;
      src+=src_ystride;
    }
    /*Filter the last row of vertical block edges.*/
    dc_qi++;
    for(x=8;x<width;x+=8){
      qstep=_dec->pp_dc_scale[*dc_qi++];
      flimit=(qstep*3)>>2;
      oc_filter_vedge(dst+x-(dst_ystride<<3)-4,dst_ystride,
       qstep,flimit,variance++);
    }
  }
}

static void oc_dering_block(unsigned char *_idata,int _ystride,int _b,
 int _dc_scale,int _sharp_mod,int _strong){
  static const unsigned char OC_MOD_MAX[2]={24,32};
  static const unsigned char OC_MOD_SHIFT[2]={1,0};
  const unsigned char *psrc;
  const unsigned char *src;
  const unsigned char *nsrc;
  unsigned char       *dst;
  int                  vmod[72];
  int                  hmod[72];
  int                  mod_hi;
  int                  by;
  int                  bx;
  mod_hi=OC_MINI(3*_dc_scale,OC_MOD_MAX[_strong]);
  dst=_idata;
  src=dst;
  psrc=src-(_ystride&-!(_b&4));
  for(by=0;by<9;by++){
    for(bx=0;bx<8;bx++){
      int mod;
      mod=32+_dc_scale-(abs(src[bx]-psrc[bx])<<OC_MOD_SHIFT[_strong]);
      vmod[(by<<3)+bx]=mod<-64?_sharp_mod:OC_CLAMPI(0,mod,mod_hi);
    }
    psrc=src;
    src+=_ystride&-(!(_b&8)|by<7);
  }
  nsrc=dst;
  psrc=dst-!(_b&1);
  for(bx=0;bx<9;bx++){
    src=nsrc;
    for(by=0;by<8;by++){
      int mod;
      mod=32+_dc_scale-(abs(*src-*psrc)<<OC_MOD_SHIFT[_strong]);
      hmod[(bx<<3)+by]=mod<-64?_sharp_mod:OC_CLAMPI(0,mod,mod_hi);
      psrc+=_ystride;
      src+=_ystride;
    }
    psrc=nsrc;
    nsrc+=!(_b&2)|bx<7;
  }
  src=dst;
  psrc=src-(_ystride&-!(_b&4));
  nsrc=src+_ystride;
  for(by=0;by<8;by++){
    int a;
    int b;
    int w;
    a=128;
    b=64;
    w=hmod[by];
    a-=w;
    b+=w**(src-!(_b&1));
    w=vmod[by<<3];
    a-=w;
    b+=w*psrc[0];
    w=vmod[by+1<<3];
    a-=w;
    b+=w*nsrc[0];
    w=hmod[(1<<3)+by];
    a-=w;
    b+=w*src[1];
    dst[0]=OC_CLAMP255(a*src[0]+b>>7);
    for(bx=1;bx<7;bx++){
      a=128;
      b=64;
      w=hmod[(bx<<3)+by];
      a-=w;
      b+=w*src[bx-1];
      w=vmod[(by<<3)+bx];
      a-=w;
      b+=w*psrc[bx];
      w=vmod[(by+1<<3)+bx];
      a-=w;
      b+=w*nsrc[bx];
      w=hmod[(bx+1<<3)+by];
      a-=w;
      b+=w*src[bx+1];
      dst[bx]=OC_CLAMP255(a*src[bx]+b>>7);
    }
    a=128;
    b=64;
    w=hmod[(7<<3)+by];
    a-=w;
    b+=w*src[6];
    w=vmod[(by<<3)+7];
    a-=w;
    b+=w*psrc[7];
    w=vmod[(by+1<<3)+7];
    a-=w;
    b+=w*nsrc[7];
    w=hmod[(8<<3)+by];
    a-=w;
    b+=w*src[7+!(_b&2)];
    dst[7]=OC_CLAMP255(a*src[7]+b>>7);
    dst+=_ystride;
    psrc=src;
    src=nsrc;
    nsrc+=_ystride&-(!(_b&8)|by<6);
  }
}

#define OC_DERING_THRESH1 (384)
#define OC_DERING_THRESH2 (4*OC_DERING_THRESH1)
#define OC_DERING_THRESH3 (5*OC_DERING_THRESH1)
#define OC_DERING_THRESH4 (10*OC_DERING_THRESH1)

static void oc_dec_dering_frag_rows(oc_dec_ctx *_dec,th_img_plane *_img,
 int _pli,int _fragy0,int _fragy_end){
  th_img_plane      *iplane;
  oc_fragment_plane *fplane;
  oc_fragment       *frag;
  int               *variance;
  unsigned char     *idata;
  ptrdiff_t          froffset;
  int                ystride;
  int                nhfrags;
  int                sthresh;
  int                strong;
  int                y_end;
  int                width;
  int                height;
  int                y;
  int                x;
  iplane=_img+_pli;
  fplane=_dec->state.fplanes+_pli;
  nhfrags=fplane->nhfrags;
  froffset=fplane->froffset+_fragy0*(ptrdiff_t)nhfrags;
  variance=_dec->variances+froffset;
  frag=_dec->state.frags+froffset;
  strong=_dec->pp_level>=(_pli?OC_PP_LEVEL_SDERINGC:OC_PP_LEVEL_SDERINGY);
  sthresh=_pli?OC_DERING_THRESH4:OC_DERING_THRESH3;
  y=_fragy0<<3;
  ystride=iplane->stride;
  idata=iplane->data+y*(ptrdiff_t)ystride;
  y_end=_fragy_end<<3;
  width=iplane->width;
  height=iplane->height;
  for(;y<y_end;y+=8){
    for(x=0;x<width;x+=8){
      int b;
      int qi;
      int var;
      qi=_dec->state.qis[frag->qii];
      var=*variance;
      b=(x<=0)|(x+8>=width)<<1|(y<=0)<<2|(y+8>=height)<<3;
      if(strong&&var>sthresh){
        oc_dering_block(idata+x,ystride,b,
         _dec->pp_dc_scale[qi],_dec->pp_sharp_mod[qi],1);
        if(_pli||!(b&1)&&*(variance-1)>OC_DERING_THRESH4||
         !(b&2)&&variance[1]>OC_DERING_THRESH4||
         !(b&4)&&*(variance-nhfrags)>OC_DERING_THRESH4||
         !(b&8)&&variance[nhfrags]>OC_DERING_THRESH4){
          oc_dering_block(idata+x,ystride,b,
           _dec->pp_dc_scale[qi],_dec->pp_sharp_mod[qi],1);
          oc_dering_block(idata+x,ystride,b,
           _dec->pp_dc_scale[qi],_dec->pp_sharp_mod[qi],1);
        }
      }
      else if(var>OC_DERING_THRESH2){
        oc_dering_block(idata+x,ystride,b,
         _dec->pp_dc_scale[qi],_dec->pp_sharp_mod[qi],1);
      }
      else if(var>OC_DERING_THRESH1){
        oc_dering_block(idata+x,ystride,b,
         _dec->pp_dc_scale[qi],_dec->pp_sharp_mod[qi],0);
      }
      frag++;
      variance++;
    }
    idata+=ystride<<3;
  }
}



th_dec_ctx *th_decode_alloc(const th_info *_info,const th_setup_info *_setup){
  oc_dec_ctx *dec;
  if(_info==NULL||_setup==NULL)return NULL;
  dec=oc_aligned_malloc(sizeof(*dec),16);
  if(dec==NULL||oc_dec_init(dec,_info,_setup)<0){
    oc_aligned_free(dec);
    return NULL;
  }
  dec->state.curframe_num=0;
  return dec;
}

void th_decode_free(th_dec_ctx *_dec){
  if(_dec!=NULL){
    oc_dec_clear(_dec);
    oc_aligned_free(_dec);
  }
}

int th_decode_ctl(th_dec_ctx *_dec,int _req,void *_buf,
 size_t _buf_sz){
  switch(_req){
  case TH_DECCTL_GET_PPLEVEL_MAX:{
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    (*(int *)_buf)=OC_PP_LEVEL_MAX;
    return 0;
  }break;
  case TH_DECCTL_SET_PPLEVEL:{
    int pp_level;
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    pp_level=*(int *)_buf;
    if(pp_level<0||pp_level>OC_PP_LEVEL_MAX)return TH_EINVAL;
    _dec->pp_level=pp_level;
    return 0;
  }break;
  case TH_DECCTL_SET_GRANPOS:{
    ogg_int64_t granpos;
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(ogg_int64_t))return TH_EINVAL;
    granpos=*(ogg_int64_t *)_buf;
    if(granpos<0)return TH_EINVAL;
    _dec->state.granpos=granpos;
    _dec->state.keyframe_num=(granpos>>_dec->state.info.keyframe_granule_shift)
     -_dec->state.granpos_bias;
    _dec->state.curframe_num=_dec->state.keyframe_num
     +(granpos&(1<<_dec->state.info.keyframe_granule_shift)-1);
    return 0;
  }break;
  case TH_DECCTL_SET_STRIPE_CB:{
    th_stripe_callback *cb;
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(th_stripe_callback))return TH_EINVAL;
    cb=(th_stripe_callback *)_buf;
    _dec->stripe_cb.ctx=cb->ctx;
    _dec->stripe_cb.stripe_decoded=cb->stripe_decoded;
    return 0;
  }break;
#ifdef HAVE_CAIRO
  case TH_DECCTL_SET_TELEMETRY_MBMODE:{
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    _dec->telemetry_mbmode=*(int *)_buf;
    return 0;
  }break;
  case TH_DECCTL_SET_TELEMETRY_MV:{
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    _dec->telemetry_mv=*(int *)_buf;
    return 0;
  }break;
  case TH_DECCTL_SET_TELEMETRY_QI:{
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    _dec->telemetry_qi=*(int *)_buf;
    return 0;
  }break;
  case TH_DECCTL_SET_TELEMETRY_BITS:{
    if(_dec==NULL||_buf==NULL)return TH_EFAULT;
    if(_buf_sz!=sizeof(int))return TH_EINVAL;
    _dec->telemetry_bits=*(int *)_buf;
    return 0;
  }break;
#endif
  default:return TH_EIMPL;
  }
}

/*We're decoding an INTER frame, but have no initialized reference
   buffers (i.e., decoding did not start on a key frame).
  We initialize them to a solid gray here.*/
static void oc_dec_init_dummy_frame(th_dec_ctx *_dec){
  th_info   *info;
  size_t     yplane_sz;
  size_t     cplane_sz;
  ptrdiff_t  yoffset;
  int        yhstride;
  int        yheight;
  int        chstride;
  int        cheight;
  _dec->state.ref_frame_idx[OC_FRAME_GOLD]=0;
  _dec->state.ref_frame_idx[OC_FRAME_PREV]=0;
  _dec->state.ref_frame_idx[OC_FRAME_SELF]=0;
  _dec->state.ref_frame_data[OC_FRAME_GOLD]=
   _dec->state.ref_frame_data[OC_FRAME_PREV]=
   _dec->state.ref_frame_data[OC_FRAME_SELF]=
   _dec->state.ref_frame_bufs[0][0].data;
  memcpy(_dec->pp_frame_buf,_dec->state.ref_frame_bufs[0],
   sizeof(_dec->pp_frame_buf[0])*3);
  info=&_dec->state.info;
  yhstride=abs(_dec->state.ref_ystride[0]);
  yheight=info->frame_height+2*OC_UMV_PADDING;
  chstride=abs(_dec->state.ref_ystride[1]);
  cheight=yheight>>!(info->pixel_fmt&2);
  yplane_sz=yhstride*(size_t)yheight+16;
  cplane_sz=chstride*(size_t)cheight;
  yoffset=yhstride*(ptrdiff_t)(yheight-OC_UMV_PADDING-1)+OC_UMV_PADDING;
  memset(_dec->state.ref_frame_data[0]-yoffset,0x80,yplane_sz+2*cplane_sz);
}

#if defined(HAVE_CAIRO)
static void oc_render_telemetry(th_dec_ctx *_dec,th_ycbcr_buffer _ycbcr,
 int _telemetry){
  /*Stuff the plane into cairo.*/
  cairo_surface_t *cs;
  unsigned char   *data;
  unsigned char   *y_row;
  unsigned char   *u_row;
  unsigned char   *v_row;
  unsigned char   *rgb_row;
  int              cstride;
  int              w;
  int              h;
  int              x;
  int              y;
  int              hdec;
  int              vdec;
  w=_ycbcr[0].width;
  h=_ycbcr[0].height;
  hdec=!(_dec->state.info.pixel_fmt&1);
  vdec=!(_dec->state.info.pixel_fmt&2);
  /*Lazy data buffer init.
    We could try to re-use the post-processing buffer, which would save
     memory, but complicate the allocation logic there.
    I don't think anyone cares about memory usage when using telemetry; it is
     not meant for embedded devices.*/
  if(_dec->telemetry_frame_data==NULL){
    _dec->telemetry_frame_data=_ogg_malloc(
     (w*h+2*(w>>hdec)*(h>>vdec))*sizeof(*_dec->telemetry_frame_data));
    if(_dec->telemetry_frame_data==NULL)return;
  }
  cs=cairo_image_surface_create(CAIRO_FORMAT_RGB24,w,h);
  /*Sadly, no YUV support in Cairo (yet); convert into the RGB buffer.*/
  data=cairo_image_surface_get_data(cs);
  if(data==NULL){
    cairo_surface_destroy(cs);
    return;
  }
  cstride=cairo_image_surface_get_stride(cs);
  y_row=_ycbcr[0].data;
  u_row=_ycbcr[1].data;
  v_row=_ycbcr[2].data;
  rgb_row=data;
  for(y=0;y<h;y++){
    for(x=0;x<w;x++){
      int r;
      int g;
      int b;
      r=(1904000*y_row[x]+2609823*v_row[x>>hdec]-363703744)/1635200;
      g=(3827562*y_row[x]-1287801*u_row[x>>hdec]
       -2672387*v_row[x>>hdec]+447306710)/3287200;
      b=(952000*y_row[x]+1649289*u_row[x>>hdec]-225932192)/817600;
      rgb_row[4*x+0]=OC_CLAMP255(b);
      rgb_row[4*x+1]=OC_CLAMP255(g);
      rgb_row[4*x+2]=OC_CLAMP255(r);
    }
    y_row+=_ycbcr[0].stride;
    u_row+=_ycbcr[1].stride&-((y&1)|!vdec);
    v_row+=_ycbcr[2].stride&-((y&1)|!vdec);
    rgb_row+=cstride;
  }
  /*Draw coded identifier for each macroblock (stored in Hilbert order).*/
  {
    cairo_t           *c;
    const oc_fragment *frags;
    oc_mv             *frag_mvs;
    const signed char *mb_modes;
    oc_mb_map         *mb_maps;
    size_t             nmbs;
    size_t             mbi;
    int                row2;
    int                col2;
    int                qim[3]={0,0,0};
    if(_dec->state.nqis==2){
      int bqi;
      bqi=_dec->state.qis[0];
      if(_dec->state.qis[1]>bqi)qim[1]=1;
      if(_dec->state.qis[1]<bqi)qim[1]=-1;
    }
    if(_dec->state.nqis==3){
      int bqi;
      int cqi;
      int dqi;
      bqi=_dec->state.qis[0];
      cqi=_dec->state.qis[1];
      dqi=_dec->state.qis[2];
      if(cqi>bqi&&dqi>bqi){
        if(dqi>cqi){
          qim[1]=1;
          qim[2]=2;
        }
        else{
          qim[1]=2;
          qim[2]=1;
        }
      }
      else if(cqi<bqi&&dqi<bqi){
        if(dqi<cqi){
          qim[1]=-1;
          qim[2]=-2;
        }
        else{
          qim[1]=-2;
          qim[2]=-1;
        }
      }
      else{
        if(cqi<bqi)qim[1]=-1;
        else qim[1]=1;
        if(dqi<bqi)qim[2]=-1;
        else qim[2]=1;
      }
    }
    c=cairo_create(cs);
    frags=_dec->state.frags;
    frag_mvs=_dec->state.frag_mvs;
    mb_modes=_dec->state.mb_modes;
    mb_maps=_dec->state.mb_maps;
    nmbs=_dec->state.nmbs;
    row2=0;
    col2=0;
    for(mbi=0;mbi<nmbs;mbi++){
      float x;
      float y;
      int   bi;
      y=h-(row2+((col2+1>>1)&1))*16-16;
      x=(col2>>1)*16;
      cairo_set_line_width(c,1.);
      /*Keyframe (all intra) red box.*/
      if(_dec->state.frame_type==OC_INTRA_FRAME){
        if(_dec->telemetry_mbmode&0x02){
          cairo_set_source_rgba(c,1.,0,0,.5);
          cairo_rectangle(c,x+2.5,y+2.5,11,11);
          cairo_stroke_preserve(c);
          cairo_set_source_rgba(c,1.,0,0,.25);
          cairo_fill(c);
        }
      }
      else{
        ptrdiff_t fragi;
        int       frag_mvx;
        int       frag_mvy;
        for(bi=0;bi<4;bi++){
          fragi=mb_maps[mbi][0][bi];
          if(fragi>=0&&frags[fragi].coded){
            frag_mvx=OC_MV_X(frag_mvs[fragi]);
            frag_mvy=OC_MV_Y(frag_mvs[fragi]);
            break;
          }
        }
        if(bi<4){
          switch(mb_modes[mbi]){
            case OC_MODE_INTRA:{
              if(_dec->telemetry_mbmode&0x02){
                cairo_set_source_rgba(c,1.,0,0,.5);
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,1.,0,0,.25);
                cairo_fill(c);
              }
            }break;
            case OC_MODE_INTER_NOMV:{
              if(_dec->telemetry_mbmode&0x01){
                cairo_set_source_rgba(c,0,0,1.,.5);
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,0,0,1.,.25);
                cairo_fill(c);
              }
            }break;
            case OC_MODE_INTER_MV:{
              if(_dec->telemetry_mbmode&0x04){
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_set_source_rgba(c,0,1.,0,.5);
                cairo_stroke(c);
              }
              if(_dec->telemetry_mv&0x04){
                cairo_move_to(c,x+8+frag_mvx,y+8-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+8+frag_mvx*.66,y+8-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+8+frag_mvx*.33,y+8-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+8,y+8);
                cairo_stroke(c);
              }
            }break;
            case OC_MODE_INTER_MV_LAST:{
              if(_dec->telemetry_mbmode&0x08){
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_set_source_rgba(c,0,1.,0,.5);
                cairo_move_to(c,x+13.5,y+2.5);
                cairo_line_to(c,x+2.5,y+8);
                cairo_line_to(c,x+13.5,y+13.5);
                cairo_stroke(c);
              }
              if(_dec->telemetry_mv&0x08){
                cairo_move_to(c,x+8+frag_mvx,y+8-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+8+frag_mvx*.66,y+8-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+8+frag_mvx*.33,y+8-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+8,y+8);
                cairo_stroke(c);
              }
            }break;
            case OC_MODE_INTER_MV_LAST2:{
              if(_dec->telemetry_mbmode&0x10){
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_set_source_rgba(c,0,1.,0,.5);
                cairo_move_to(c,x+8,y+2.5);
                cairo_line_to(c,x+2.5,y+8);
                cairo_line_to(c,x+8,y+13.5);
                cairo_move_to(c,x+13.5,y+2.5);
                cairo_line_to(c,x+8,y+8);
                cairo_line_to(c,x+13.5,y+13.5);
                cairo_stroke(c);
              }
              if(_dec->telemetry_mv&0x10){
                cairo_move_to(c,x+8+frag_mvx,y+8-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+8+frag_mvx*.66,y+8-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+8+frag_mvx*.33,y+8-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+8,y+8);
                cairo_stroke(c);
              }
            }break;
            case OC_MODE_GOLDEN_NOMV:{
              if(_dec->telemetry_mbmode&0x20){
                cairo_set_source_rgba(c,1.,1.,0,.5);
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,1.,1.,0,.25);
                cairo_fill(c);
              }
            }break;
            case OC_MODE_GOLDEN_MV:{
              if(_dec->telemetry_mbmode&0x40){
                cairo_rectangle(c,x+2.5,y+2.5,11,11);
                cairo_set_source_rgba(c,1.,1.,0,.5);
                cairo_stroke(c);
              }
              if(_dec->telemetry_mv&0x40){
                cairo_move_to(c,x+8+frag_mvx,y+8-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+8+frag_mvx*.66,y+8-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+8+frag_mvx*.33,y+8-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+8,y+8);
                cairo_stroke(c);
              }
            }break;
            case OC_MODE_INTER_MV_FOUR:{
              if(_dec->telemetry_mbmode&0x80){
                cairo_rectangle(c,x+2.5,y+2.5,4,4);
                cairo_rectangle(c,x+9.5,y+2.5,4,4);
                cairo_rectangle(c,x+2.5,y+9.5,4,4);
                cairo_rectangle(c,x+9.5,y+9.5,4,4);
                cairo_set_source_rgba(c,0,1.,0,.5);
                cairo_stroke(c);
              }
              /*4mv is odd, coded in raster order.*/
              fragi=mb_maps[mbi][0][0];
              if(frags[fragi].coded&&_dec->telemetry_mv&0x80){
                frag_mvx=OC_MV_X(frag_mvs[fragi]);
                frag_mvx=OC_MV_Y(frag_mvs[fragi]);
                cairo_move_to(c,x+4+frag_mvx,y+12-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+4+frag_mvx*.66,y+12-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+4+frag_mvx*.33,y+12-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+4,y+12);
                cairo_stroke(c);
              }
              fragi=mb_maps[mbi][0][1];
              if(frags[fragi].coded&&_dec->telemetry_mv&0x80){
                frag_mvx=OC_MV_X(frag_mvs[fragi]);
                frag_mvx=OC_MV_Y(frag_mvs[fragi]);
                cairo_move_to(c,x+12+frag_mvx,y+12-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+12+frag_mvx*.66,y+12-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+12+frag_mvx*.33,y+12-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+12,y+12);
                cairo_stroke(c);
              }
              fragi=mb_maps[mbi][0][2];
              if(frags[fragi].coded&&_dec->telemetry_mv&0x80){
                frag_mvx=OC_MV_X(frag_mvs[fragi]);
                frag_mvx=OC_MV_Y(frag_mvs[fragi]);
                cairo_move_to(c,x+4+frag_mvx,y+4-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+4+frag_mvx*.66,y+4-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+4+frag_mvx*.33,y+4-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+4,y+4);
                cairo_stroke(c);
              }
              fragi=mb_maps[mbi][0][3];
              if(frags[fragi].coded&&_dec->telemetry_mv&0x80){
                frag_mvx=OC_MV_X(frag_mvs[fragi]);
                frag_mvx=OC_MV_Y(frag_mvs[fragi]);
                cairo_move_to(c,x+12+frag_mvx,y+4-frag_mvy);
                cairo_set_source_rgba(c,1.,1.,1.,.9);
                cairo_set_line_width(c,3.);
                cairo_line_to(c,x+12+frag_mvx*.66,y+4-frag_mvy*.66);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,2.);
                cairo_line_to(c,x+12+frag_mvx*.33,y+4-frag_mvy*.33);
                cairo_stroke_preserve(c);
                cairo_set_line_width(c,1.);
                cairo_line_to(c,x+12,y+4);
                cairo_stroke(c);
              }
            }break;
          }
        }
      }
      /*qii illustration.*/
      if(_dec->telemetry_qi&0x2){
        cairo_set_line_cap(c,CAIRO_LINE_CAP_SQUARE);
        for(bi=0;bi<4;bi++){
          ptrdiff_t fragi;
          int       qiv;
          int       xp;
          int       yp;
          xp=x+(bi&1)*8;
          yp=y+8-(bi&2)*4;
          fragi=mb_maps[mbi][0][bi];
          if(fragi>=0&&frags[fragi].coded){
            qiv=qim[frags[fragi].qii];
            cairo_set_line_width(c,3.);
            cairo_set_source_rgba(c,0.,0.,0.,.5);
            switch(qiv){
              /*Double plus:*/
              case 2:{
                if((bi&1)^((bi&2)>>1)){
                  cairo_move_to(c,xp+2.5,yp+1.5);
                  cairo_line_to(c,xp+2.5,yp+3.5);
                  cairo_move_to(c,xp+1.5,yp+2.5);
                  cairo_line_to(c,xp+3.5,yp+2.5);
                  cairo_move_to(c,xp+5.5,yp+4.5);
                  cairo_line_to(c,xp+5.5,yp+6.5);
                  cairo_move_to(c,xp+4.5,yp+5.5);
                  cairo_line_to(c,xp+6.5,yp+5.5);
                  cairo_stroke_preserve(c);
                  cairo_set_source_rgba(c,0.,1.,1.,1.);
                }
                else{
                  cairo_move_to(c,xp+5.5,yp+1.5);
                  cairo_line_to(c,xp+5.5,yp+3.5);
                  cairo_move_to(c,xp+4.5,yp+2.5);
                  cairo_line_to(c,xp+6.5,yp+2.5);
                  cairo_move_to(c,xp+2.5,yp+4.5);
                  cairo_line_to(c,xp+2.5,yp+6.5);
                  cairo_move_to(c,xp+1.5,yp+5.5);
                  cairo_line_to(c,xp+3.5,yp+5.5);
                  cairo_stroke_preserve(c);
                  cairo_set_source_rgba(c,0.,1.,1.,1.);
                }
              }break;
              /*Double minus:*/
              case -2:{
                cairo_move_to(c,xp+2.5,yp+2.5);
                cairo_line_to(c,xp+5.5,yp+2.5);
                cairo_move_to(c,xp+2.5,yp+5.5);
                cairo_line_to(c,xp+5.5,yp+5.5);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,1.,1.,1.,1.);
              }break;
              /*Plus:*/
              case 1:{
                if((bi&2)==0)yp-=2;
                if((bi&1)==0)xp-=2;
                cairo_move_to(c,xp+4.5,yp+2.5);
                cairo_line_to(c,xp+4.5,yp+6.5);
                cairo_move_to(c,xp+2.5,yp+4.5);
                cairo_line_to(c,xp+6.5,yp+4.5);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,.1,1.,.3,1.);
                break;
              }
              /*Fall through.*/
              /*Minus:*/
              case -1:{
                cairo_move_to(c,xp+2.5,yp+4.5);
                cairo_line_to(c,xp+6.5,yp+4.5);
                cairo_stroke_preserve(c);
                cairo_set_source_rgba(c,1.,.3,.1,1.);
              }break;
              default:continue;
            }
            cairo_set_line_width(c,1.);
            cairo_stroke(c);
          }
        }
      }
      col2++;
      if((col2>>1)>=_dec->state.nhmbs){
        col2=0;
        row2+=2;
      }
    }
    /*Bit usage indicator[s]:*/
    if(_dec->telemetry_bits){
      int widths[6];
      int fpsn;
      int fpsd;
      int mult;
      int fullw;
      int padw;
      int i;
      fpsn=_dec->state.info.fps_numerator;
      fpsd=_dec->state.info.fps_denominator;
      mult=(_dec->telemetry_bits>=0xFF?1:_dec->telemetry_bits);
      fullw=250.f*h*fpsd*mult/fpsn;
      padw=w-24;
      /*Header and coded block bits.*/
      if(_dec->telemetry_frame_bytes<0||
       _dec->telemetry_frame_bytes==OC_LOTS_OF_BITS){
        _dec->telemetry_frame_bytes=0;
      }
      if(_dec->telemetry_coding_bytes<0||
       _dec->telemetry_coding_bytes>_dec->telemetry_frame_bytes){
        _dec->telemetry_coding_bytes=0;
      }
      if(_dec->telemetry_mode_bytes<0||
       _dec->telemetry_mode_bytes>_dec->telemetry_frame_bytes){
        _dec->telemetry_mode_bytes=0;
      }
      if(_dec->telemetry_mv_bytes<0||
       _dec->telemetry_mv_bytes>_dec->telemetry_frame_bytes){
        _dec->telemetry_mv_bytes=0;
      }
      if(_dec->telemetry_qi_bytes<0||
       _dec->telemetry_qi_bytes>_dec->telemetry_frame_bytes){
        _dec->telemetry_qi_bytes=0;
      }
      if(_dec->telemetry_dc_bytes<0||
       _dec->telemetry_dc_bytes>_dec->telemetry_frame_bytes){
        _dec->telemetry_dc_bytes=0;
      }
      widths[0]=padw*
       (_dec->telemetry_frame_bytes-_dec->telemetry_coding_bytes)/fullw;
      widths[1]=padw*
       (_dec->telemetry_coding_bytes-_dec->telemetry_mode_bytes)/fullw;
      widths[2]=padw*
       (_dec->telemetry_mode_bytes-_dec->telemetry_mv_bytes)/fullw;
      widths[3]=padw*(_dec->telemetry_mv_bytes-_dec->telemetry_qi_bytes)/fullw;
      widths[4]=padw*(_dec->telemetry_qi_bytes-_dec->telemetry_dc_bytes)/fullw;
      widths[5]=padw*(_dec->telemetry_dc_bytes)/fullw;
      for(i=0;i<6;i++)if(widths[i]>w)widths[i]=w;
      cairo_set_source_rgba(c,.0,.0,.0,.6);
      cairo_rectangle(c,10,h-33,widths[0]+1,5);
      cairo_rectangle(c,10,h-29,widths[1]+1,5);
      cairo_rectangle(c,10,h-25,widths[2]+1,5);
      cairo_rectangle(c,10,h-21,widths[3]+1,5);
      cairo_rectangle(c,10,h-17,widths[4]+1,5);
      cairo_rectangle(c,10,h-13,widths[5]+1,5);
      cairo_fill(c);
      cairo_set_source_rgb(c,1,0,0);
      cairo_rectangle(c,10.5,h-32.5,widths[0],4);
      cairo_fill(c);
      cairo_set_source_rgb(c,0,1,0);
      cairo_rectangle(c,10.5,h-28.5,widths[1],4);
      cairo_fill(c);
      cairo_set_source_rgb(c,0,0,1);
      cairo_rectangle(c,10.5,h-24.5,widths[2],4);
      cairo_fill(c);
      cairo_set_source_rgb(c,.6,.4,.0);
      cairo_rectangle(c,10.5,h-20.5,widths[3],4);
      cairo_fill(c);
      cairo_set_source_rgb(c,.3,.3,.3);
      cairo_rectangle(c,10.5,h-16.5,widths[4],4);
      cairo_fill(c);
      cairo_set_source_rgb(c,.5,.5,.8);
      cairo_rectangle(c,10.5,h-12.5,widths[5],4);
      cairo_fill(c);
    }
    /*Master qi indicator[s]:*/
    if(_dec->telemetry_qi&0x1){
      cairo_text_extents_t extents;
      char                 buffer[10];
      int                  p;
      int                  y;
      p=0;
      y=h-7.5;
      if(_dec->state.qis[0]>=10)buffer[p++]=48+_dec->state.qis[0]/10;
      buffer[p++]=48+_dec->state.qis[0]%10;
      if(_dec->state.nqis>=2){
        buffer[p++]=' ';
        if(_dec->state.qis[1]>=10)buffer[p++]=48+_dec->state.qis[1]/10;
        buffer[p++]=48+_dec->state.qis[1]%10;
      }
      if(_dec->state.nqis==3){
        buffer[p++]=' ';
        if(_dec->state.qis[2]>=10)buffer[p++]=48+_dec->state.qis[2]/10;
        buffer[p++]=48+_dec->state.qis[2]%10;
      }
      buffer[p++]='\0';
      cairo_select_font_face(c,"sans",
       CAIRO_FONT_SLANT_NORMAL,CAIRO_FONT_WEIGHT_BOLD);
      cairo_set_font_size(c,18);
      cairo_text_extents(c,buffer,&extents);
      cairo_set_source_rgb(c,1,1,1);
      cairo_move_to(c,w-extents.x_advance-10,y);
      cairo_show_text(c,buffer);
      cairo_set_source_rgb(c,0,0,0);
      cairo_move_to(c,w-extents.x_advance-10,y);
      cairo_text_path(c,buffer);
      cairo_set_line_width(c,.8);
      cairo_set_line_join(c,CAIRO_LINE_JOIN_ROUND);
      cairo_stroke(c);
    }
    cairo_destroy(c);
  }
  /*Out of the Cairo plane into the telemetry YUV buffer.*/
  _ycbcr[0].data=_dec->telemetry_frame_data;
  _ycbcr[0].stride=_ycbcr[0].width;
  _ycbcr[1].data=_ycbcr[0].data+h*_ycbcr[0].stride;
  _ycbcr[1].stride=_ycbcr[1].width;
  _ycbcr[2].data=_ycbcr[1].data+(h>>vdec)*_ycbcr[1].stride;
  _ycbcr[2].stride=_ycbcr[2].width;
  y_row=_ycbcr[0].data;
  u_row=_ycbcr[1].data;
  v_row=_ycbcr[2].data;
  rgb_row=data;
  /*This is one of the few places it's worth handling chroma on a
     case-by-case basis.*/
  switch(_dec->state.info.pixel_fmt){
    case TH_PF_420:{
      for(y=0;y<h;y+=2){
        unsigned char *y_row2;
        unsigned char *rgb_row2;
        y_row2=y_row+_ycbcr[0].stride;
        rgb_row2=rgb_row+cstride;
        for(x=0;x<w;x+=2){
          int y;
          int u;
          int v;
          y=(65481*rgb_row[4*x+2]+128553*rgb_row[4*x+1]
           +24966*rgb_row[4*x+0]+4207500)/255000;
          y_row[x]=OC_CLAMP255(y);
          y=(65481*rgb_row[4*x+6]+128553*rgb_row[4*x+5]
           +24966*rgb_row[4*x+4]+4207500)/255000;
          y_row[x+1]=OC_CLAMP255(y);
          y=(65481*rgb_row2[4*x+2]+128553*rgb_row2[4*x+1]
           +24966*rgb_row2[4*x+0]+4207500)/255000;
          y_row2[x]=OC_CLAMP255(y);
          y=(65481*rgb_row2[4*x+6]+128553*rgb_row2[4*x+5]
           +24966*rgb_row2[4*x+4]+4207500)/255000;
          y_row2[x+1]=OC_CLAMP255(y);
          u=(-8372*(rgb_row[4*x+2]+rgb_row[4*x+6]
           +rgb_row2[4*x+2]+rgb_row2[4*x+6])
           -16436*(rgb_row[4*x+1]+rgb_row[4*x+5]
           +rgb_row2[4*x+1]+rgb_row2[4*x+5])
           +24808*(rgb_row[4*x+0]+rgb_row[4*x+4]
           +rgb_row2[4*x+0]+rgb_row2[4*x+4])+29032005)/225930;
          v=(39256*(rgb_row[4*x+2]+rgb_row[4*x+6]
           +rgb_row2[4*x+2]+rgb_row2[4*x+6])
           -32872*(rgb_row[4*x+1]+rgb_row[4*x+5]
            +rgb_row2[4*x+1]+rgb_row2[4*x+5])
           -6384*(rgb_row[4*x+0]+rgb_row[4*x+4]
            +rgb_row2[4*x+0]+rgb_row2[4*x+4])+45940035)/357510;
          u_row[x>>1]=OC_CLAMP255(u);
          v_row[x>>1]=OC_CLAMP255(v);
        }
        y_row+=_ycbcr[0].stride<<1;
        u_row+=_ycbcr[1].stride;
        v_row+=_ycbcr[2].stride;
        rgb_row+=cstride<<1;
      }
    }break;
    case TH_PF_422:{
      for(y=0;y<h;y++){
        for(x=0;x<w;x+=2){
          int y;
          int u;
          int v;
          y=(65481*rgb_row[4*x+2]+128553*rgb_row[4*x+1]
           +24966*rgb_row[4*x+0]+4207500)/255000;
          y_row[x]=OC_CLAMP255(y);
          y=(65481*rgb_row[4*x+6]+128553*rgb_row[4*x+5]
           +24966*rgb_row[4*x+4]+4207500)/255000;
          y_row[x+1]=OC_CLAMP255(y);
          u=(-16744*(rgb_row[4*x+2]+rgb_row[4*x+6])
           -32872*(rgb_row[4*x+1]+rgb_row[4*x+5])
           +49616*(rgb_row[4*x+0]+rgb_row[4*x+4])+29032005)/225930;
          v=(78512*(rgb_row[4*x+2]+rgb_row[4*x+6])
           -65744*(rgb_row[4*x+1]+rgb_row[4*x+5])
           -12768*(rgb_row[4*x+0]+rgb_row[4*x+4])+45940035)/357510;
          u_row[x>>1]=OC_CLAMP255(u);
          v_row[x>>1]=OC_CLAMP255(v);
        }
        y_row+=_ycbcr[0].stride;
        u_row+=_ycbcr[1].stride;
        v_row+=_ycbcr[2].stride;
        rgb_row+=cstride;
      }
    }break;
    /*case TH_PF_444:*/
    default:{
      for(y=0;y<h;y++){
        for(x=0;x<w;x++){
          int y;
          int u;
          int v;
          y=(65481*rgb_row[4*x+2]+128553*rgb_row[4*x+1]
           +24966*rgb_row[4*x+0]+4207500)/255000;
          u=(-33488*rgb_row[4*x+2]-65744*rgb_row[4*x+1]
           +99232*rgb_row[4*x+0]+29032005)/225930;
          v=(157024*rgb_row[4*x+2]-131488*rgb_row[4*x+1]
           -25536*rgb_row[4*x+0]+45940035)/357510;
          y_row[x]=OC_CLAMP255(y);
          u_row[x]=OC_CLAMP255(u);
          v_row[x]=OC_CLAMP255(v);
        }
        y_row+=_ycbcr[0].stride;
        u_row+=_ycbcr[1].stride;
        v_row+=_ycbcr[2].stride;
        rgb_row+=cstride;
      }
    }break;
  }
  /*Finished.
    Destroy the surface.*/
  cairo_surface_destroy(cs);
}
#endif

int th_decode_packetin(th_dec_ctx *_dec,const ogg_packet *_op,
 ogg_int64_t *_granpos){
  int ret;
  if(_dec==NULL||_op==NULL)return TH_EFAULT;
  /*A completely empty packet indicates a dropped frame and is treated exactly
     like an inter frame with no coded blocks.*/
  if(_op->bytes==0){
    _dec->state.frame_type=OC_INTER_FRAME;
    _dec->state.ntotal_coded_fragis=0;
  }
  else{
    oc_pack_readinit(&_dec->opb,_op->packet,_op->bytes);
    ret=oc_dec_frame_header_unpack(_dec);
    if(ret<0)return ret;
    if(_dec->state.frame_type==OC_INTRA_FRAME)oc_dec_mark_all_intra(_dec);
    else oc_dec_coded_flags_unpack(_dec);
  }
  /*If there have been no reference frames, and we need one, initialize one.*/
  if(_dec->state.frame_type!=OC_INTRA_FRAME&&
   (_dec->state.ref_frame_idx[OC_FRAME_GOLD]<0||
   _dec->state.ref_frame_idx[OC_FRAME_PREV]<0)){
    oc_dec_init_dummy_frame(_dec);
  }
  /*If this was an inter frame with no coded blocks...*/
  if(_dec->state.ntotal_coded_fragis<=0){
    /*Just update the granule position and return.*/
    _dec->state.granpos=(_dec->state.keyframe_num+_dec->state.granpos_bias<<
     _dec->state.info.keyframe_granule_shift)
     +(_dec->state.curframe_num-_dec->state.keyframe_num);
    _dec->state.curframe_num++;
    if(_granpos!=NULL)*_granpos=_dec->state.granpos;
    return TH_DUPFRAME;
  }
  else{
    th_ycbcr_buffer stripe_buf;
    int             stripe_fragy;
    int             refi;
    int             pli;
    int             notstart;
    int             notdone;
#ifdef HAVE_CAIRO
    int             telemetry;
    /*Save the current telemetry state.
      This prevents it from being modified in the middle of decoding this
       frame, which could cause us to skip calls to the striped decoding
       callback.*/
    telemetry=_dec->telemetry_mbmode||_dec->telemetry_mv||
     _dec->telemetry_qi||_dec->telemetry_bits;
#endif
    /*Select a free buffer to use for the reconstructed version of this frame.*/
    for(refi=0;refi==_dec->state.ref_frame_idx[OC_FRAME_GOLD]||
     refi==_dec->state.ref_frame_idx[OC_FRAME_PREV];refi++);
    _dec->state.ref_frame_idx[OC_FRAME_SELF]=refi;
    _dec->state.ref_frame_data[OC_FRAME_SELF]=
     _dec->state.ref_frame_bufs[refi][0].data;
#if defined(HAVE_CAIRO)
    _dec->telemetry_frame_bytes=_op->bytes;
#endif
    if(_dec->state.frame_type==OC_INTRA_FRAME){
      _dec->state.keyframe_num=_dec->state.curframe_num;
#if defined(HAVE_CAIRO)
      _dec->telemetry_coding_bytes=
       _dec->telemetry_mode_bytes=
       _dec->telemetry_mv_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
    }
    else{
#if defined(HAVE_CAIRO)
      _dec->telemetry_coding_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
      oc_dec_mb_modes_unpack(_dec);
#if defined(HAVE_CAIRO)
      _dec->telemetry_mode_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
      oc_dec_mv_unpack_and_frag_modes_fill(_dec);
#if defined(HAVE_CAIRO)
      _dec->telemetry_mv_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
    }
    oc_dec_block_qis_unpack(_dec);
#if defined(HAVE_CAIRO)
    _dec->telemetry_qi_bytes=oc_pack_bytes_left(&_dec->opb);
#endif
    oc_dec_residual_tokens_unpack(_dec);
    /*Update granule position.
      This must be done before the striped decode callbacks so that the
       application knows what to do with the frame data.*/
    _dec->state.granpos=(_dec->state.keyframe_num+_dec->state.granpos_bias<<
     _dec->state.info.keyframe_granule_shift)
     +(_dec->state.curframe_num-_dec->state.keyframe_num);
    _dec->state.curframe_num++;
    if(_granpos!=NULL)*_granpos=_dec->state.granpos;
    /*All of the rest of the operations -- DC prediction reversal,
       reconstructing coded fragments, copying uncoded fragments, loop
       filtering, extending borders, and out-of-loop post-processing -- should
       be pipelined.
      I.e., DC prediction reversal, reconstruction, and uncoded fragment
       copying are done for one or two super block rows, then loop filtering is
       run as far as it can, then bordering copying, then post-processing.
      For 4:2:0 video a Minimum Codable Unit or MCU contains two luma super
       block rows, and one chroma.
      Otherwise, an MCU consists of one super block row from each plane.
      Inside each MCU, we perform all of the steps on one color plane before
       moving on to the next.
      After reconstruction, the additional filtering stages introduce a delay
       since they need some pixels from the next fragment row.
      Thus the actual number of decoded rows available is slightly smaller for
       the first MCU, and slightly larger for the last.

      This entire process allows us to operate on the data while it is still in
       cache, resulting in big performance improvements.
      An application callback allows further application processing (blitting
       to video memory, color conversion, etc.) to also use the data while it's
       in cache.*/
    oc_dec_pipeline_init(_dec,&_dec->pipe);
    oc_ycbcr_buffer_flip(stripe_buf,_dec->pp_frame_buf);
    notstart=0;
    notdone=1;
    for(stripe_fragy=0;notdone;stripe_fragy+=_dec->pipe.mcu_nvfrags){
      int avail_fragy0;
      int avail_fragy_end;
      avail_fragy0=avail_fragy_end=_dec->state.fplanes[0].nvfrags;
      notdone=stripe_fragy+_dec->pipe.mcu_nvfrags<avail_fragy_end;
      for(pli=0;pli<3;pli++){
        oc_fragment_plane *fplane;
        int                frag_shift;
        int                pp_offset;
        int                sdelay;
        int                edelay;
        fplane=_dec->state.fplanes+pli;
        /*Compute the first and last fragment row of the current MCU for this
           plane.*/
        frag_shift=pli!=0&&!(_dec->state.info.pixel_fmt&2);
        _dec->pipe.fragy0[pli]=stripe_fragy>>frag_shift;
        _dec->pipe.fragy_end[pli]=OC_MINI(fplane->nvfrags,
         _dec->pipe.fragy0[pli]+(_dec->pipe.mcu_nvfrags>>frag_shift));
        oc_dec_dc_unpredict_mcu_plane(_dec,&_dec->pipe,pli);
        oc_dec_frags_recon_mcu_plane(_dec,&_dec->pipe,pli);
        sdelay=edelay=0;
        if(_dec->pipe.loop_filter){
          sdelay+=notstart;
          edelay+=notdone;
          oc_state_loop_filter_frag_rows(&_dec->state,
           _dec->pipe.bounding_values,OC_FRAME_SELF,pli,
           _dec->pipe.fragy0[pli]-sdelay,_dec->pipe.fragy_end[pli]-edelay);
        }
        /*To fill the borders, we have an additional two pixel delay, since a
           fragment in the next row could filter its top edge, using two pixels
           from a fragment in this row.
          But there's no reason to delay a full fragment between the two.*/
        oc_state_borders_fill_rows(&_dec->state,refi,pli,
         (_dec->pipe.fragy0[pli]-sdelay<<3)-(sdelay<<1),
         (_dec->pipe.fragy_end[pli]-edelay<<3)-(edelay<<1));
        /*Out-of-loop post-processing.*/
        pp_offset=3*(pli!=0);
        if(_dec->pipe.pp_level>=OC_PP_LEVEL_DEBLOCKY+pp_offset){
          /*Perform de-blocking in one plane.*/
          sdelay+=notstart;
          edelay+=notdone;
          oc_dec_deblock_frag_rows(_dec,_dec->pp_frame_buf,
           _dec->state.ref_frame_bufs[refi],pli,
           _dec->pipe.fragy0[pli]-sdelay,_dec->pipe.fragy_end[pli]-edelay);
          if(_dec->pipe.pp_level>=OC_PP_LEVEL_DERINGY+pp_offset){
            /*Perform de-ringing in one plane.*/
            sdelay+=notstart;
            edelay+=notdone;
            oc_dec_dering_frag_rows(_dec,_dec->pp_frame_buf,pli,
             _dec->pipe.fragy0[pli]-sdelay,_dec->pipe.fragy_end[pli]-edelay);
          }
        }
        /*If no post-processing is done, we still need to delay a row for the
           loop filter, thanks to the strange filtering order VP3 chose.*/
        else if(_dec->pipe.loop_filter){
          sdelay+=notstart;
          edelay+=notdone;
        }
        /*Compute the intersection of the available rows in all planes.
          If chroma is sub-sampled, the effect of each of its delays is
           doubled, but luma might have more post-processing filters enabled
           than chroma, so we don't know up front which one is the limiting
           factor.*/
        avail_fragy0=OC_MINI(avail_fragy0,
         _dec->pipe.fragy0[pli]-sdelay<<frag_shift);
        avail_fragy_end=OC_MINI(avail_fragy_end,
         _dec->pipe.fragy_end[pli]-edelay<<frag_shift);
      }
#ifdef HAVE_CAIRO
      if(_dec->stripe_cb.stripe_decoded!=NULL&&!telemetry){
#else
      if(_dec->stripe_cb.stripe_decoded!=NULL){
#endif
        /*The callback might want to use the FPU, so let's make sure they can.
          We violate all kinds of ABI restrictions by not doing this until
           now, but none of them actually matter since we don't use floating
           point ourselves.*/
        oc_restore_fpu(&_dec->state);
        /*Make the callback, ensuring we flip the sense of the "start" and
           "end" of the available region upside down.*/
        (*_dec->stripe_cb.stripe_decoded)(_dec->stripe_cb.ctx,stripe_buf,
         _dec->state.fplanes[0].nvfrags-avail_fragy_end,
         _dec->state.fplanes[0].nvfrags-avail_fragy0);
      }
      notstart=1;
    }
    /*Finish filling in the reference frame borders.*/
    for(pli=0;pli<3;pli++)oc_state_borders_fill_caps(&_dec->state,refi,pli);
    /*Update the reference frame indices.*/
    if(_dec->state.frame_type==OC_INTRA_FRAME){
      /*The new frame becomes both the previous and gold reference frames.*/
      _dec->state.ref_frame_idx[OC_FRAME_GOLD]=
       _dec->state.ref_frame_idx[OC_FRAME_PREV]=
       _dec->state.ref_frame_idx[OC_FRAME_SELF];
      _dec->state.ref_frame_data[OC_FRAME_GOLD]=
       _dec->state.ref_frame_data[OC_FRAME_PREV]=
       _dec->state.ref_frame_data[OC_FRAME_SELF];
    }
    else{
      /*Otherwise, just replace the previous reference frame.*/
      _dec->state.ref_frame_idx[OC_FRAME_PREV]=
       _dec->state.ref_frame_idx[OC_FRAME_SELF];
      _dec->state.ref_frame_data[OC_FRAME_PREV]=
       _dec->state.ref_frame_data[OC_FRAME_SELF];
    }
    /*Restore the FPU before dump_frame, since that _does_ use the FPU (for PNG
       gamma values, if nothing else).*/
    oc_restore_fpu(&_dec->state);
#ifdef HAVE_CAIRO
    /*If telemetry ioctls are active, we need to draw to the output buffer.*/
    if(telemetry){
      oc_render_telemetry(_dec,stripe_buf,telemetry);
      oc_ycbcr_buffer_flip(_dec->pp_frame_buf,stripe_buf);
      /*If we had a striped decoding callback, we skipped calling it above
         (because the telemetry wasn't rendered yet).
        Call it now with the whole frame.*/
      if(_dec->stripe_cb.stripe_decoded!=NULL){
        (*_dec->stripe_cb.stripe_decoded)(_dec->stripe_cb.ctx,
         stripe_buf,0,_dec->state.fplanes[0].nvfrags);
      }
    }
#endif
#if defined(OC_DUMP_IMAGES)
    /*We only dump images if there were some coded blocks.*/
    oc_state_dump_frame(&_dec->state,OC_FRAME_SELF,"dec");
#endif
    return 0;
  }
}

int th_decode_ycbcr_out(th_dec_ctx *_dec,th_ycbcr_buffer _ycbcr){
  if(_dec==NULL||_ycbcr==NULL)return TH_EFAULT;
  oc_ycbcr_buffer_flip(_ycbcr,_dec->pp_frame_buf);
  return 0;
}
