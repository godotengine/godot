/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function: mode selection code
  last mod: $Id$

 ********************************************************************/
#include <limits.h>
#include <string.h>
#include "encint.h"
#include "modedec.h"



typedef struct oc_fr_state           oc_fr_state;
typedef struct oc_qii_state          oc_qii_state;
typedef struct oc_enc_pipeline_state oc_enc_pipeline_state;
typedef struct oc_rd_metric          oc_rd_metric;
typedef struct oc_mode_choice        oc_mode_choice;



/*There are 8 possible schemes used to encode macro block modes.
  Schemes 0-6 use a maximally-skewed Huffman code to code each of the modes.
  The same set of Huffman codes is used for each of these 7 schemes, but the
   mode assigned to each codeword varies.
  Scheme 0 writes a custom mapping from codeword to MB mode to the bitstream,
   while schemes 1-6 have a fixed mapping.
  Scheme 7 just encodes each mode directly in 3 bits.*/

/*The mode orderings for the various mode coding schemes.
  Scheme 0 uses a custom alphabet, which is not stored in this table.
  This is the inverse of the equivalent table OC_MODE_ALPHABETS in the
   decoder.*/
static const unsigned char OC_MODE_RANKS[7][OC_NMODES]={
  /*Last MV dominates.*/ 
  /*L P M N I G GM 4*/
  {3,4,2,0,1,5,6,7},
  /*L P N M I G GM 4*/
  {2,4,3,0,1,5,6,7},
  /*L M P N I G GM 4*/
  {3,4,1,0,2,5,6,7},
  /*L M N P I G GM 4*/
  {2,4,1,0,3,5,6,7},
  /*No MV dominates.*/
  /*N L P M I G GM 4*/
  {0,4,3,1,2,5,6,7},
  /*N G L P M I GM 4*/
  {0,5,4,2,3,1,6,7},
  /*Default ordering.*/
  /*N I M L P G GM 4*/
  {0,1,2,3,4,5,6,7}
};



/*Initialize the mode scheme chooser.
  This need only be called once per encoder.*/
void oc_mode_scheme_chooser_init(oc_mode_scheme_chooser *_chooser){
  int si;
  _chooser->mode_ranks[0]=_chooser->scheme0_ranks;
  for(si=1;si<8;si++)_chooser->mode_ranks[si]=OC_MODE_RANKS[si-1];
}

/*Reset the mode scheme chooser.
  This needs to be called once for each frame, including the first.*/
static void oc_mode_scheme_chooser_reset(oc_mode_scheme_chooser *_chooser){
  int si;
  memset(_chooser->mode_counts,0,OC_NMODES*sizeof(*_chooser->mode_counts));
  /*Scheme 0 starts with 24 bits to store the mode list in.*/
  _chooser->scheme_bits[0]=24;
  memset(_chooser->scheme_bits+1,0,7*sizeof(*_chooser->scheme_bits));
  for(si=0;si<8;si++){
    /*Scheme 7 should always start first, and scheme 0 should always start
       last.*/
    _chooser->scheme_list[si]=7-si;
    _chooser->scheme0_list[si]=_chooser->scheme0_ranks[si]=si;
  }
}


/*This is the real purpose of this data structure: not actually selecting a
   mode scheme, but estimating the cost of coding a given mode given all the
   modes selected so far.
  This is done via opportunity cost: the cost is defined as the number of bits
   required to encode all the modes selected so far including the current one
   using the best possible scheme, minus the number of bits required to encode
   all the modes selected so far not including the current one using the best
   possible scheme.
  The computational expense of doing this probably makes it overkill.
  Just be happy we take a greedy approach instead of trying to solve the
   global mode-selection problem (which is NP-hard).
  _mb_mode: The mode to determine the cost of.
  Return: The number of bits required to code this mode.*/
static int oc_mode_scheme_chooser_cost(oc_mode_scheme_chooser *_chooser,
 int _mb_mode){
  int scheme0;
  int scheme1;
  int best_bits;
  int mode_bits;
  int si;
  int scheme_bits;
  scheme0=_chooser->scheme_list[0];
  scheme1=_chooser->scheme_list[1];
  best_bits=_chooser->scheme_bits[scheme0];
  mode_bits=OC_MODE_BITS[scheme0+1>>3][_chooser->mode_ranks[scheme0][_mb_mode]];
  /*Typical case: If the difference between the best scheme and the next best
     is greater than 6 bits, then adding just one mode cannot change which
     scheme we use.*/
  if(_chooser->scheme_bits[scheme1]-best_bits>6)return mode_bits;
  /*Otherwise, check to see if adding this mode selects a different scheme as
     the best.*/
  si=1;
  best_bits+=mode_bits;
  do{
    /*For any scheme except 0, we can just use the bit cost of the mode's rank
       in that scheme.*/
    if(scheme1!=0){
      scheme_bits=_chooser->scheme_bits[scheme1]+
       OC_MODE_BITS[scheme1+1>>3][_chooser->mode_ranks[scheme1][_mb_mode]];
    }
    else{
      int ri;
      /*For scheme 0, incrementing the mode count could potentially change the
         mode's rank.
        Find the index where the mode would be moved to in the optimal list,
         and use its bit cost instead of the one for the mode's current
         position in the list.*/
      /*We don't recompute scheme bits; this is computing opportunity cost, not
         an update.*/
      for(ri=_chooser->scheme0_ranks[_mb_mode];ri>0&&
       _chooser->mode_counts[_mb_mode]>=
       _chooser->mode_counts[_chooser->scheme0_list[ri-1]];ri--);
      scheme_bits=_chooser->scheme_bits[0]+OC_MODE_BITS[0][ri];
    }
    if(scheme_bits<best_bits)best_bits=scheme_bits;
    if(++si>=8)break;
    scheme1=_chooser->scheme_list[si];
  }
  while(_chooser->scheme_bits[scheme1]-_chooser->scheme_bits[scheme0]<=6);
  return best_bits-_chooser->scheme_bits[scheme0];
}

/*Incrementally update the mode counts and per-scheme bit counts and re-order
   the scheme lists once a mode has been selected.
  _mb_mode: The mode that was chosen.*/
static void oc_mode_scheme_chooser_update(oc_mode_scheme_chooser *_chooser,
 int _mb_mode){
  int ri;
  int si;
  _chooser->mode_counts[_mb_mode]++;
  /*Re-order the scheme0 mode list if necessary.*/
  for(ri=_chooser->scheme0_ranks[_mb_mode];ri>0;ri--){
    int pmode;
    pmode=_chooser->scheme0_list[ri-1];
    if(_chooser->mode_counts[pmode]>=_chooser->mode_counts[_mb_mode])break;
    /*Reorder the mode ranking.*/
    _chooser->scheme0_ranks[pmode]++;
    _chooser->scheme0_list[ri]=pmode;
  }
  _chooser->scheme0_ranks[_mb_mode]=ri;
  _chooser->scheme0_list[ri]=_mb_mode;
  /*Now add the bit cost for the mode to each scheme.*/
  for(si=0;si<8;si++){
    _chooser->scheme_bits[si]+=
     OC_MODE_BITS[si+1>>3][_chooser->mode_ranks[si][_mb_mode]];
  }
  /*Finally, re-order the list of schemes.*/
  for(si=1;si<8;si++){
    int sj;
    int scheme0;
    int bits0;
    sj=si;
    scheme0=_chooser->scheme_list[si];
    bits0=_chooser->scheme_bits[scheme0];
    do{
      int scheme1;
      scheme1=_chooser->scheme_list[sj-1];
      if(bits0>=_chooser->scheme_bits[scheme1])break;
      _chooser->scheme_list[sj]=scheme1;
    }
    while(--sj>0);
    _chooser->scheme_list[sj]=scheme0;
  }
}



/*The number of bits required to encode a super block run.
  _run_count: The desired run count; must be positive and less than 4130.*/
static int oc_sb_run_bits(int _run_count){
  int i;
  for(i=0;_run_count>=OC_SB_RUN_VAL_MIN[i+1];i++);
  return OC_SB_RUN_CODE_NBITS[i];
}

/*The number of bits required to encode a block run.
  _run_count: The desired run count; must be positive and less than 30.*/
static int oc_block_run_bits(int _run_count){
  return OC_BLOCK_RUN_CODE_NBITS[_run_count-1];
}



/*State to track coded block flags and their bit cost.*/
struct oc_fr_state{
  ptrdiff_t  bits;
  unsigned   sb_partial_count:16;
  unsigned   sb_full_count:16;
  unsigned   b_coded_count_prev:8;
  unsigned   b_coded_count:8;
  unsigned   b_count:8;
  signed int sb_partial:2;
  signed int sb_full:2;
  signed int b_coded_prev:2;
  signed int b_coded:2;
};



static void oc_fr_state_init(oc_fr_state *_fr){
  _fr->bits=0;
  _fr->sb_partial_count=0;
  _fr->sb_full_count=0;
  _fr->b_coded_count_prev=0;
  _fr->b_coded_count=0;
  _fr->b_count=0;
  _fr->sb_partial=-1;
  _fr->sb_full=-1;
  _fr->b_coded_prev=-1;
  _fr->b_coded=-1;
}


static void oc_fr_state_advance_sb(oc_fr_state *_fr,
 int _sb_partial,int _sb_full){
  ptrdiff_t bits;
  int       sb_partial_count;
  int       sb_full_count;
  bits=_fr->bits;
  /*Extend the sb_partial run, or start a new one.*/
  sb_partial_count=_fr->sb_partial;
  if(_fr->sb_partial==_sb_partial){
    if(sb_partial_count>=4129){
      bits++;
      sb_partial_count=0;
    }
    else bits-=oc_sb_run_bits(sb_partial_count);
  }
  else sb_partial_count=0;
  sb_partial_count++;
  bits+=oc_sb_run_bits(sb_partial_count);
  if(!_sb_partial){
    /*Extend the sb_full run, or start a new one.*/
    sb_full_count=_fr->sb_full_count;
    if(_fr->sb_full==_sb_full){
      if(sb_full_count>=4129){
        bits++;
        sb_full_count=0;
      }
      else bits-=oc_sb_run_bits(sb_full_count);
    }
    else sb_full_count=0;
    sb_full_count++;
    bits+=oc_sb_run_bits(sb_full_count);
    _fr->sb_full=_sb_full;
    _fr->sb_full_count=sb_full_count;
  }
  _fr->bits=bits;
  _fr->sb_partial=_sb_partial;
  _fr->sb_partial_count=sb_partial_count;
}

/*Flush any outstanding block flags for a SB (e.g., one with fewer than 16
   blocks).*/
static void oc_fr_state_flush_sb(oc_fr_state *_fr){
  ptrdiff_t bits;
  int       sb_partial;
  int       sb_full=sb_full;
  int       b_coded_count;
  int       b_coded;
  int       b_count;
  b_count=_fr->b_count;
  if(b_count>0){
    bits=_fr->bits;
    b_coded=_fr->b_coded;
    b_coded_count=_fr->b_coded_count;
    if(b_coded_count>=b_count){
      /*This SB was fully coded/uncoded; roll back the partial block flags.*/
      bits-=oc_block_run_bits(b_coded_count);
      if(b_coded_count>b_count)bits+=oc_block_run_bits(b_coded_count-b_count);
      sb_partial=0;
      sb_full=b_coded;
      b_coded=_fr->b_coded_prev;
      b_coded_count=_fr->b_coded_count_prev;
    }
    else{
      /*It was partially coded.*/
      sb_partial=1;
      /*sb_full is unused.*/
    }
    _fr->bits=bits;
    _fr->b_coded_count=b_coded_count;
    _fr->b_coded_count_prev=b_coded_count;
    _fr->b_count=0;
    _fr->b_coded=b_coded;
    _fr->b_coded_prev=b_coded;
    oc_fr_state_advance_sb(_fr,sb_partial,sb_full);
  }
}

static void oc_fr_state_advance_block(oc_fr_state *_fr,int _b_coded){
  ptrdiff_t bits;
  int       b_coded_count;
  int       b_count;
  int       sb_partial;
  int       sb_full=sb_full;
  bits=_fr->bits;
  /*Extend the b_coded run, or start a new one.*/
  b_coded_count=_fr->b_coded_count;
  if(_fr->b_coded==_b_coded)bits-=oc_block_run_bits(b_coded_count);
  else b_coded_count=0;
  b_coded_count++;
  b_count=_fr->b_count+1;
  if(b_count>=16){
    /*We finished a superblock.*/
    if(b_coded_count>=16){
      /*It was fully coded/uncoded; roll back the partial block flags.*/
      if(b_coded_count>16)bits+=oc_block_run_bits(b_coded_count-16);
      sb_partial=0;
      sb_full=_b_coded;
      _b_coded=_fr->b_coded_prev;
      b_coded_count=_fr->b_coded_count_prev;
    }
    else{
      bits+=oc_block_run_bits(b_coded_count);
      /*It was partially coded.*/
      sb_partial=1;
      /*sb_full is unused.*/
    }
    _fr->bits=bits;
    _fr->b_coded_count=b_coded_count;
    _fr->b_coded_count_prev=b_coded_count;
    _fr->b_count=0;
    _fr->b_coded=_b_coded;
    _fr->b_coded_prev=_b_coded;
    oc_fr_state_advance_sb(_fr,sb_partial,sb_full);
  }
  else{
    bits+=oc_block_run_bits(b_coded_count);
    _fr->bits=bits;
    _fr->b_coded_count=b_coded_count;
    _fr->b_count=b_count;
    _fr->b_coded=_b_coded;
  }
}

static void oc_fr_skip_block(oc_fr_state *_fr){
  oc_fr_state_advance_block(_fr,0);
}

static void oc_fr_code_block(oc_fr_state *_fr){
  oc_fr_state_advance_block(_fr,1);
}

static int oc_fr_cost1(const oc_fr_state *_fr){
  oc_fr_state tmp;
  ptrdiff_t   bits;
  *&tmp=*_fr;
  oc_fr_skip_block(&tmp);
  bits=tmp.bits;
  *&tmp=*_fr;
  oc_fr_code_block(&tmp);
  return (int)(tmp.bits-bits);
}

static int oc_fr_cost4(const oc_fr_state *_pre,const oc_fr_state *_post){
  oc_fr_state tmp;
  *&tmp=*_pre;
  oc_fr_skip_block(&tmp);
  oc_fr_skip_block(&tmp);
  oc_fr_skip_block(&tmp);
  oc_fr_skip_block(&tmp);
  return (int)(_post->bits-tmp.bits);
}



struct oc_qii_state{
  ptrdiff_t  bits;
  unsigned   qi01_count:14;
  signed int qi01:2;
  unsigned   qi12_count:14;
  signed int qi12:2;
};



static void oc_qii_state_init(oc_qii_state *_qs){
  _qs->bits=0;
  _qs->qi01_count=0;
  _qs->qi01=-1;
  _qs->qi12_count=0;
  _qs->qi12=-1;
}


static void oc_qii_state_advance(oc_qii_state *_qd,
 const oc_qii_state *_qs,int _qii){
  ptrdiff_t bits;
  int       qi01;
  int       qi01_count;
  int       qi12;
  int       qi12_count;
  bits=_qs->bits;
  qi01=_qii+1>>1;
  qi01_count=_qs->qi01_count;
  if(qi01==_qs->qi01){
    if(qi01_count>=4129){
      bits++;
      qi01_count=0;
    }
    else bits-=oc_sb_run_bits(qi01_count);
  }
  else qi01_count=0;
  qi01_count++;
  bits+=oc_sb_run_bits(qi01_count);
  qi12_count=_qs->qi12_count;
  if(_qii){
    qi12=_qii>>1;
    if(qi12==_qs->qi12){
      if(qi12_count>=4129){
        bits++;
        qi12_count=0;
      }
      else bits-=oc_sb_run_bits(qi12_count);
    }
    else qi12_count=0;
    qi12_count++;
    bits+=oc_sb_run_bits(qi12_count);
  }
  else qi12=_qs->qi12;
  _qd->bits=bits;
  _qd->qi01=qi01;
  _qd->qi01_count=qi01_count;
  _qd->qi12=qi12;
  _qd->qi12_count=qi12_count;
}



/*Temporary encoder state for the analysis pipeline.*/
struct oc_enc_pipeline_state{
  int                 bounding_values[256];
  oc_fr_state         fr[3];
  oc_qii_state        qs[3];
  /*Condensed dequantization tables.*/
  const ogg_uint16_t *dequant[3][3][2];
  /*Condensed quantization tables.*/
  const oc_iquant    *enquant[3][3][2];
  /*Skip SSD storage for the current MCU in each plane.*/
  unsigned           *skip_ssd[3];
  /*Coded/uncoded fragment lists for each plane for the current MCU.*/
  ptrdiff_t          *coded_fragis[3];
  ptrdiff_t          *uncoded_fragis[3];
  ptrdiff_t           ncoded_fragis[3];
  ptrdiff_t           nuncoded_fragis[3];
  /*The starting fragment for the current MCU in each plane.*/
  ptrdiff_t           froffset[3];
  /*The starting row for the current MCU in each plane.*/
  int                 fragy0[3];
  /*The ending row for the current MCU in each plane.*/
  int                 fragy_end[3];
  /*The starting superblock for the current MCU in each plane.*/
  unsigned            sbi0[3];
  /*The ending superblock for the current MCU in each plane.*/
  unsigned            sbi_end[3];
  /*The number of tokens for zzi=1 for each color plane.*/
  int                 ndct_tokens1[3];
  /*The outstanding eob_run count for zzi=1 for each color plane.*/
  int                 eob_run1[3];
  /*Whether or not the loop filter is enabled.*/
  int                 loop_filter;
};


static void oc_enc_pipeline_init(oc_enc_ctx *_enc,oc_enc_pipeline_state *_pipe){
  ptrdiff_t *coded_fragis;
  unsigned   mcu_nvsbs;
  ptrdiff_t  mcu_nfrags;
  int        hdec;
  int        vdec;
  int        pli;
  int        qii;
  int        qti;
  /*Initialize the per-plane coded block flag trackers.
    These are used for bit-estimation purposes only; the real flag bits span
     all three planes, so we can't compute them in parallel.*/
  for(pli=0;pli<3;pli++)oc_fr_state_init(_pipe->fr+pli);
  for(pli=0;pli<3;pli++)oc_qii_state_init(_pipe->qs+pli);
  /*Set up the per-plane skip SSD storage pointers.*/
  mcu_nvsbs=_enc->mcu_nvsbs;
  mcu_nfrags=mcu_nvsbs*_enc->state.fplanes[0].nhsbs*16;
  hdec=!(_enc->state.info.pixel_fmt&1);
  vdec=!(_enc->state.info.pixel_fmt&2);
  _pipe->skip_ssd[0]=_enc->mcu_skip_ssd;
  _pipe->skip_ssd[1]=_pipe->skip_ssd[0]+mcu_nfrags;
  _pipe->skip_ssd[2]=_pipe->skip_ssd[1]+(mcu_nfrags>>hdec+vdec);
  /*Set up per-plane pointers to the coded and uncoded fragments lists.
    Unlike the decoder, each planes' coded and uncoded fragment list is kept
     separate during the analysis stage; we only make the coded list for all
     three planes contiguous right before the final packet is output
     (destroying the uncoded lists, which are no longer needed).*/
  coded_fragis=_enc->state.coded_fragis;
  for(pli=0;pli<3;pli++){
    _pipe->coded_fragis[pli]=coded_fragis;
    coded_fragis+=_enc->state.fplanes[pli].nfrags;
    _pipe->uncoded_fragis[pli]=coded_fragis;
  }
  memset(_pipe->ncoded_fragis,0,sizeof(_pipe->ncoded_fragis));
  memset(_pipe->nuncoded_fragis,0,sizeof(_pipe->nuncoded_fragis));
  /*Set up condensed quantizer tables.*/
  for(pli=0;pli<3;pli++){
    for(qii=0;qii<_enc->state.nqis;qii++){
      int qi;
      qi=_enc->state.qis[qii];
      for(qti=0;qti<2;qti++){
        _pipe->dequant[pli][qii][qti]=_enc->state.dequant_tables[qi][pli][qti];
        _pipe->enquant[pli][qii][qti]=_enc->enquant_tables[qi][pli][qti];
      }
    }
  }
  /*Initialize the tokenization state.*/
  for(pli=0;pli<3;pli++){
    _pipe->ndct_tokens1[pli]=0;
    _pipe->eob_run1[pli]=0;
  }
  /*Initialize the bounding value array for the loop filter.*/
  _pipe->loop_filter=!oc_state_loop_filter_init(&_enc->state,
   _pipe->bounding_values);
}

/*Sets the current MCU stripe to super block row _sby.
  Return: A non-zero value if this was the last MCU.*/
static int oc_enc_pipeline_set_stripe(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _sby){
  const oc_fragment_plane *fplane;
  unsigned                 mcu_nvsbs;
  int                      sby_end;
  int                      notdone;
  int                      vdec;
  int                      pli;
  mcu_nvsbs=_enc->mcu_nvsbs;
  sby_end=_enc->state.fplanes[0].nvsbs;
  notdone=_sby+mcu_nvsbs<sby_end;
  if(notdone)sby_end=_sby+mcu_nvsbs;
  vdec=0;
  for(pli=0;pli<3;pli++){
    fplane=_enc->state.fplanes+pli;
    _pipe->sbi0[pli]=fplane->sboffset+(_sby>>vdec)*fplane->nhsbs;
    _pipe->fragy0[pli]=_sby<<2-vdec;
    _pipe->froffset[pli]=fplane->froffset
     +_pipe->fragy0[pli]*(ptrdiff_t)fplane->nhfrags;
    if(notdone){
      _pipe->sbi_end[pli]=fplane->sboffset+(sby_end>>vdec)*fplane->nhsbs;
      _pipe->fragy_end[pli]=sby_end<<2-vdec;
    }
    else{
      _pipe->sbi_end[pli]=fplane->sboffset+fplane->nsbs;
      _pipe->fragy_end[pli]=fplane->nvfrags;
    }
    vdec=!(_enc->state.info.pixel_fmt&2);
  }
  return notdone;
}

static void oc_enc_pipeline_finish_mcu_plane(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,int _sdelay,int _edelay){
  int refi;
  /*Copy over all the uncoded fragments from this plane and advance the uncoded
     fragment list.*/
  _pipe->uncoded_fragis[_pli]-=_pipe->nuncoded_fragis[_pli];
  oc_state_frag_copy_list(&_enc->state,_pipe->uncoded_fragis[_pli],
   _pipe->nuncoded_fragis[_pli],OC_FRAME_SELF,OC_FRAME_PREV,_pli);
  _pipe->nuncoded_fragis[_pli]=0;
  /*Perform DC prediction.*/
  oc_enc_pred_dc_frag_rows(_enc,_pli,
   _pipe->fragy0[_pli],_pipe->fragy_end[_pli]);
  /*Finish DC tokenization.*/
  oc_enc_tokenize_dc_frag_list(_enc,_pli,
   _pipe->coded_fragis[_pli],_pipe->ncoded_fragis[_pli],
   _pipe->ndct_tokens1[_pli],_pipe->eob_run1[_pli]);
  _pipe->ndct_tokens1[_pli]=_enc->ndct_tokens[_pli][1];
  _pipe->eob_run1[_pli]=_enc->eob_run[_pli][1];
  /*And advance the coded fragment list.*/
  _enc->state.ncoded_fragis[_pli]+=_pipe->ncoded_fragis[_pli];
  _pipe->coded_fragis[_pli]+=_pipe->ncoded_fragis[_pli];
  _pipe->ncoded_fragis[_pli]=0;
  /*Apply the loop filter if necessary.*/
  refi=_enc->state.ref_frame_idx[OC_FRAME_SELF];
  if(_pipe->loop_filter){
    oc_state_loop_filter_frag_rows(&_enc->state,_pipe->bounding_values,
     refi,_pli,_pipe->fragy0[_pli]-_sdelay,_pipe->fragy_end[_pli]-_edelay);
  }
  else _sdelay=_edelay=0;
  /*To fill borders, we have an additional two pixel delay, since a fragment
     in the next row could filter its top edge, using two pixels from a
     fragment in this row.
    But there's no reason to delay a full fragment between the two.*/
  oc_state_borders_fill_rows(&_enc->state,refi,_pli,
   (_pipe->fragy0[_pli]-_sdelay<<3)-(_sdelay<<1),
   (_pipe->fragy_end[_pli]-_edelay<<3)-(_edelay<<1));
}



/*Cost information about the coded blocks in a MB.*/
struct oc_rd_metric{
  int uncoded_ac_ssd;
  int coded_ac_ssd;
  int ac_bits;
  int dc_flag;
};



static int oc_enc_block_transform_quantize(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,ptrdiff_t _fragi,int _overhead_bits,
 oc_rd_metric *_mo,oc_token_checkpoint **_stack){
  OC_ALIGN16(ogg_int16_t  dct[64]);
  OC_ALIGN16(ogg_int16_t  data[64]);
  ogg_uint16_t            dc_dequant;
  const ogg_uint16_t     *dequant;
  const oc_iquant        *enquant;
  ptrdiff_t               frag_offs;
  int                     ystride;
  const unsigned char    *src;
  const unsigned char    *ref;
  unsigned char          *dst;
  int                     frame_type;
  int                     nonzero;
  unsigned                uncoded_ssd;
  unsigned                coded_ssd;
  int                     coded_dc;
  oc_token_checkpoint    *checkpoint;
  oc_fragment            *frags;
  int                     mb_mode;
  int                     mv_offs[2];
  int                     nmv_offs;
  int                     ac_bits;
  int                     borderi;
  int                     qti;
  int                     qii;
  int                     pi;
  int                     zzi;
  int                     v;
  int                     val;
  int                     d;
  int                     s;
  int                     dc;
  frags=_enc->state.frags;
  frag_offs=_enc->state.frag_buf_offs[_fragi];
  ystride=_enc->state.ref_ystride[_pli];
  src=_enc->state.ref_frame_data[OC_FRAME_IO]+frag_offs;
  borderi=frags[_fragi].borderi;
  qii=frags[_fragi].qii;
  if(qii&~3){
#if !defined(OC_COLLECT_METRICS)
    if(_enc->sp_level>=OC_SP_LEVEL_EARLY_SKIP){
      /*Enable early skip detection.*/
      frags[_fragi].coded=0;
      return 0;
    }
#endif
    /*Try and code this block anyway.*/
    qii&=3;
    frags[_fragi].qii=qii;
  }
  mb_mode=frags[_fragi].mb_mode;
  ref=_enc->state.ref_frame_data[
   _enc->state.ref_frame_idx[OC_FRAME_FOR_MODE(mb_mode)]]+frag_offs;
  dst=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[OC_FRAME_SELF]]
   +frag_offs;
  /*Motion compensation:*/
  switch(mb_mode){
    case OC_MODE_INTRA:{
      nmv_offs=0;
      oc_enc_frag_sub_128(_enc,data,src,ystride);
    }break;
    case OC_MODE_GOLDEN_NOMV:
    case OC_MODE_INTER_NOMV:{
      nmv_offs=1;
      mv_offs[0]=0;
      oc_enc_frag_sub(_enc,data,src,ref,ystride);
    }break;
    default:{
      const oc_mv *frag_mvs;
      frag_mvs=(const oc_mv *)_enc->state.frag_mvs;
      nmv_offs=oc_state_get_mv_offsets(&_enc->state,mv_offs,_pli,
       frag_mvs[_fragi][0],frag_mvs[_fragi][1]);
      if(nmv_offs>1){
        oc_enc_frag_copy2(_enc,dst,
         ref+mv_offs[0],ref+mv_offs[1],ystride);
        oc_enc_frag_sub(_enc,data,src,dst,ystride);
      }
      else oc_enc_frag_sub(_enc,data,src,ref+mv_offs[0],ystride);
    }break;
  }
#if defined(OC_COLLECT_METRICS)
  {
    unsigned satd;
    switch(nmv_offs){
      case 0:satd=oc_enc_frag_intra_satd(_enc,src,ystride);break;
      case 1:{
        satd=oc_enc_frag_satd_thresh(_enc,src,ref+mv_offs[0],ystride,UINT_MAX);
      }break;
      default:{
        satd=oc_enc_frag_satd_thresh(_enc,src,dst,ystride,UINT_MAX);
      }
    }
    _enc->frag_satd[_fragi]=satd;
  }
#endif
  /*Transform:*/
  oc_enc_fdct8x8(_enc,dct,data);
  /*Quantize the DC coefficient:*/
  qti=mb_mode!=OC_MODE_INTRA;
  enquant=_pipe->enquant[_pli][0][qti];
  dc_dequant=_pipe->dequant[_pli][0][qti][0];
  v=dct[0];
  val=v<<1;
  s=OC_SIGNMASK(val);
  val+=dc_dequant+s^s;
  val=((enquant[0].m*(ogg_int32_t)val>>16)+val>>enquant[0].l)-s;
  dc=OC_CLAMPI(-580,val,580);
  nonzero=0;
  /*Quantize the AC coefficients:*/
  dequant=_pipe->dequant[_pli][qii][qti];
  enquant=_pipe->enquant[_pli][qii][qti];
  for(zzi=1;zzi<64;zzi++){
    v=dct[OC_FZIG_ZAG[zzi]];
    d=dequant[zzi];
    val=v<<1;
    v=abs(val);
    if(v>=d){
      s=OC_SIGNMASK(val);
      /*The bias added here rounds ties away from zero, since token
         optimization can only decrease the magnitude of the quantized
         value.*/
      val+=d+s^s;
      /*Note the arithmetic right shift is not guaranteed by ANSI C.
        Hopefully no one still uses ones-complement architectures.*/
      val=((enquant[zzi].m*(ogg_int32_t)val>>16)+val>>enquant[zzi].l)-s;
      data[zzi]=OC_CLAMPI(-580,val,580);
      nonzero=zzi;
    }
    else data[zzi]=0;
  }
  /*Tokenize.*/
  checkpoint=*_stack;
  ac_bits=oc_enc_tokenize_ac(_enc,_pli,_fragi,data,dequant,dct,nonzero+1,
   _stack,qti?0:3);
  /*Reconstruct.
    TODO: nonzero may need to be adjusted after tokenization.*/
  if(nonzero==0){
    ogg_int16_t p;
    int         ci;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_int16_t)(dc*(ogg_int32_t)dc_dequant+15>>5);
    /*LOOP VECTORIZES.*/
    for(ci=0;ci<64;ci++)data[ci]=p;
  }
  else{
    data[0]=dc*dc_dequant;
    oc_idct8x8(&_enc->state,data,nonzero+1);
  }
  if(!qti)oc_enc_frag_recon_intra(_enc,dst,ystride,data);
  else{
    oc_enc_frag_recon_inter(_enc,dst,
     nmv_offs==1?ref+mv_offs[0]:dst,ystride,data);
  }
  frame_type=_enc->state.frame_type;
#if !defined(OC_COLLECT_METRICS)
  if(frame_type!=OC_INTRA_FRAME)
#endif
  {
    /*In retrospect, should we have skipped this block?*/
    oc_enc_frag_sub(_enc,data,src,dst,ystride);
    coded_ssd=coded_dc=0;
    if(borderi<0){
      for(pi=0;pi<64;pi++){
        coded_ssd+=data[pi]*data[pi];
        coded_dc+=data[pi];
      }
    }
    else{
      ogg_int64_t mask;
      mask=_enc->state.borders[borderi].mask;
      for(pi=0;pi<64;pi++,mask>>=1)if(mask&1){
        coded_ssd+=data[pi]*data[pi];
        coded_dc+=data[pi];
      }
    }
    /*Scale to match DCT domain.*/
    coded_ssd<<=4;
    /*We actually only want the AC contribution to the SSD.*/
    coded_ssd-=coded_dc*coded_dc>>2;
#if defined(OC_COLLECT_METRICS)
    _enc->frag_ssd[_fragi]=coded_ssd;
  }
  if(frame_type!=OC_INTRA_FRAME){
#endif
    uncoded_ssd=_pipe->skip_ssd[_pli][_fragi-_pipe->froffset[_pli]];
    if(uncoded_ssd<UINT_MAX){
      /*Although the fragment coding overhead determination is accurate, it is
         greedy, using very coarse-grained local information.
        Allowing it to mildly discourage coding turns out to be beneficial, but
         it's not clear that allowing it to encourage coding through negative
         coding overhead deltas is useful.
        For that reason, we disallow negative coding_overheads.*/
      if(_overhead_bits<0)_overhead_bits=0;
      if(uncoded_ssd<=coded_ssd+(_overhead_bits+ac_bits)*_enc->lambda&&
       /*Don't allow luma blocks to be skipped in 4MV mode when VP3
          compatibility is enabled.*/
       (!_enc->vp3_compatible||mb_mode!=OC_MODE_INTER_MV_FOUR||_pli)){
        /*Hm, not worth it; roll back.*/
        oc_enc_tokenlog_rollback(_enc,checkpoint,(*_stack)-checkpoint);
        *_stack=checkpoint;
        frags[_fragi].coded=0;
        return 0;
      }
    }
    else _mo->dc_flag=1;
    _mo->uncoded_ac_ssd+=uncoded_ssd;
    _mo->coded_ac_ssd+=coded_ssd;
    _mo->ac_bits+=ac_bits;
  }
  oc_qii_state_advance(_pipe->qs+_pli,_pipe->qs+_pli,qii);
  frags[_fragi].dc=dc;
  frags[_fragi].coded=1;
  return 1;
}

static int oc_enc_mb_transform_quantize_luma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,unsigned _mbi,int _mode_overhead){
  /*Worst case token stack usage for 4 fragments.*/
  oc_token_checkpoint  stack[64*4];
  oc_token_checkpoint *stackptr;
  const oc_sb_map     *sb_maps;
  signed char         *mb_modes;
  oc_fragment         *frags;
  ptrdiff_t           *coded_fragis;
  ptrdiff_t            ncoded_fragis;
  ptrdiff_t           *uncoded_fragis;
  ptrdiff_t            nuncoded_fragis;
  oc_rd_metric         mo;
  oc_fr_state          fr_checkpoint;
  oc_qii_state         qs_checkpoint;
  int                  mb_mode;
  int                  ncoded;
  ptrdiff_t            fragi;
  int                  bi;
  *&fr_checkpoint=*(_pipe->fr+0);
  *&qs_checkpoint=*(_pipe->qs+0);
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  mb_modes=_enc->state.mb_modes;
  frags=_enc->state.frags;
  coded_fragis=_pipe->coded_fragis[0];
  ncoded_fragis=_pipe->ncoded_fragis[0];
  uncoded_fragis=_pipe->uncoded_fragis[0];
  nuncoded_fragis=_pipe->nuncoded_fragis[0];
  mb_mode=mb_modes[_mbi];
  ncoded=0;
  stackptr=stack;
  memset(&mo,0,sizeof(mo));
  for(bi=0;bi<4;bi++){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frags[fragi].mb_mode=mb_mode;
    if(oc_enc_block_transform_quantize(_enc,
     _pipe,0,fragi,oc_fr_cost1(_pipe->fr+0),&mo,&stackptr)){
      oc_fr_code_block(_pipe->fr+0);
      coded_fragis[ncoded_fragis++]=fragi;
      ncoded++;
    }
    else{
      *(uncoded_fragis-++nuncoded_fragis)=fragi;
      oc_fr_skip_block(_pipe->fr+0);
    }
  }
  if(_enc->state.frame_type!=OC_INTRA_FRAME){
    if(ncoded>0&&!mo.dc_flag){
      int cost;
      /*Some individual blocks were worth coding.
        See if that's still true when accounting for mode and MV overhead.*/
      cost=mo.coded_ac_ssd+_enc->lambda*(mo.ac_bits
       +oc_fr_cost4(&fr_checkpoint,_pipe->fr+0)+_mode_overhead);
      if(mo.uncoded_ac_ssd<=cost){
        /*Taking macroblock overhead into account, it is not worth coding this
           MB.*/
        oc_enc_tokenlog_rollback(_enc,stack,stackptr-stack);
        *(_pipe->fr+0)=*&fr_checkpoint;
        *(_pipe->qs+0)=*&qs_checkpoint;
        for(bi=0;bi<4;bi++){
          fragi=sb_maps[_mbi>>2][_mbi&3][bi];
          if(frags[fragi].coded){
            *(uncoded_fragis-++nuncoded_fragis)=fragi;
            frags[fragi].coded=0;
          }
          oc_fr_skip_block(_pipe->fr+0);
        }
        ncoded_fragis-=ncoded;
        ncoded=0;
      }
    }
    /*If no luma blocks coded, the mode is forced.*/
    if(ncoded==0)mb_modes[_mbi]=OC_MODE_INTER_NOMV;
    /*Assume that a 1MV with a single coded block is always cheaper than a 4MV
       with a single coded block.
      This may not be strictly true: a 4MV computes chroma MVs using (0,0) for
       skipped blocks, while a 1MV does not.*/
    else if(ncoded==1&&mb_mode==OC_MODE_INTER_MV_FOUR){
      mb_modes[_mbi]=OC_MODE_INTER_MV;
    }
  }
  _pipe->ncoded_fragis[0]=ncoded_fragis;
  _pipe->nuncoded_fragis[0]=nuncoded_fragis;
  return ncoded;
}

static void oc_enc_sb_transform_quantize_chroma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,int _sbi_start,int _sbi_end){
  const oc_sb_map *sb_maps;
  oc_sb_flags     *sb_flags;
  ptrdiff_t       *coded_fragis;
  ptrdiff_t        ncoded_fragis;
  ptrdiff_t       *uncoded_fragis;
  ptrdiff_t        nuncoded_fragis;
  int              sbi;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  sb_flags=_enc->state.sb_flags;
  coded_fragis=_pipe->coded_fragis[_pli];
  ncoded_fragis=_pipe->ncoded_fragis[_pli];
  uncoded_fragis=_pipe->uncoded_fragis[_pli];
  nuncoded_fragis=_pipe->nuncoded_fragis[_pli];
  for(sbi=_sbi_start;sbi<_sbi_end;sbi++){
    /*Worst case token stack usage for 1 fragment.*/
    oc_token_checkpoint stack[64];
    oc_rd_metric        mo;
    int                 quadi;
    int                 bi;
    memset(&mo,0,sizeof(mo));
    for(quadi=0;quadi<4;quadi++)for(bi=0;bi<4;bi++){
      ptrdiff_t fragi;
      fragi=sb_maps[sbi][quadi][bi];
      if(fragi>=0){
        oc_token_checkpoint *stackptr;
        stackptr=stack;
        if(oc_enc_block_transform_quantize(_enc,
         _pipe,_pli,fragi,oc_fr_cost1(_pipe->fr+_pli),&mo,&stackptr)){
          coded_fragis[ncoded_fragis++]=fragi;
          oc_fr_code_block(_pipe->fr+_pli);
        }
        else{
          *(uncoded_fragis-++nuncoded_fragis)=fragi;
          oc_fr_skip_block(_pipe->fr+_pli);
        }
      }
    }
    oc_fr_state_flush_sb(_pipe->fr+_pli);
    sb_flags[sbi].coded_fully=_pipe->fr[_pli].sb_full;
    sb_flags[sbi].coded_partially=_pipe->fr[_pli].sb_partial;
  }
  _pipe->ncoded_fragis[_pli]=ncoded_fragis;
  _pipe->nuncoded_fragis[_pli]=nuncoded_fragis;
}

/*Mode decision is done by exhaustively examining all potential choices.
  Obviously, doing the motion compensation, fDCT, tokenization, and then
   counting the bits each token uses is computationally expensive.
  Theora's EOB runs can also split the cost of these tokens across multiple
   fragments, and naturally we don't know what the optimal choice of Huffman
   codes will be until we know all the tokens we're going to encode in all the
   fragments.
  So we use a simple approach to estimating the bit cost and distortion of each
   mode based upon the SATD value of the residual before coding.
  The mathematics behind the technique are outlined by Kim \cite{Kim03}, but
   the process (modified somewhat from that of the paper) is very simple.
  We build a non-linear regression of the mappings from
   (pre-transform+quantization) SATD to (post-transform+quantization) bits and
   SSD for each qi.
  A separate set of mappings is kept for each quantization type and color
   plane.
  The mappings are constructed by partitioning the SATD values into a small
   number of bins (currently 24) and using a linear regression in each bin
   (as opposed to the 0th-order regression used by Kim).
  The bit counts and SSD measurements are obtained by examining actual encoded
   frames, with appropriate lambda values and optimal Huffman codes selected.
  EOB bits are assigned to the fragment that started the EOB run (as opposed to
   dividing them among all the blocks in the run; though the latter approach
   seems more theoretically correct, Monty's testing showed a small improvement
   with the former, though that may have been merely statistical noise).

  @ARTICLE{Kim03,
    author="Hyun Mun Kim",
    title="Adaptive Rate Control Using Nonlinear Regression",
    journal="IEEE Transactions on Circuits and Systems for Video Technology",
    volume=13,
    number=5,
    pages="432--439",
    month=May,
    year=2003
  }*/

/*Computes (_ssd+_lambda*_rate)/(1<<OC_BIT_SCALE) with rounding, avoiding
   overflow for large lambda values.*/
#define OC_MODE_RD_COST(_ssd,_rate,_lambda) \
 ((_ssd)>>OC_BIT_SCALE)+((_rate)>>OC_BIT_SCALE)*(_lambda) \
 +(((_ssd)&(1<<OC_BIT_SCALE)-1)+((_rate)&(1<<OC_BIT_SCALE)-1)*(_lambda) \
 +((1<<OC_BIT_SCALE)>>1)>>OC_BIT_SCALE)

/*Estimate the R-D cost of the DCT coefficients given the SATD of a block after
   prediction.*/
static unsigned oc_dct_cost2(unsigned *_ssd,
 int _qi,int _pli,int _qti,int _satd){
  unsigned rmse;
  int      bin;
  int      dx;
  int      y0;
  int      z0;
  int      dy;
  int      dz;
  /*SATD metrics for chroma planes vary much less than luma, so we scale them
     by 4 to distribute them into the mode decision bins more evenly.*/
  _satd<<=_pli+1&2;
  bin=OC_MINI(_satd>>OC_SAD_SHIFT,OC_SAD_BINS-2);
  dx=_satd-(bin<<OC_SAD_SHIFT);
  y0=OC_MODE_RD[_qi][_pli][_qti][bin].rate;
  z0=OC_MODE_RD[_qi][_pli][_qti][bin].rmse;
  dy=OC_MODE_RD[_qi][_pli][_qti][bin+1].rate-y0;
  dz=OC_MODE_RD[_qi][_pli][_qti][bin+1].rmse-z0;
  rmse=OC_MAXI(z0+(dz*dx>>OC_SAD_SHIFT),0);
  *_ssd=rmse*rmse>>2*OC_RMSE_SCALE-OC_BIT_SCALE;
  return OC_MAXI(y0+(dy*dx>>OC_SAD_SHIFT),0);
}

/*Select luma block-level quantizers for a MB in an INTRA frame.*/
static unsigned oc_analyze_intra_mb_luma(oc_enc_ctx *_enc,
 const oc_qii_state *_qs,unsigned _mbi){
  const unsigned char *src;
  const ptrdiff_t     *frag_buf_offs;
  const oc_sb_map     *sb_maps;
  oc_fragment         *frags;
  ptrdiff_t            frag_offs;
  ptrdiff_t            fragi;
  oc_qii_state         qs[4][3];
  unsigned             cost[4][3];
  unsigned             ssd[4][3];
  unsigned             rate[4][3];
  int                  prev[3][3];
  unsigned             satd;
  unsigned             best_cost;
  unsigned             best_ssd;
  unsigned             best_rate;
  int                  best_qii;
  int                  qii;
  int                  lambda;
  int                  ystride;
  int                  nqis;
  int                  bi;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[0];
  fragi=sb_maps[_mbi>>2][_mbi&3][0];
  frag_offs=frag_buf_offs[fragi];
  satd=oc_enc_frag_intra_satd(_enc,src+frag_offs,ystride);
  nqis=_enc->state.nqis;
  lambda=_enc->lambda;
  for(qii=0;qii<nqis;qii++){
    oc_qii_state_advance(qs[0]+qii,_qs,qii);
    rate[0][qii]=oc_dct_cost2(ssd[0]+qii,_enc->state.qis[qii],0,0,satd)
     +(qs[0][qii].bits-_qs->bits<<OC_BIT_SCALE);
    cost[0][qii]=OC_MODE_RD_COST(ssd[0][qii],rate[0][qii],lambda);
  }
  for(bi=1;bi<4;bi++){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frag_offs=frag_buf_offs[fragi];
    satd=oc_enc_frag_intra_satd(_enc,src+frag_offs,ystride);
    for(qii=0;qii<nqis;qii++){
      oc_qii_state qt[3];
      unsigned     cur_ssd;
      unsigned     cur_rate;
      int          best_qij;
      int          qij;
      oc_qii_state_advance(qt+0,qs[bi-1]+0,qii);
      cur_rate=oc_dct_cost2(&cur_ssd,_enc->state.qis[qii],0,0,satd);
      best_ssd=ssd[bi-1][0]+cur_ssd;
      best_rate=rate[bi-1][0]+cur_rate
       +(qt[0].bits-qs[bi-1][0].bits<<OC_BIT_SCALE);
      best_cost=OC_MODE_RD_COST(best_ssd,best_rate,lambda);
      best_qij=0;
      for(qij=1;qij<nqis;qij++){
        unsigned chain_ssd;
        unsigned chain_rate;
        unsigned chain_cost;
        oc_qii_state_advance(qt+qij,qs[bi-1]+qij,qii);
        chain_ssd=ssd[bi-1][qij]+cur_ssd;
        chain_rate=rate[bi-1][qij]+cur_rate
         +(qt[qij].bits-qs[bi-1][qij].bits<<OC_BIT_SCALE);
        chain_cost=OC_MODE_RD_COST(chain_ssd,chain_rate,lambda);
        if(chain_cost<best_cost){
          best_cost=chain_cost;
          best_ssd=chain_ssd;
          best_rate=chain_rate;
          best_qij=qij;
        }
      }
      *(qs[bi]+qii)=*(qt+best_qij);
      cost[bi][qii]=best_cost;
      ssd[bi][qii]=best_ssd;
      rate[bi][qii]=best_rate;
      prev[bi-1][qii]=best_qij;
    }
  }
  best_qii=0;
  best_cost=cost[3][0];
  for(qii=1;qii<nqis;qii++){
    if(cost[3][qii]<best_cost){
      best_cost=cost[3][qii];
      best_qii=qii;
    }
  }
  frags=_enc->state.frags;
  for(bi=3;;){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frags[fragi].qii=best_qii;
    if(bi--<=0)break;
    best_qii=prev[bi][best_qii];
  }
  return best_cost;
}

/*Select a block-level quantizer for a single chroma block in an INTRA frame.*/
static unsigned oc_analyze_intra_chroma_block(oc_enc_ctx *_enc,
 const oc_qii_state *_qs,int _pli,ptrdiff_t _fragi){
  const unsigned char *src;
  oc_fragment         *frags;
  ptrdiff_t            frag_offs;
  oc_qii_state         qt[3];
  unsigned             cost[3];
  unsigned             satd;
  unsigned             best_cost;
  int                  best_qii;
  int                  qii;
  int                  lambda;
  int                  ystride;
  int                  nqis;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[_pli];
  frag_offs=_enc->state.frag_buf_offs[_fragi];
  satd=oc_enc_frag_intra_satd(_enc,src+frag_offs,ystride);
  nqis=_enc->state.nqis;
  lambda=_enc->lambda;
  best_qii=0;
  for(qii=0;qii<nqis;qii++){
    unsigned cur_rate;
    unsigned cur_ssd;
    oc_qii_state_advance(qt+qii,_qs,qii);
    cur_rate=oc_dct_cost2(&cur_ssd,_enc->state.qis[qii],_pli,0,satd)
     +(qt[qii].bits-_qs->bits<<OC_BIT_SCALE);
    cost[qii]=OC_MODE_RD_COST(cur_ssd,cur_rate,lambda);
  }
  best_cost=cost[0];
  for(qii=1;qii<nqis;qii++){
    if(cost[qii]<best_cost){
      best_cost=cost[qii];
      best_qii=qii;
    }
  }
  frags=_enc->state.frags;
  frags[_fragi].qii=best_qii;
  return best_cost;
}

static void oc_enc_sb_transform_quantize_intra_chroma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,int _sbi_start,int _sbi_end){
  const oc_sb_map *sb_maps;
  oc_sb_flags     *sb_flags;
  ptrdiff_t       *coded_fragis;
  ptrdiff_t        ncoded_fragis;
  int              sbi;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  sb_flags=_enc->state.sb_flags;
  coded_fragis=_pipe->coded_fragis[_pli];
  ncoded_fragis=_pipe->ncoded_fragis[_pli];
  for(sbi=_sbi_start;sbi<_sbi_end;sbi++){
    /*Worst case token stack usage for 1 fragment.*/
    oc_token_checkpoint stack[64];
    int                 quadi;
    int                 bi;
    for(quadi=0;quadi<4;quadi++)for(bi=0;bi<4;bi++){
      ptrdiff_t fragi;
      fragi=sb_maps[sbi][quadi][bi];
      if(fragi>=0){
        oc_token_checkpoint *stackptr;
        oc_analyze_intra_chroma_block(_enc,_pipe->qs+_pli,_pli,fragi);
        stackptr=stack;
        oc_enc_block_transform_quantize(_enc,
         _pipe,_pli,fragi,0,NULL,&stackptr);
        coded_fragis[ncoded_fragis++]=fragi;
      }
    }
  }
  _pipe->ncoded_fragis[_pli]=ncoded_fragis;
}

/*Analysis stage for an INTRA frame.*/
void oc_enc_analyze_intra(oc_enc_ctx *_enc,int _recode){
  oc_enc_pipeline_state   pipe;
  const unsigned char    *map_idxs;
  int                     nmap_idxs;
  oc_sb_flags            *sb_flags;
  signed char            *mb_modes;
  const oc_mb_map        *mb_maps;
  oc_mb_enc_info         *embs;
  oc_fragment            *frags;
  unsigned                stripe_sby;
  unsigned                mcu_nvsbs;
  int                     notstart;
  int                     notdone;
  int                     refi;
  int                     pli;
  _enc->state.frame_type=OC_INTRA_FRAME;
  oc_enc_tokenize_start(_enc);
  oc_enc_pipeline_init(_enc,&pipe);
  /*Choose MVs and MB modes and quantize and code luma.
    Must be done in Hilbert order.*/
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  nmap_idxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  _enc->state.ncoded_fragis[0]=0;
  _enc->state.ncoded_fragis[1]=0;
  _enc->state.ncoded_fragis[2]=0;
  sb_flags=_enc->state.sb_flags;
  mb_modes=_enc->state.mb_modes;
  mb_maps=(const oc_mb_map *)_enc->state.mb_maps;
  embs=_enc->mb_info;
  frags=_enc->state.frags;
  notstart=0;
  notdone=1;
  mcu_nvsbs=_enc->mcu_nvsbs;
  for(stripe_sby=0;notdone;stripe_sby+=mcu_nvsbs){
    unsigned sbi;
    unsigned sbi_end;
    notdone=oc_enc_pipeline_set_stripe(_enc,&pipe,stripe_sby);
    sbi_end=pipe.sbi_end[0];
    for(sbi=pipe.sbi0[0];sbi<sbi_end;sbi++){
      int quadi;
      /*Mode addressing is through Y plane, always 4 MB per SB.*/
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        unsigned  mbi;
        int       mapii;
        int       mapi;
        int       bi;
        ptrdiff_t fragi;
        mbi=sbi<<2|quadi;
        /*Motion estimation:
          We always do a basic 1MV search for all macroblocks, coded or not,
           keyframe or not.*/
        if(!_recode&&_enc->state.curframe_num>0)oc_mcenc_search(_enc,mbi);
        oc_analyze_intra_mb_luma(_enc,pipe.qs+0,mbi);
        mb_modes[mbi]=OC_MODE_INTRA;
        oc_enc_mb_transform_quantize_luma(_enc,&pipe,mbi,0);
        /*Propagate final MB mode and MVs to the chroma blocks.*/
        for(mapii=4;mapii<nmap_idxs;mapii++){
          mapi=map_idxs[mapii];
          pli=mapi>>2;
          bi=mapi&3;
          fragi=mb_maps[mbi][pli][bi];
          frags[fragi].mb_mode=OC_MODE_INTRA;
        }
      }
    }
    oc_enc_pipeline_finish_mcu_plane(_enc,&pipe,0,notstart,notdone);
    /*Code chroma planes.*/
    for(pli=1;pli<3;pli++){
      oc_enc_sb_transform_quantize_intra_chroma(_enc,&pipe,
       pli,pipe.sbi0[pli],pipe.sbi_end[pli]);
      oc_enc_pipeline_finish_mcu_plane(_enc,&pipe,pli,notstart,notdone);
    }
    notstart=1;
  }
  /*Finish filling in the reference frame borders.*/
  refi=_enc->state.ref_frame_idx[OC_FRAME_SELF];
  for(pli=0;pli<3;pli++)oc_state_borders_fill_caps(&_enc->state,refi,pli);
  _enc->state.ntotal_coded_fragis=_enc->state.nfrags;
}



/*Cost information about a MB mode.*/
struct oc_mode_choice{
  unsigned      cost;
  unsigned      ssd;
  unsigned      rate;
  unsigned      overhead;
  unsigned char qii[12];
};



static void oc_mode_set_cost(oc_mode_choice *_modec,int _lambda){
  _modec->cost=OC_MODE_RD_COST(_modec->ssd,
   _modec->rate+_modec->overhead,_lambda);
}

/*A set of skip SSD's to use to disable early skipping.*/
static const unsigned OC_NOSKIP[12]={
  UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,
  UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,
  UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX
};

/*The estimated number of bits used by a coded chroma block to specify the AC
   quantizer.
  TODO: Currently this is just 0.5*log2(3) (estimating about 50% compression);
   measurements suggest this is in the right ballpark, but it varies somewhat
   with lambda.*/
#define OC_CHROMA_QII_RATE ((0xCAE00D1DU>>31-OC_BIT_SCALE)+1>>1)

static void oc_analyze_mb_mode_luma(oc_enc_ctx *_enc,
 oc_mode_choice *_modec,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12],int _qti){
  oc_fr_state  fr;
  oc_qii_state qs;
  unsigned     ssd;
  unsigned     rate;
  int          overhead;
  unsigned     satd;
  unsigned     best_ssd;
  unsigned     best_rate;
  int          best_overhead;
  int          best_fri;
  int          best_qii;
  unsigned     cur_cost;
  unsigned     cur_ssd;
  unsigned     cur_rate;
  int          cur_overhead;
  int          lambda;
  int          nqis;
  int          nskipped;
  int          bi;
  int          qii;
  lambda=_enc->lambda;
  nqis=_enc->state.nqis;
  /*We could do a trellis optimization here, but we don't make final skip
     decisions until after transform+quantization, so the result wouldn't be
     optimal anyway.
    Instead we just use a greedy approach; for most SATD values, the
     differences between the qiis are large enough to drown out the cost to
     code the flags, anyway.*/
  *&fr=*_fr;
  *&qs=*_qs;
  ssd=rate=overhead=nskipped=0;
  for(bi=0;bi<4;bi++){
    oc_fr_state  ft[2];
    oc_qii_state qt[3];
    unsigned     best_cost;
    satd=_frag_satd[bi];
    *(ft+0)=*&fr;
    oc_fr_code_block(ft+0);
    oc_qii_state_advance(qt+0,&qs,0);
    best_overhead=(ft[0].bits-fr.bits<<OC_BIT_SCALE);
    best_rate=oc_dct_cost2(&best_ssd,_enc->state.qis[0],0,_qti,satd)
     +(qt[0].bits-qs.bits<<OC_BIT_SCALE);
    best_cost=OC_MODE_RD_COST(ssd+best_ssd,rate+best_rate+best_overhead,lambda);
    best_fri=0;
    best_qii=0;
    for(qii=1;qii<nqis;qii++){
      oc_qii_state_advance(qt+qii,&qs,qii);
      cur_rate=oc_dct_cost2(&cur_ssd,_enc->state.qis[qii],0,_qti,satd)
       +(qt[qii].bits-qs.bits<<OC_BIT_SCALE);
      cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_rate+best_overhead,lambda);
      if(cur_cost<best_cost){
        best_cost=cur_cost;
        best_ssd=cur_ssd;
        best_rate=cur_rate;
        best_qii=qii;
      }
    }
    if(_skip_ssd[bi]<UINT_MAX&&nskipped<3){
      *(ft+1)=*&fr;
      oc_fr_skip_block(ft+1);
      cur_overhead=ft[1].bits-fr.bits<<OC_BIT_SCALE;
      cur_ssd=_skip_ssd[bi]<<OC_BIT_SCALE;
      cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_overhead,lambda);
      if(cur_cost<=best_cost){
        best_ssd=cur_ssd;
        best_rate=0;
        best_overhead=cur_overhead;
        best_fri=1;
        best_qii+=4;
      }
    }
    rate+=best_rate;
    ssd+=best_ssd;
    overhead+=best_overhead;
    *&fr=*(ft+best_fri);
    if(best_fri==0)*&qs=*(qt+best_qii);
    else nskipped++;
    _modec->qii[bi]=best_qii;
  }
  _modec->ssd=ssd;
  _modec->rate=rate;
  _modec->overhead=OC_MAXI(overhead,0);
}

static void oc_analyze_mb_mode_chroma(oc_enc_ctx *_enc,
 oc_mode_choice *_modec,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12],int _qti){
  unsigned ssd;
  unsigned rate;
  unsigned satd;
  unsigned best_ssd;
  unsigned best_rate;
  int      best_qii;
  unsigned cur_cost;
  unsigned cur_ssd;
  unsigned cur_rate;
  int      lambda;
  int      nblocks;
  int      nqis;
  int      pli;
  int      bi;
  int      qii;
  lambda=_enc->lambda;
  nqis=_enc->state.nqis;
  ssd=_modec->ssd;
  rate=_modec->rate;
  /*Because (except in 4:4:4 mode) we aren't considering chroma blocks in coded
     order, we assume a constant overhead for coded block and qii flags.*/
  nblocks=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  nblocks=(nblocks-4>>1)+4;
  bi=4;
  for(pli=1;pli<3;pli++){
    for(;bi<nblocks;bi++){
      unsigned best_cost;
      satd=_frag_satd[bi];
      best_rate=oc_dct_cost2(&best_ssd,_enc->state.qis[0],pli,_qti,satd)
       +OC_CHROMA_QII_RATE;
      best_cost=OC_MODE_RD_COST(ssd+best_ssd,rate+best_rate,lambda);
      best_qii=0;
      for(qii=1;qii<nqis;qii++){
        cur_rate=oc_dct_cost2(&cur_ssd,_enc->state.qis[qii],0,_qti,satd)
         +OC_CHROMA_QII_RATE;
        cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_rate,lambda);
        if(cur_cost<best_cost){
          best_cost=cur_cost;
          best_ssd=cur_ssd;
          best_rate=cur_rate;
          best_qii=qii;
        }
      }
      if(_skip_ssd[bi]<UINT_MAX){
        cur_ssd=_skip_ssd[bi]<<OC_BIT_SCALE;
        cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate,lambda);
        if(cur_cost<=best_cost){
          best_ssd=cur_ssd;
          best_rate=0;
          best_qii+=4;
        }
      }
      rate+=best_rate;
      ssd+=best_ssd;
      _modec->qii[bi]=best_qii;
    }
    nblocks=(nblocks-4<<1)+4;
  }
  _modec->ssd=ssd;
  _modec->rate=rate;
}

static void oc_skip_cost(oc_enc_ctx *_enc,oc_enc_pipeline_state *_pipe,
 unsigned _mbi,unsigned _ssd[12]){
  OC_ALIGN16(ogg_int16_t  buffer[64]);
  const unsigned char    *src;
  const unsigned char    *ref;
  int                     ystride;
  const oc_fragment      *frags;
  const ptrdiff_t        *frag_buf_offs;
  const ptrdiff_t        *sb_map;
  const oc_mb_map_plane  *mb_map;
  const unsigned char    *map_idxs;
  int                     map_nidxs;
  ogg_int64_t             mask;
  unsigned                uncoded_ssd;
  int                     uncoded_dc;
  unsigned                dc_dequant;
  int                     dc_flag;
  int                     mapii;
  int                     mapi;
  int                     pli;
  int                     bi;
  ptrdiff_t               fragi;
  ptrdiff_t               frag_offs;
  int                     borderi;
  int                     pi;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[OC_FRAME_PREV]];
  ystride=_enc->state.ref_ystride[0];
  frags=_enc->state.frags;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  dc_dequant=_enc->state.dequant_tables[_enc->state.qis[0]][0][1][0];
  for(bi=0;bi<4;bi++){
    fragi=sb_map[bi];
    frag_offs=frag_buf_offs[fragi];
    oc_enc_frag_sub(_enc,buffer,src+frag_offs,ref+frag_offs,ystride);
    borderi=frags[fragi].borderi;
    uncoded_ssd=uncoded_dc=0;
    if(borderi<0){
      for(pi=0;pi<64;pi++){
        uncoded_ssd+=buffer[pi]*buffer[pi];
        uncoded_dc+=buffer[pi];
      }
    }
    else{
      ogg_int64_t mask;
      mask=_enc->state.borders[borderi].mask;
      for(pi=0;pi<64;pi++,mask>>=1)if(mask&1){
        uncoded_ssd+=buffer[pi]*buffer[pi];
        uncoded_dc+=buffer[pi];
      }
    }
    /*Scale to match DCT domain.*/
    uncoded_ssd<<=4;
    /*We actually only want the AC contribution to the SSD.*/
    uncoded_ssd-=uncoded_dc*uncoded_dc>>2;
    /*DC is a special case; if there's more than a full-quantizer improvement
       in the effective DC component, always force-code the block.*/
    dc_flag=abs(uncoded_dc)>dc_dequant<<1;
    uncoded_ssd|=-dc_flag;
    _pipe->skip_ssd[0][fragi-_pipe->froffset[0]]=_ssd[bi]=uncoded_ssd;
  }
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=(map_nidxs-4>>1)+4;
  mapii=4;
  for(pli=1;pli<3;pli++){
    ystride=_enc->state.ref_ystride[pli];
    dc_dequant=_enc->state.dequant_tables[_enc->state.qis[0]][pli][1][0];
    for(;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      frag_offs=frag_buf_offs[fragi];
      oc_enc_frag_sub(_enc,buffer,src+frag_offs,ref+frag_offs,ystride);
      borderi=frags[fragi].borderi;
      uncoded_ssd=uncoded_dc=0;
      if(borderi<0){
        for(pi=0;pi<64;pi++){
          uncoded_ssd+=buffer[pi]*buffer[pi];
          uncoded_dc+=buffer[pi];
        }
      }
      else{
        mask=_enc->state.borders[borderi].mask;
        for(pi=0;pi<64;pi++,mask>>=1)if(mask&1){
          uncoded_ssd+=buffer[pi]*buffer[pi];
          uncoded_dc+=buffer[pi];
        }
      }
      /*Scale to match DCT domain.*/
      uncoded_ssd<<=4;
      /*We actually only want the AC contribution to the SSD.*/
      uncoded_ssd-=uncoded_dc*uncoded_dc>>2;
      /*DC is a special case; if there's more than a full-quantizer improvement
         in the effective DC component, always force-code the block.*/
      dc_flag=abs(uncoded_dc)>dc_dequant<<1;
      uncoded_ssd|=-dc_flag;
      _pipe->skip_ssd[pli][fragi-_pipe->froffset[pli]]=_ssd[mapii]=uncoded_ssd;
    }
    map_nidxs=(map_nidxs-4<<1)+4;
  }
}

static void oc_mb_intra_satd(oc_enc_ctx *_enc,unsigned _mbi,
 unsigned _frag_satd[12]){
  const unsigned char   *src;
  const ptrdiff_t       *frag_buf_offs;
  const ptrdiff_t       *sb_map;
  const oc_mb_map_plane *mb_map;
  const unsigned char   *map_idxs;
  int                    map_nidxs;
  int                    mapii;
  int                    mapi;
  int                    ystride;
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[0];
  for(bi=0;bi<4;bi++){
    fragi=sb_map[bi];
    frag_offs=frag_buf_offs[fragi];
    _frag_satd[bi]=oc_enc_frag_intra_satd(_enc,src+frag_offs,ystride);
  }
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  /*Note: This assumes ref_ystride[1]==ref_ystride[2].*/
  ystride=_enc->state.ref_ystride[1];
  for(mapii=4;mapii<map_nidxs;mapii++){
    mapi=map_idxs[mapii];
    pli=mapi>>2;
    bi=mapi&3;
    fragi=mb_map[pli][bi];
    frag_offs=frag_buf_offs[fragi];
    _frag_satd[mapii]=oc_enc_frag_intra_satd(_enc,src+frag_offs,ystride);
  }
}

static void oc_cost_intra(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12]){
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,_frag_satd,_skip_ssd,0);
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,_frag_satd,_skip_ssd,0);
  _modec->overhead+=
   oc_mode_scheme_chooser_cost(&_enc->chooser,OC_MODE_INTRA)<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

static void oc_cost_inter(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,const signed char *_mv,
 const oc_fr_state *_fr,const oc_qii_state *_qs,const unsigned _skip_ssd[12]){
  unsigned               frag_satd[12];
  const unsigned char   *src;
  const unsigned char   *ref;
  int                    ystride;
  const ptrdiff_t       *frag_buf_offs;
  const ptrdiff_t       *sb_map;
  const oc_mb_map_plane *mb_map;
  const unsigned char   *map_idxs;
  int                    map_nidxs;
  int                    mapii;
  int                    mapi;
  int                    mv_offs[2];
  int                    dx;
  int                    dy;
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[
   _enc->state.ref_frame_idx[OC_FRAME_FOR_MODE(_mb_mode)]];
  ystride=_enc->state.ref_ystride[0];
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  dx=_mv[0];
  dy=_mv[1];
  _modec->rate=_modec->ssd=0;
  if(oc_state_get_mv_offsets(&_enc->state,mv_offs,0,dx,dy)>1){
    for(bi=0;bi<4;bi++){
      fragi=sb_map[bi];
      frag_offs=frag_buf_offs[fragi];
      frag_satd[bi]=oc_enc_frag_satd2_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
    }
  }
  else{
    for(bi=0;bi<4;bi++){
      fragi=sb_map[bi];
      frag_offs=frag_buf_offs[fragi];
      frag_satd[bi]=oc_enc_frag_satd_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride,UINT_MAX);
    }
  }
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  /*Note: This assumes ref_ystride[1]==ref_ystride[2].*/
  ystride=_enc->state.ref_ystride[1];
  if(oc_state_get_mv_offsets(&_enc->state,mv_offs,1,dx,dy)>1){
    for(mapii=4;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      pli=mapi>>2;
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      frag_offs=frag_buf_offs[fragi];
      frag_satd[mapii]=oc_enc_frag_satd2_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
    }
  }
  else{
    for(mapii=4;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      pli=mapi>>2;
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      frag_offs=frag_buf_offs[fragi];
      frag_satd[mapii]=oc_enc_frag_satd_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride,UINT_MAX);
    }
  }
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,frag_satd,_skip_ssd,1);
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,frag_satd,_skip_ssd,1);
  _modec->overhead+=
   oc_mode_scheme_chooser_cost(&_enc->chooser,_mb_mode)<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

static void oc_cost_inter_nomv(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _skip_ssd[12]){
  static const oc_mv OC_MV_ZERO;
  oc_cost_inter(_enc,_modec,_mbi,_mb_mode,OC_MV_ZERO,_fr,_qs,_skip_ssd);
}

static int oc_cost_inter1mv(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,const signed char *_mv,
 const oc_fr_state *_fr,const oc_qii_state *_qs,const unsigned _skip_ssd[12]){
  int bits0;
  oc_cost_inter(_enc,_modec,_mbi,_mb_mode,_mv,_fr,_qs,_skip_ssd);
  bits0=OC_MV_BITS[0][_mv[0]+31]+OC_MV_BITS[0][_mv[1]+31];
  _modec->overhead+=OC_MINI(_enc->mv_bits[0]+bits0,_enc->mv_bits[1]+12)
   -OC_MINI(_enc->mv_bits[0],_enc->mv_bits[1])<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
  return bits0;
}

/*A mapping from oc_mb_map (raster) ordering to oc_sb_map (Hilbert) ordering.*/
static const unsigned char OC_MB_PHASE[4][4]={
  {0,1,3,2},{0,3,1,2},{0,3,1,2},{2,3,1,0}
};

static void oc_cost_inter4mv(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,oc_mv _mv[4],const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _skip_ssd[12]){
  unsigned               frag_satd[12];
  oc_mv                  lbmvs[4];
  oc_mv                  cbmvs[4];
  const unsigned char   *src;
  const unsigned char   *ref;
  int                    ystride;
  const ptrdiff_t       *frag_buf_offs;
  oc_mv                 *frag_mvs;
  const oc_mb_map_plane *mb_map;
  const unsigned char   *map_idxs;
  int                    map_nidxs;
  int                    nqis;
  int                    mapii;
  int                    mapi;
  int                    mv_offs[2];
  int                    dx;
  int                    dy;
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  int                    bits0;
  int                    bits1;
  unsigned               satd;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[OC_FRAME_PREV]];
  ystride=_enc->state.ref_ystride[0];
  frag_buf_offs=_enc->state.frag_buf_offs;
  frag_mvs=_enc->state.frag_mvs;
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  _modec->rate=_modec->ssd=0;
  for(bi=0;bi<4;bi++){
    fragi=mb_map[0][bi];
    dx=_mv[bi][0];
    dy=_mv[bi][1];
    /*Save the block MVs as the current ones while we're here; we'll replace
       them if we don't ultimately choose 4MV mode.*/
    frag_mvs[fragi][0]=(signed char)dx;
    frag_mvs[fragi][1]=(signed char)dy;
    frag_offs=frag_buf_offs[fragi];
    if(oc_state_get_mv_offsets(&_enc->state,mv_offs,0,dx,dy)>1){
      satd=oc_enc_frag_satd2_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
    }
    else{
      satd=oc_enc_frag_satd_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride,UINT_MAX);
    }
    frag_satd[OC_MB_PHASE[_mbi&3][bi]]=satd;
  }
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,frag_satd,
   _enc->vp3_compatible?OC_NOSKIP:_skip_ssd,1);
  /*Figure out which blocks are being skipped and give them (0,0) MVs.*/
  bits0=0;
  bits1=0;
  nqis=_enc->state.nqis;
  for(bi=0;bi<4;bi++){
    if(_modec->qii[OC_MB_PHASE[_mbi&3][bi]]>=nqis){
      memset(lbmvs+bi,0,sizeof(*lbmvs));
    }
    else{
      memcpy(lbmvs+bi,_mv+bi,sizeof(*lbmvs));
      bits0+=OC_MV_BITS[0][_mv[bi][0]+31]+OC_MV_BITS[0][_mv[bi][1]+31];
      bits1+=12;
    }
  }
  (*OC_SET_CHROMA_MVS_TABLE[_enc->state.info.pixel_fmt])(cbmvs,
   (const oc_mv *)lbmvs);
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  /*Note: This assumes ref_ystride[1]==ref_ystride[2].*/
  ystride=_enc->state.ref_ystride[1];
  for(mapii=4;mapii<map_nidxs;mapii++){
    mapi=map_idxs[mapii];
    pli=mapi>>2;
    bi=mapi&3;
    fragi=mb_map[pli][bi];
    dx=cbmvs[bi][0];
    dy=cbmvs[bi][1];
    frag_offs=frag_buf_offs[fragi];
    /*TODO: We could save half these calls by re-using the results for the Cb
       and Cr planes; is it worth it?*/
    if(oc_state_get_mv_offsets(&_enc->state,mv_offs,pli,dx,dy)>1){
      satd=oc_enc_frag_satd2_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
    }
    else{
      satd=oc_enc_frag_satd_thresh(_enc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride,UINT_MAX);
    }
    frag_satd[mapii]=satd;
  }
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,frag_satd,_skip_ssd,1);
  _modec->overhead+=
   oc_mode_scheme_chooser_cost(&_enc->chooser,OC_MODE_INTER_MV_FOUR)
   +OC_MINI(_enc->mv_bits[0]+bits0,_enc->mv_bits[1]+bits1)
   -OC_MINI(_enc->mv_bits[0],_enc->mv_bits[1])<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

int oc_enc_analyze_inter(oc_enc_ctx *_enc,int _allow_keyframe,int _recode){
  oc_set_chroma_mvs_func  set_chroma_mvs;
  oc_enc_pipeline_state   pipe;
  oc_qii_state            intra_luma_qs;
  oc_mv                   last_mv;
  oc_mv                   prior_mv;
  ogg_int64_t             interbits;
  ogg_int64_t             intrabits;
  const unsigned char    *map_idxs;
  int                     nmap_idxs;
  unsigned               *coded_mbis;
  unsigned               *uncoded_mbis;
  size_t                  ncoded_mbis;
  size_t                  nuncoded_mbis;
  oc_sb_flags            *sb_flags;
  signed char            *mb_modes;
  const oc_sb_map        *sb_maps;
  const oc_mb_map        *mb_maps;
  oc_mb_enc_info         *embs;
  oc_fragment            *frags;
  oc_mv                  *frag_mvs;
  int                     qi;
  unsigned                stripe_sby;
  unsigned                mcu_nvsbs;
  int                     notstart;
  int                     notdone;
  int                     vdec;
  unsigned                sbi;
  unsigned                sbi_end;
  int                     refi;
  int                     pli;
  set_chroma_mvs=OC_SET_CHROMA_MVS_TABLE[_enc->state.info.pixel_fmt];
  _enc->state.frame_type=OC_INTER_FRAME;
  oc_mode_scheme_chooser_reset(&_enc->chooser);
  oc_enc_tokenize_start(_enc);
  oc_enc_pipeline_init(_enc,&pipe);
  if(_allow_keyframe)oc_qii_state_init(&intra_luma_qs);
  _enc->mv_bits[0]=_enc->mv_bits[1]=0;
  interbits=intrabits=0;
  last_mv[0]=last_mv[1]=prior_mv[0]=prior_mv[1]=0;
  /*Choose MVs and MB modes and quantize and code luma.
    Must be done in Hilbert order.*/
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  nmap_idxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  qi=_enc->state.qis[0];
  coded_mbis=_enc->coded_mbis;
  uncoded_mbis=coded_mbis+_enc->state.nmbs;
  ncoded_mbis=0;
  nuncoded_mbis=0;
  _enc->state.ncoded_fragis[0]=0;
  _enc->state.ncoded_fragis[1]=0;
  _enc->state.ncoded_fragis[2]=0;
  sb_flags=_enc->state.sb_flags;
  mb_modes=_enc->state.mb_modes;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  mb_maps=(const oc_mb_map *)_enc->state.mb_maps;
  embs=_enc->mb_info;
  frags=_enc->state.frags;
  frag_mvs=_enc->state.frag_mvs;
  vdec=!(_enc->state.info.pixel_fmt&2);
  notstart=0;
  notdone=1;
  mcu_nvsbs=_enc->mcu_nvsbs;
  for(stripe_sby=0;notdone;stripe_sby+=mcu_nvsbs){
    notdone=oc_enc_pipeline_set_stripe(_enc,&pipe,stripe_sby);
    sbi_end=pipe.sbi_end[0];
    for(sbi=pipe.sbi0[0];sbi<sbi_end;sbi++){
      int quadi;
      /*Mode addressing is through Y plane, always 4 MB per SB.*/
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        oc_mode_choice modes[8];
        unsigned       skip_ssd[12];
        unsigned       intra_satd[12];
        int            mb_mv_bits_0;
        int            mb_gmv_bits_0;
        int            inter_mv_pref;
        int            mb_mode;
        int            dx;
        int            dy;
        unsigned       mbi;
        int            mapii;
        int            mapi;
        int            bi;
        ptrdiff_t      fragi;
        mbi=sbi<<2|quadi;
        /*Motion estimation:
          We always do a basic 1MV search for all macroblocks, coded or not,
           keyframe or not.*/
        if(!_recode&&_enc->sp_level<OC_SP_LEVEL_NOMC)oc_mcenc_search(_enc,mbi);
        dx=dy=0;
        /*Find the block choice with the lowest estimated coding cost.
          If a Cb or Cr block is coded but no Y' block from a macro block then
           the mode MUST be OC_MODE_INTER_NOMV.
          This is the default state to which the mode data structure is
           initialised in encoder and decoder at the start of each frame.*/
        /*Block coding cost is estimated from correlated SATD metrics.*/
        /*At this point, all blocks that are in frame are still marked coded.*/
        if(!_recode){
          memcpy(embs[mbi].unref_mv,
           embs[mbi].analysis_mv[0],sizeof(embs[mbi].unref_mv));
          embs[mbi].refined=0;
        }
        oc_mb_intra_satd(_enc,mbi,intra_satd);
        /*Estimate the cost of coding this MB in a keyframe.*/
        if(_allow_keyframe){
          oc_cost_intra(_enc,modes+OC_MODE_INTRA,mbi,
           pipe.fr+0,&intra_luma_qs,intra_satd,OC_NOSKIP);
          intrabits+=modes[OC_MODE_INTRA].rate;
          for(bi=0;bi<4;bi++){
            oc_qii_state_advance(&intra_luma_qs,&intra_luma_qs,
             modes[OC_MODE_INTRA].qii[bi]);
          }
        }
        /*Estimate the cost in a delta frame for various modes.*/
        oc_skip_cost(_enc,&pipe,mbi,skip_ssd);
        oc_cost_inter_nomv(_enc,modes+OC_MODE_INTER_NOMV,mbi,
         OC_MODE_INTER_NOMV,pipe.fr+0,pipe.qs+0,skip_ssd);
        if(_enc->sp_level<OC_SP_LEVEL_NOMC){
          oc_cost_intra(_enc,modes+OC_MODE_INTRA,mbi,
           pipe.fr+0,pipe.qs+0,intra_satd,skip_ssd);
          mb_mv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_INTER_MV,mbi,
           OC_MODE_INTER_MV,embs[mbi].unref_mv[OC_FRAME_PREV],
           pipe.fr+0,pipe.qs+0,skip_ssd);
          oc_cost_inter(_enc,modes+OC_MODE_INTER_MV_LAST,mbi,
           OC_MODE_INTER_MV_LAST,last_mv,pipe.fr+0,pipe.qs+0,skip_ssd);
          oc_cost_inter(_enc,modes+OC_MODE_INTER_MV_LAST2,mbi,
           OC_MODE_INTER_MV_LAST2,prior_mv,pipe.fr+0,pipe.qs+0,skip_ssd);
          oc_cost_inter4mv(_enc,modes+OC_MODE_INTER_MV_FOUR,mbi,
           embs[mbi].block_mv,pipe.fr+0,pipe.qs+0,skip_ssd);
          oc_cost_inter_nomv(_enc,modes+OC_MODE_GOLDEN_NOMV,mbi,
           OC_MODE_GOLDEN_NOMV,pipe.fr+0,pipe.qs+0,skip_ssd);
          mb_gmv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_GOLDEN_MV,mbi,
           OC_MODE_GOLDEN_MV,embs[mbi].unref_mv[OC_FRAME_GOLD],
           pipe.fr+0,pipe.qs+0,skip_ssd);
          /*The explicit MV modes (2,6,7) have not yet gone through halfpel
             refinement.
            We choose the explicit MV mode that's already furthest ahead on
             R-D cost and refine only that one.
            We have to be careful to remember which ones we've refined so that
             we don't refine it again if we re-encode this frame.*/
          inter_mv_pref=_enc->lambda*3;
          if(modes[OC_MODE_INTER_MV_FOUR].cost<modes[OC_MODE_INTER_MV].cost&&
           modes[OC_MODE_INTER_MV_FOUR].cost<modes[OC_MODE_GOLDEN_MV].cost){
            if(!(embs[mbi].refined&0x80)){
              oc_mcenc_refine4mv(_enc,mbi);
              embs[mbi].refined|=0x80;
            }
            oc_cost_inter4mv(_enc,modes+OC_MODE_INTER_MV_FOUR,mbi,
             embs[mbi].ref_mv,pipe.fr+0,pipe.qs+0,skip_ssd);
          }
          else if(modes[OC_MODE_GOLDEN_MV].cost+inter_mv_pref<
           modes[OC_MODE_INTER_MV].cost){
            if(!(embs[mbi].refined&0x40)){
              oc_mcenc_refine1mv(_enc,mbi,OC_FRAME_GOLD);
              embs[mbi].refined|=0x40;
            }
            mb_gmv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_GOLDEN_MV,mbi,
             OC_MODE_GOLDEN_MV,embs[mbi].analysis_mv[0][OC_FRAME_GOLD],
             pipe.fr+0,pipe.qs+0,skip_ssd);
          }
          if(!(embs[mbi].refined&0x04)){
            oc_mcenc_refine1mv(_enc,mbi,OC_FRAME_PREV);
            embs[mbi].refined|=0x04;
          }
          mb_mv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_INTER_MV,mbi,
           OC_MODE_INTER_MV,embs[mbi].analysis_mv[0][OC_FRAME_PREV],
           pipe.fr+0,pipe.qs+0,skip_ssd);
          /*Finally, pick the mode with the cheapest estimated R-D cost.*/
          mb_mode=OC_MODE_INTER_NOMV;
          if(modes[OC_MODE_INTRA].cost<modes[OC_MODE_INTER_NOMV].cost){
            mb_mode=OC_MODE_INTRA;
          }
          if(modes[OC_MODE_INTER_MV_LAST].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_INTER_MV_LAST;
          }
          if(modes[OC_MODE_INTER_MV_LAST2].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_INTER_MV_LAST2;
          }
          if(modes[OC_MODE_GOLDEN_NOMV].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_GOLDEN_NOMV;
          }
          if(modes[OC_MODE_GOLDEN_MV].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_GOLDEN_MV;
          }
          if(modes[OC_MODE_INTER_MV_FOUR].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_INTER_MV_FOUR;
          }
          /*We prefer OC_MODE_INTER_MV, but not over LAST and LAST2.*/
          if(mb_mode==OC_MODE_INTER_MV_LAST||mb_mode==OC_MODE_INTER_MV_LAST2){
            inter_mv_pref=0;
          }
          if(modes[OC_MODE_INTER_MV].cost<modes[mb_mode].cost+inter_mv_pref){
            mb_mode=OC_MODE_INTER_MV;
          }
        }
        else{
          oc_cost_inter_nomv(_enc,modes+OC_MODE_GOLDEN_NOMV,mbi,
           OC_MODE_GOLDEN_NOMV,pipe.fr+0,pipe.qs+0,skip_ssd);
          mb_mode=OC_MODE_INTER_NOMV;
          if(modes[OC_MODE_INTRA].cost<modes[OC_MODE_INTER_NOMV].cost){
            mb_mode=OC_MODE_INTRA;
          }
          if(modes[OC_MODE_GOLDEN_NOMV].cost<modes[mb_mode].cost){
            mb_mode=OC_MODE_GOLDEN_NOMV;
          }
          mb_mv_bits_0=mb_gmv_bits_0=0;
        }
        mb_modes[mbi]=mb_mode;
        /*Propagate the MVs to the luma blocks.*/
        if(mb_mode!=OC_MODE_INTER_MV_FOUR){
          switch(mb_mode){
            case OC_MODE_INTER_MV:{
              dx=embs[mbi].analysis_mv[0][OC_FRAME_PREV][0];
              dy=embs[mbi].analysis_mv[0][OC_FRAME_PREV][1];
            }break;
            case OC_MODE_INTER_MV_LAST:{
              dx=last_mv[0];
              dy=last_mv[1];
            }break;
            case OC_MODE_INTER_MV_LAST2:{
              dx=prior_mv[0];
              dy=prior_mv[1];
            }break;
            case OC_MODE_GOLDEN_MV:{
              dx=embs[mbi].analysis_mv[0][OC_FRAME_GOLD][0];
              dy=embs[mbi].analysis_mv[0][OC_FRAME_GOLD][1];
            }break;
          }
          for(bi=0;bi<4;bi++){
            fragi=mb_maps[mbi][0][bi];
            frag_mvs[fragi][0]=(signed char)dx;
            frag_mvs[fragi][1]=(signed char)dy;
          }
        }
        for(bi=0;bi<4;bi++){
          fragi=sb_maps[mbi>>2][mbi&3][bi];
          frags[fragi].qii=modes[mb_mode].qii[bi];
        }
        if(oc_enc_mb_transform_quantize_luma(_enc,&pipe,mbi,
         modes[mb_mode].overhead>>OC_BIT_SCALE)>0){
          int orig_mb_mode;
          orig_mb_mode=mb_mode;
          mb_mode=mb_modes[mbi];
          switch(mb_mode){
            case OC_MODE_INTER_MV:{
              memcpy(prior_mv,last_mv,sizeof(prior_mv));
              /*If we're backing out from 4MV, find the MV we're actually
                 using.*/
              if(orig_mb_mode==OC_MODE_INTER_MV_FOUR){
                for(bi=0;;bi++){
                  fragi=mb_maps[mbi][0][bi];
                  if(frags[fragi].coded){
                    memcpy(last_mv,frag_mvs[fragi],sizeof(last_mv));
                    dx=frag_mvs[fragi][0];
                    dy=frag_mvs[fragi][1];
                    break;
                  }
                }
                mb_mv_bits_0=OC_MV_BITS[0][dx+31]+OC_MV_BITS[0][dy+31];
              }
              /*Otherwise we used the original analysis MV.*/
              else{
                memcpy(last_mv,
                 embs[mbi].analysis_mv[0][OC_FRAME_PREV],sizeof(last_mv));
              }
              _enc->mv_bits[0]+=mb_mv_bits_0;
              _enc->mv_bits[1]+=12;
            }break;
            case OC_MODE_INTER_MV_LAST2:{
              oc_mv tmp_mv;
              memcpy(tmp_mv,prior_mv,sizeof(tmp_mv));
              memcpy(prior_mv,last_mv,sizeof(prior_mv));
              memcpy(last_mv,tmp_mv,sizeof(last_mv));
            }break;
            case OC_MODE_GOLDEN_MV:{
              _enc->mv_bits[0]+=mb_gmv_bits_0;
              _enc->mv_bits[1]+=12;
            }break;
            case OC_MODE_INTER_MV_FOUR:{
              oc_mv lbmvs[4];
              oc_mv cbmvs[4];
              memcpy(prior_mv,last_mv,sizeof(prior_mv));
              for(bi=0;bi<4;bi++){
                fragi=mb_maps[mbi][0][bi];
                if(frags[fragi].coded){
                  memcpy(last_mv,frag_mvs[fragi],sizeof(last_mv));
                  memcpy(lbmvs[bi],frag_mvs[fragi],sizeof(lbmvs[bi]));
                  _enc->mv_bits[0]+=OC_MV_BITS[0][frag_mvs[fragi][0]+31]
                   +OC_MV_BITS[0][frag_mvs[fragi][1]+31];
                  _enc->mv_bits[1]+=12;
                }
                /*Replace the block MVs for not-coded blocks with (0,0).*/
                else memset(lbmvs[bi],0,sizeof(lbmvs[bi]));
              }
              (*set_chroma_mvs)(cbmvs,(const oc_mv *)lbmvs);
              for(mapii=4;mapii<nmap_idxs;mapii++){
                mapi=map_idxs[mapii];
                pli=mapi>>2;
                bi=mapi&3;
                fragi=mb_maps[mbi][pli][bi];
                frags[fragi].mb_mode=mb_mode;
                frags[fragi].qii=modes[OC_MODE_INTER_MV_FOUR].qii[mapii];
                memcpy(frag_mvs[fragi],cbmvs[bi],sizeof(frag_mvs[fragi]));
              }
            }break;
          }
          coded_mbis[ncoded_mbis++]=mbi;
          oc_mode_scheme_chooser_update(&_enc->chooser,mb_mode);
          interbits+=modes[mb_mode].rate+modes[mb_mode].overhead;
        }
        else{
          *(uncoded_mbis-++nuncoded_mbis)=mbi;
          mb_mode=OC_MODE_INTER_NOMV;
          dx=dy=0;
        }
        /*Propagate final MB mode and MVs to the chroma blocks.
          This has already been done for 4MV mode, since it requires individual
           block motion vectors.*/
        if(mb_mode!=OC_MODE_INTER_MV_FOUR){
          for(mapii=4;mapii<nmap_idxs;mapii++){
            mapi=map_idxs[mapii];
            pli=mapi>>2;
            bi=mapi&3;
            fragi=mb_maps[mbi][pli][bi];
            frags[fragi].mb_mode=mb_mode;
            /*If we switched from 4MV mode to INTER_MV mode, then the qii
               values won't have been chosen with the right MV, but it's
               probaby not worth re-estimating them.*/
            frags[fragi].qii=modes[mb_mode].qii[mapii];
            frag_mvs[fragi][0]=(signed char)dx;
            frag_mvs[fragi][1]=(signed char)dy;
          }
        }
      }
      oc_fr_state_flush_sb(pipe.fr+0);
      sb_flags[sbi].coded_fully=pipe.fr[0].sb_full;
      sb_flags[sbi].coded_partially=pipe.fr[0].sb_partial;
    }
    oc_enc_pipeline_finish_mcu_plane(_enc,&pipe,0,notstart,notdone);
    /*Code chroma planes.*/
    for(pli=1;pli<3;pli++){
      oc_enc_sb_transform_quantize_chroma(_enc,&pipe,
       pli,pipe.sbi0[pli],pipe.sbi_end[pli]);
      oc_enc_pipeline_finish_mcu_plane(_enc,&pipe,pli,notstart,notdone);
    }
    notstart=1;
  }
  /*Finish filling in the reference frame borders.*/
  refi=_enc->state.ref_frame_idx[OC_FRAME_SELF];
  for(pli=0;pli<3;pli++)oc_state_borders_fill_caps(&_enc->state,refi,pli);
  /*Finish adding flagging overhead costs to inter bit counts to determine if
     we should have coded a key frame instead.*/
  if(_allow_keyframe){
    if(interbits>intrabits)return 1;
    /*Technically the chroma plane counts are over-estimations, because they
       don't account for continuing runs from the luma planes, but the
       inaccuracy is small.*/
    for(pli=0;pli<3;pli++)interbits+=pipe.fr[pli].bits<<OC_BIT_SCALE;
    interbits+=OC_MINI(_enc->mv_bits[0],_enc->mv_bits[1])<<OC_BIT_SCALE;
    interbits+=
     _enc->chooser.scheme_bits[_enc->chooser.scheme_list[0]]<<OC_BIT_SCALE;
    if(interbits>intrabits)return 1;
  }
  _enc->ncoded_mbis=ncoded_mbis;
  /*Compact the coded fragment list.*/
  {
    ptrdiff_t ncoded_fragis;
    ncoded_fragis=_enc->state.ncoded_fragis[0];
    for(pli=1;pli<3;pli++){
      memmove(_enc->state.coded_fragis+ncoded_fragis,
       _enc->state.coded_fragis+_enc->state.fplanes[pli].froffset,
       _enc->state.ncoded_fragis[pli]*sizeof(*_enc->state.coded_fragis));
      ncoded_fragis+=_enc->state.ncoded_fragis[pli];
    }
    _enc->state.ntotal_coded_fragis=ncoded_fragis;
  }
  return 0;
}

#if defined(OC_COLLECT_METRICS)
# include <stdio.h>
# include <math.h>

/*TODO: It may be helpful (for block-level quantizers especially) to separate
   out the contributions from AC and DC into separate tables.*/

# define OC_ZWEIGHT   (0.25)

static void oc_mode_metrics_add(oc_mode_metrics *_metrics,
 double _w,int _satd,int _rate,double _rmse){
  double rate;
  /*Accumulate statistics without the scaling; this lets us change the scale
     factor yet still use old data.*/
  rate=ldexp(_rate,-OC_BIT_SCALE);
  if(_metrics->fragw>0){
    double dsatd;
    double drate;
    double drmse;
    double w;
    dsatd=_satd-_metrics->satd/_metrics->fragw;
    drate=rate-_metrics->rate/_metrics->fragw;
    drmse=_rmse-_metrics->rmse/_metrics->fragw;
    w=_metrics->fragw*_w/(_metrics->fragw+_w);
    _metrics->satd2+=dsatd*dsatd*w;
    _metrics->satdrate+=dsatd*drate*w;
    _metrics->rate2+=drate*drate*w;
    _metrics->satdrmse+=dsatd*drmse*w;
    _metrics->rmse2+=drmse*drmse*w;
  }
  _metrics->fragw+=_w;
  _metrics->satd+=_satd*_w;
  _metrics->rate+=rate*_w;
  _metrics->rmse+=_rmse*_w;
}

static void oc_mode_metrics_merge(oc_mode_metrics *_dst,
 const oc_mode_metrics *_src,int _n){
  int i;
  /*Find a non-empty set of metrics.*/
  for(i=0;i<_n&&_src[i].fragw<=0;i++);
  if(i>=_n){
    memset(_dst,0,sizeof(*_dst));
    return;
  }
  memcpy(_dst,_src+i,sizeof(*_dst));
  /*And iterate over the remaining non-empty sets of metrics.*/
  for(i++;i<_n;i++)if(_src[i].fragw>0){
    double wa;
    double wb;
    double dsatd;
    double drate;
    double drmse;
    double w;
    wa=_dst->fragw;
    wb=_src[i].fragw;
    dsatd=_src[i].satd/wb-_dst->satd/wa;
    drate=_src[i].rate/wb-_dst->rate/wa;
    drmse=_src[i].rmse/wb-_dst->rmse/wa;
    w=wa*wb/(wa+wb);
    _dst->fragw+=_src[i].fragw;
    _dst->satd+=_src[i].satd;
    _dst->rate+=_src[i].rate;
    _dst->rmse+=_src[i].rmse;
    _dst->satd2+=_src[i].satd2+dsatd*dsatd*w;
    _dst->satdrate+=_src[i].satdrate+dsatd*drate*w;
    _dst->rate2+=_src[i].rate2+drate*drate*w;
    _dst->satdrmse+=_src[i].satdrmse+dsatd*drmse*w;
    _dst->rmse2+=_src[i].rmse2+drmse*drmse*w;
  }
}

/*Compile collected SATD/rate/RMSE metrics into a form that's immediately
   useful for mode decision.*/
static void oc_enc_mode_metrics_update(oc_enc_ctx *_enc,int _qi){
  int pli;
  int qti;
  oc_restore_fpu(&_enc->state);
  /*Convert raw collected data into cleaned up sample points.*/
  for(pli=0;pli<3;pli++){
    for(qti=0;qti<2;qti++){
      double fragw;
      int    bin0;
      int    bin1;
      int    bin;
      fragw=0;
      bin0=bin1=0;
      for(bin=0;bin<OC_SAD_BINS;bin++){
        oc_mode_metrics metrics;
        OC_MODE_RD[_qi][pli][qti][bin].rate=0;
        OC_MODE_RD[_qi][pli][qti][bin].rmse=0;
        /*Find some points on either side of the current bin.*/
        while((bin1<bin+1||fragw<OC_ZWEIGHT)&&bin1<OC_SAD_BINS-1){
          fragw+=OC_MODE_METRICS[_qi][pli][qti][bin1++].fragw;
        }
        while(bin0+1<bin&&bin0+1<bin1&&
         fragw-OC_MODE_METRICS[_qi][pli][qti][bin0].fragw>=OC_ZWEIGHT){
          fragw-=OC_MODE_METRICS[_qi][pli][qti][bin0++].fragw;
        }
        /*Merge statistics and fit lines.*/
        oc_mode_metrics_merge(&metrics,
         OC_MODE_METRICS[_qi][pli][qti]+bin0,bin1-bin0);
        if(metrics.fragw>0&&metrics.satd2>0){
          double a;
          double b;
          double msatd;
          double mrate;
          double mrmse;
          double rate;
          double rmse;
          msatd=metrics.satd/metrics.fragw;
          mrate=metrics.rate/metrics.fragw;
          mrmse=metrics.rmse/metrics.fragw;
          /*Compute the points on these lines corresponding to the actual bin
             value.*/
          b=metrics.satdrate/metrics.satd2;
          a=mrate-b*msatd;
          rate=ldexp(a+b*(bin<<OC_SAD_SHIFT),OC_BIT_SCALE);
          OC_MODE_RD[_qi][pli][qti][bin].rate=
           (ogg_int16_t)OC_CLAMPI(-32768,(int)(rate+0.5),32767);
          b=metrics.satdrmse/metrics.satd2;
          a=mrmse-b*msatd;
          rmse=ldexp(a+b*(bin<<OC_SAD_SHIFT),OC_RMSE_SCALE);
          OC_MODE_RD[_qi][pli][qti][bin].rmse=
           (ogg_int16_t)OC_CLAMPI(-32768,(int)(rmse+0.5),32767);
        }
      }
    }
  }
}



/*The following token skipping code used to also be used in the decoder (and
   even at one point other places in the encoder).
  However, it was obsoleted by other optimizations, and is now only used here.
  It has been moved here to avoid generating the code when it's not needed.*/

/*Determines the number of blocks or coefficients to be skipped for a given
   token value.
  _token:      The token value to skip.
  _extra_bits: The extra bits attached to this token.
  Return: A positive value indicates that number of coefficients are to be
           skipped in the current block.
          Otherwise, the negative of the return value indicates that number of
           blocks are to be ended.*/
typedef ptrdiff_t (*oc_token_skip_func)(int _token,int _extra_bits);

/*Handles the simple end of block tokens.*/
static ptrdiff_t oc_token_skip_eob(int _token,int _extra_bits){
  int nblocks_adjust;
  nblocks_adjust=OC_UNIBBLE_TABLE32(0,1,2,3,7,15,0,0,_token)+1;
  return -_extra_bits-nblocks_adjust;
}

/*The last EOB token has a special case, where an EOB run of size zero ends all
   the remaining blocks in the frame.*/
static ptrdiff_t oc_token_skip_eob6(int _token,int _extra_bits){
  /*Note: We want to return -PTRDIFF_MAX, but that requires C99, which is not
     yet available everywhere; this should be equivalent.*/
  if(!_extra_bits)return -(~(size_t)0>>1);
  return -_extra_bits;
}

/*Handles the pure zero run tokens.*/
static ptrdiff_t oc_token_skip_zrl(int _token,int _extra_bits){
  return _extra_bits+1;
}

/*Handles a normal coefficient value token.*/
static ptrdiff_t oc_token_skip_val(void){
  return 1;
}

/*Handles a category 1A zero run/coefficient value combo token.*/
static ptrdiff_t oc_token_skip_run_cat1a(int _token){
  return _token-OC_DCT_RUN_CAT1A+2;
}

/*Handles category 1b, 1c, 2a, and 2b zero run/coefficient value combo tokens.*/
static ptrdiff_t oc_token_skip_run(int _token,int _extra_bits){
  int run_cati;
  int ncoeffs_mask;
  int ncoeffs_adjust;
  run_cati=_token-OC_DCT_RUN_CAT1B;
  ncoeffs_mask=OC_BYTE_TABLE32(3,7,0,1,run_cati);
  ncoeffs_adjust=OC_BYTE_TABLE32(7,11,2,3,run_cati);
  return (_extra_bits&ncoeffs_mask)+ncoeffs_adjust;
}

/*A jump table for computing the number of coefficients or blocks to skip for
   a given token value.
  This reduces all the conditional branches, etc., needed to parse these token
   values down to one indirect jump.*/
static const oc_token_skip_func OC_TOKEN_SKIP_TABLE[TH_NDCT_TOKENS]={
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob,
  oc_token_skip_eob6,
  oc_token_skip_zrl,
  oc_token_skip_zrl,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_val,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  (oc_token_skip_func)oc_token_skip_run_cat1a,
  oc_token_skip_run,
  oc_token_skip_run,
  oc_token_skip_run,
  oc_token_skip_run
};

/*Determines the number of blocks or coefficients to be skipped for a given
   token value.
  _token:      The token value to skip.
  _extra_bits: The extra bits attached to this token.
  Return: A positive value indicates that number of coefficients are to be
           skipped in the current block.
          Otherwise, the negative of the return value indicates that number of
           blocks are to be ended.
          0 will never be returned, so that at least one coefficient in one
           block will always be decoded for every token.*/
static ptrdiff_t oc_dct_token_skip(int _token,int _extra_bits){
  return (*OC_TOKEN_SKIP_TABLE[_token])(_token,_extra_bits);
}



void oc_enc_mode_metrics_collect(oc_enc_ctx *_enc){
  static const unsigned char OC_ZZI_HUFF_OFFSET[64]={
     0,16,16,16,16,16,32,32,
    32,32,32,32,32,32,32,48,
    48,48,48,48,48,48,48,48,
    48,48,48,48,64,64,64,64,
    64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64,
    64,64,64,64,64,64,64,64
  };
  const oc_fragment *frags;
  const unsigned    *frag_satd;
  const unsigned    *frag_ssd;
  const ptrdiff_t   *coded_fragis;
  ptrdiff_t          ncoded_fragis;
  ptrdiff_t          fragii;
  double             fragw;
  int                qti;
  int                qii;
  int                qi;
  int                pli;
  int                zzi;
  int                token;
  int                eb;
  oc_restore_fpu(&_enc->state);
  /*Load any existing mode metrics if we haven't already.*/
  if(!oc_has_mode_metrics){
    FILE *fmetrics;
    memset(OC_MODE_METRICS,0,sizeof(OC_MODE_METRICS));
    fmetrics=fopen("modedec.stats","rb");
    if(fmetrics!=NULL){
      fread(OC_MODE_METRICS,sizeof(OC_MODE_METRICS),1,fmetrics);
      fclose(fmetrics);
    }
    for(qi=0;qi<64;qi++)oc_enc_mode_metrics_update(_enc,qi);
    oc_has_mode_metrics=1;
  }
  qti=_enc->state.frame_type;
  frags=_enc->state.frags;
  frag_satd=_enc->frag_satd;
  frag_ssd=_enc->frag_ssd;
  coded_fragis=_enc->state.coded_fragis;
  ncoded_fragis=fragii=0;
  /*Weight the fragments by the inverse frame size; this prevents HD content
     from dominating the statistics.*/
  fragw=1.0/_enc->state.nfrags;
  for(pli=0;pli<3;pli++){
    ptrdiff_t ti[64];
    int       eob_token[64];
    int       eob_run[64];
    /*Set up token indices and eob run counts.
      We don't bother trying to figure out the real cost of the runs that span
       coefficients; instead we use the costs that were available when R-D
       token optimization was done.*/
    for(zzi=0;zzi<64;zzi++){
      ti[zzi]=_enc->dct_token_offs[pli][zzi];
      if(ti[zzi]>0){
        token=_enc->dct_tokens[pli][zzi][0];
        eb=_enc->extra_bits[pli][zzi][0];
        eob_token[zzi]=token;
        eob_run[zzi]=-oc_dct_token_skip(token,eb);
      }
      else{
        eob_token[zzi]=OC_NDCT_EOB_TOKEN_MAX;
        eob_run[zzi]=0;
      }
    }
    /*Scan the list of coded fragments for this plane.*/
    ncoded_fragis+=_enc->state.ncoded_fragis[pli];
    for(;fragii<ncoded_fragis;fragii++){
      ptrdiff_t    fragi;
      ogg_uint32_t frag_bits;
      int          huffi;
      int          skip;
      int          mb_mode;
      unsigned     satd;
      int          bin;
      fragi=coded_fragis[fragii];
      frag_bits=0;
      for(zzi=0;zzi<64;){
        if(eob_run[zzi]>0){
          /*We've reached the end of the block.*/
          eob_run[zzi]--;
          break;
        }
        huffi=_enc->huff_idxs[qti][zzi>0][pli+1>>1]
         +OC_ZZI_HUFF_OFFSET[zzi];
        if(eob_token[zzi]<OC_NDCT_EOB_TOKEN_MAX){
          /*This token caused an EOB run to be flushed.
            Therefore it gets the bits associated with it.*/
          frag_bits+=_enc->huff_codes[huffi][eob_token[zzi]].nbits
           +OC_DCT_TOKEN_EXTRA_BITS[eob_token[zzi]];
          eob_token[zzi]=OC_NDCT_EOB_TOKEN_MAX;
        }
        token=_enc->dct_tokens[pli][zzi][ti[zzi]];
        eb=_enc->extra_bits[pli][zzi][ti[zzi]];
        ti[zzi]++;
        skip=oc_dct_token_skip(token,eb);
        if(skip<0){
          eob_token[zzi]=token;
          eob_run[zzi]=-skip;
        }
        else{
          /*A regular DCT value token; accumulate the bits for it.*/
          frag_bits+=_enc->huff_codes[huffi][token].nbits
           +OC_DCT_TOKEN_EXTRA_BITS[token];
          zzi+=skip;
        }
      }
      mb_mode=frags[fragi].mb_mode;
      qi=_enc->state.qis[frags[fragi].qii];
      satd=frag_satd[fragi]<<(pli+1&2);
      bin=OC_MINI(satd>>OC_SAD_SHIFT,OC_SAD_BINS-1);
      oc_mode_metrics_add(OC_MODE_METRICS[qi][pli][mb_mode!=OC_MODE_INTRA]+bin,
       fragw,satd,frag_bits<<OC_BIT_SCALE,sqrt(frag_ssd[fragi]));
    }
  }
  /*Update global SATD/rate/RMSE estimation matrix.*/
  for(qii=0;qii<_enc->state.nqis;qii++){
    oc_enc_mode_metrics_update(_enc,_enc->state.qis[qii]);
  }
}

void oc_enc_mode_metrics_dump(oc_enc_ctx *_enc){
  FILE *fmetrics;
  int   qi;
  /*Generate sample points for complete list of QI values.*/
  for(qi=0;qi<64;qi++)oc_enc_mode_metrics_update(_enc,qi);
  fmetrics=fopen("modedec.stats","wb");
  if(fmetrics!=NULL){
    fwrite(OC_MODE_METRICS,sizeof(OC_MODE_METRICS),1,fmetrics);
    fclose(fmetrics);
  }
  fprintf(stdout,
   "/*File generated by libtheora with OC_COLLECT_METRICS"
   " defined at compile time.*/\n"
   "#if !defined(_modedec_H)\n"
   "# define _modedec_H (1)\n"
   "\n"
   "\n"
   "\n"
   "# if defined(OC_COLLECT_METRICS)\n"
   "typedef struct oc_mode_metrics oc_mode_metrics;\n"
   "# endif\n"
   "typedef struct oc_mode_rd      oc_mode_rd;\n"
   "\n"
   "\n"
   "\n"
   "/*The number of extra bits of precision at which to store rate"
   " metrics.*/\n"
   "# define OC_BIT_SCALE  (%i)\n"
   "/*The number of extra bits of precision at which to store RMSE metrics.\n"
   "  This must be at least half OC_BIT_SCALE (rounded up).*/\n"
   "# define OC_RMSE_SCALE (%i)\n"
   "/*The number of bins to partition statistics into.*/\n"
   "# define OC_SAD_BINS   (%i)\n"
   "/*The number of bits of precision to drop"
   " from SAD scores to assign them to a\n"
   "   bin.*/\n"
   "# define OC_SAD_SHIFT  (%i)\n"
   "\n"
   "\n"
   "\n"
   "# if defined(OC_COLLECT_METRICS)\n"
   "struct oc_mode_metrics{\n"
   "  double fragw;\n"
   "  double satd;\n"
   "  double rate;\n"
   "  double rmse;\n"
   "  double satd2;\n"
   "  double satdrate;\n"
   "  double rate2;\n"
   "  double satdrmse;\n"
   "  double rmse2;\n"
   "};\n"
   "\n"
   "\n"
   "int             oc_has_mode_metrics;\n"
   "oc_mode_metrics OC_MODE_METRICS[64][3][2][OC_SAD_BINS];\n"
   "# endif\n"
   "\n"
   "\n"
   "\n"
   "struct oc_mode_rd{\n"
   "  ogg_int16_t rate;\n"
   "  ogg_int16_t rmse;\n"
   "};\n"
   "\n"
   "\n"
   "# if !defined(OC_COLLECT_METRICS)\n"
   "static const\n"
   "# endif\n"
   "oc_mode_rd OC_MODE_RD[64][3][2][OC_SAD_BINS]={\n",
   OC_BIT_SCALE,OC_RMSE_SCALE,OC_SAD_BINS,OC_SAD_SHIFT);
  for(qi=0;qi<64;qi++){
    int pli;
    fprintf(stdout,"  {\n");
    for(pli=0;pli<3;pli++){
      int qti;
      fprintf(stdout,"    {\n");
      for(qti=0;qti<2;qti++){
        int bin;
        static const char *pl_names[3]={"Y'","Cb","Cr"};
        static const char *qti_names[2]={"INTRA","INTER"};
        fprintf(stdout,"      /*%s  qi=%i  %s*/\n",
         pl_names[pli],qi,qti_names[qti]);
        fprintf(stdout,"      {\n");
        fprintf(stdout,"        ");
        for(bin=0;bin<OC_SAD_BINS;bin++){
          if(bin&&!(bin&0x3))fprintf(stdout,"\n        ");
          fprintf(stdout,"{%5i,%5i}",
           OC_MODE_RD[qi][pli][qti][bin].rate,
           OC_MODE_RD[qi][pli][qti][bin].rmse);
          if(bin+1<OC_SAD_BINS)fprintf(stdout,",");
        }
        fprintf(stdout,"\n      }");
        if(qti<1)fprintf(stdout,",");
        fprintf(stdout,"\n");
      }
      fprintf(stdout,"    }");
      if(pli<2)fprintf(stdout,",");
      fprintf(stdout,"\n");
    }
    fprintf(stdout,"  }");
    if(qi<63)fprintf(stdout,",");
    fprintf(stdout,"\n");
  }
  fprintf(stdout,
   "};\n"
   "\n"
   "#endif\n");
}
#endif
