/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009,2025           *
 * by the Xiph.Org Foundation https://www.xiph.org/                 *
 *                                                                  *
 ********************************************************************

  function: mode selection code

 ********************************************************************/
#include <limits.h>
#include <string.h>
#include "encint.h"
#include "modedec.h"
#if defined(OC_COLLECT_METRICS)
# include "collect.c"
#endif



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

/*Return the cost of coding _mb_mode in the specified scheme.*/
static int oc_mode_scheme_chooser_scheme_mb_cost(
 const oc_mode_scheme_chooser *_chooser,int _scheme,int _mb_mode){
  int codebook;
  int ri;
  codebook=_scheme+1>>3;
  /*For any scheme except 0, we can just use the bit cost of the mode's rank
     in that scheme.*/
  ri=_chooser->mode_ranks[_scheme][_mb_mode];
  if(_scheme==0){
    int mc;
    /*For scheme 0, incrementing the mode count could potentially change the
       mode's rank.
      Find the index where the mode would be moved to in the optimal list,
       and use its bit cost instead of the one for the mode's current
       position in the list.*/
    /*We don't actually reorder the list; this is for computing opportunity
       cost, not an update.*/
    mc=_chooser->mode_counts[_mb_mode];
    while(ri>0&&mc>=_chooser->mode_counts[_chooser->scheme0_list[ri-1]])ri--;
  }
  return OC_MODE_BITS[codebook][ri];
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
  int scheme0_bits;
  int scheme1_bits;
  scheme0=_chooser->scheme_list[0];
  scheme1=_chooser->scheme_list[1];
  scheme0_bits=_chooser->scheme_bits[scheme0];
  scheme1_bits=_chooser->scheme_bits[scheme1];
  mode_bits=oc_mode_scheme_chooser_scheme_mb_cost(_chooser,scheme0,_mb_mode);
  /*Typical case: If the difference between the best scheme and the next best
     is greater than 6 bits, then adding just one mode cannot change which
     scheme we use.*/
  if(scheme1_bits-scheme0_bits>6)return mode_bits;
  /*Otherwise, check to see if adding this mode selects a different scheme as
     the best.*/
  si=1;
  best_bits=scheme0_bits+mode_bits;
  do{
    int cur_bits;
    cur_bits=scheme1_bits+
     oc_mode_scheme_chooser_scheme_mb_cost(_chooser,scheme1,_mb_mode);
    if(cur_bits<best_bits)best_bits=cur_bits;
    if(++si>=8)break;
    scheme1=_chooser->scheme_list[si];
    scheme1_bits=_chooser->scheme_bits[scheme1];
  }
  while(scheme1_bits-scheme0_bits<=6);
  return best_bits-scheme0_bits;
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



static void oc_fr_state_init(oc_fr_state *_fr){
  _fr->bits=0;
  _fr->sb_partial_count=0;
  _fr->sb_full_count=0;
  _fr->b_coded_count_prev=0;
  _fr->b_coded_count=0;
  _fr->b_count=0;
  _fr->sb_prefer_partial=0;
  _fr->sb_bits=0;
  _fr->sb_partial=-1;
  _fr->sb_full=-1;
  _fr->b_coded_prev=-1;
  _fr->b_coded=-1;
}


static int oc_fr_state_sb_cost(const oc_fr_state *_fr,
 int _sb_partial,int _sb_full){
  int bits;
  int sb_partial_count;
  int sb_full_count;
  bits=0;
  sb_partial_count=_fr->sb_partial_count;
  /*Extend the sb_partial run, or start a new one.*/
  if(_fr->sb_partial==_sb_partial){
    if(sb_partial_count>=4129){
      bits++;
      sb_partial_count=0;
    }
    else bits-=oc_sb_run_bits(sb_partial_count);
  }
  else sb_partial_count=0;
  bits+=oc_sb_run_bits(++sb_partial_count);
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
    bits+=oc_sb_run_bits(++sb_full_count);
  }
  return bits;
}

static void oc_fr_state_advance_sb(oc_fr_state *_fr,
 int _sb_partial,int _sb_full){
  int sb_partial_count;
  int sb_full_count;
  sb_partial_count=_fr->sb_partial_count;
  if(_fr->sb_partial!=_sb_partial||sb_partial_count>=4129)sb_partial_count=0;
  sb_partial_count++;
  if(!_sb_partial){
    sb_full_count=_fr->sb_full_count;
    if(_fr->sb_full!=_sb_full||sb_full_count>=4129)sb_full_count=0;
    sb_full_count++;
    _fr->sb_full_count=sb_full_count;
    _fr->sb_full=_sb_full;
    /*Roll back the partial block state.*/
    _fr->b_coded=_fr->b_coded_prev;
    _fr->b_coded_count=_fr->b_coded_count_prev;
  }
  else{
    /*Commit back the partial block state.*/
    _fr->b_coded_prev=_fr->b_coded;
    _fr->b_coded_count_prev=_fr->b_coded_count;
  }
  _fr->sb_partial_count=sb_partial_count;
  _fr->sb_partial=_sb_partial;
  _fr->b_count=0;
  _fr->sb_prefer_partial=0;
  _fr->sb_bits=0;
}

/*Commit the state of the current super block and advance to the next.*/
static void oc_fr_state_flush_sb(oc_fr_state *_fr){
  int sb_partial;
  int sb_full;
  int b_coded_count;
  int b_count;
  b_count=_fr->b_count;
  b_coded_count=_fr->b_coded_count;
  sb_full=_fr->b_coded;
  sb_partial=b_coded_count<b_count;
  if(!sb_partial){
    /*If the super block is fully coded/uncoded...*/
    if(_fr->sb_prefer_partial){
      /*So far coding this super block as partial was cheaper anyway.*/
      if(b_coded_count>15||_fr->b_coded_prev<0){
        int sb_bits;
        /*If the block run is too long, this will limit how far it can be
           extended into the next partial super block.
          If we need to extend it farther, we don't want to have to roll all
           the way back here (since there could be many full SBs between now
           and then), so we disallow this.
          Similarly, if this is the start of a stripe, we don't know how the
           length of the outstanding block run from the previous stripe.*/
        sb_bits=oc_fr_state_sb_cost(_fr,sb_partial,sb_full);
        _fr->bits+=sb_bits-_fr->sb_bits;
        _fr->sb_bits=sb_bits;
      }
      else sb_partial=1;
    }
  }
  oc_fr_state_advance_sb(_fr,sb_partial,sb_full);
}

static void oc_fr_state_advance_block(oc_fr_state *_fr,int _b_coded){
  ptrdiff_t bits;
  int       sb_bits;
  int       b_coded_count;
  int       b_count;
  int       sb_prefer_partial;
  sb_bits=_fr->sb_bits;
  bits=_fr->bits-sb_bits;
  b_count=_fr->b_count;
  b_coded_count=_fr->b_coded_count;
  sb_prefer_partial=_fr->sb_prefer_partial;
  if(b_coded_count>=b_count){
    int sb_partial_bits;
    /*This super block is currently fully coded/uncoded.*/
    if(b_count<=0){
      /*This is the first block in this SB.*/
      b_count=1;
      /*Check to see whether it's cheaper to code it partially or fully.*/
      if(_fr->b_coded==_b_coded){
        sb_partial_bits=-oc_block_run_bits(b_coded_count);
        sb_partial_bits+=oc_block_run_bits(++b_coded_count);
      }
      else{
        b_coded_count=1;
        sb_partial_bits=2;
      }
      sb_partial_bits+=oc_fr_state_sb_cost(_fr,1,_b_coded);
      sb_bits=oc_fr_state_sb_cost(_fr,0,_b_coded);
      sb_prefer_partial=sb_partial_bits<sb_bits;
      sb_bits^=(sb_partial_bits^sb_bits)&-sb_prefer_partial;
    }
    else if(_fr->b_coded==_b_coded){
      b_coded_count++;
      if(++b_count<16){
        if(sb_prefer_partial){
          /*Check to see if it's cheaper to code it fully.*/
          sb_partial_bits=sb_bits;
          sb_partial_bits+=oc_block_run_bits(b_coded_count);
          if(b_coded_count>0){
            sb_partial_bits-=oc_block_run_bits(b_coded_count-1);
          }
          sb_bits=oc_fr_state_sb_cost(_fr,0,_b_coded);
          sb_prefer_partial=sb_partial_bits<sb_bits;
          sb_bits^=(sb_partial_bits^sb_bits)&-sb_prefer_partial;
        }
        /*There's no need to check the converse (whether it's cheaper to code
           this SB partially if we were coding it fully), since the cost to
           code a SB partially can only increase as we add more blocks, whereas
           the cost to code it fully stays constant.*/
      }
      else{
        /*If we get to the end and this SB is still full, then force it to be
           coded full.
          Otherwise we might not be able to extend the block run far enough
           into the next partial SB.*/
        if(sb_prefer_partial){
          sb_prefer_partial=0;
          sb_bits=oc_fr_state_sb_cost(_fr,0,_b_coded);
        }
      }
    }
    else{
      /*This SB was full, but now must be made partial.*/
      if(!sb_prefer_partial){
        sb_bits=oc_block_run_bits(b_coded_count);
        if(b_coded_count>b_count){
          sb_bits-=oc_block_run_bits(b_coded_count-b_count);
        }
        sb_bits+=oc_fr_state_sb_cost(_fr,1,_b_coded);
      }
      b_count++;
      b_coded_count=1;
      sb_prefer_partial=1;
      sb_bits+=2;
    }
  }
  else{
    b_count++;
    if(_fr->b_coded==_b_coded)sb_bits-=oc_block_run_bits(b_coded_count);
    else b_coded_count=0;
    sb_bits+=oc_block_run_bits(++b_coded_count);
  }
  _fr->bits=bits+sb_bits;
  _fr->b_coded_count=b_coded_count;
  _fr->b_coded=_b_coded;
  _fr->b_count=b_count;
  _fr->sb_prefer_partial=sb_prefer_partial;
  _fr->sb_bits=sb_bits;
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



static void oc_enc_pipeline_init(oc_enc_ctx *_enc,oc_enc_pipeline_state *_pipe){
  ptrdiff_t *coded_fragis;
  unsigned   mcu_nvsbs;
  ptrdiff_t  mcu_nfrags;
  int        flimit;
  int        hdec;
  int        vdec;
  int        pli;
  int        nqis;
  int        qii;
  int        qi0;
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
  qi0=_enc->state.qis[0];
  nqis=_enc->state.nqis;
  for(pli=0;pli<3;pli++){
    for(qii=0;qii<nqis;qii++){
      int qi;
      qi=_enc->state.qis[qii];
      for(qti=0;qti<2;qti++){
        /*Set the DC coefficient in the dequantization table.*/
        _enc->state.dequant_tables[qi][pli][qti][0]=
         _enc->dequant_dc[qi0][pli][qti];
        _enc->dequant[pli][qii][qti]=_enc->state.dequant_tables[qi][pli][qti];
        /*Copy over the quantization table.*/
        memcpy(_enc->enquant[pli][qii][qti],_enc->enquant_tables[qi][pli][qti],
         _enc->opt_data.enquant_table_size);
      }
    }
  }
  /*Fix up the DC coefficients in the quantization tables.*/
  oc_enc_enquant_table_fixup(_enc,_enc->enquant,nqis);
  /*Initialize the tokenization state.*/
  for(pli=0;pli<3;pli++){
    _pipe->ndct_tokens1[pli]=0;
    _pipe->eob_run1[pli]=0;
  }
  /*Initialize the bounding value array for the loop filter.*/
  flimit=_enc->state.loop_filter_limits[_enc->state.qis[0]];
  _pipe->loop_filter=flimit!=0;
  if(flimit!=0)oc_loop_filter_init(&_enc->state,_pipe->bounding_values,flimit);
  /*Clear the temporary DCT scratch space.*/
  memset(_pipe->dct_data,0,sizeof(_pipe->dct_data));
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
  /*Copy over all the uncoded fragments from this plane and advance the uncoded
     fragment list.*/
  if(_pipe->nuncoded_fragis[_pli]>0){
    _pipe->uncoded_fragis[_pli]-=_pipe->nuncoded_fragis[_pli];
    oc_frag_copy_list(&_enc->state,
     _enc->state.ref_frame_data[OC_FRAME_SELF],
     _enc->state.ref_frame_data[OC_FRAME_PREV],
     _enc->state.ref_ystride[_pli],_pipe->uncoded_fragis[_pli],
     _pipe->nuncoded_fragis[_pli],_enc->state.frag_buf_offs);
    _pipe->nuncoded_fragis[_pli]=0;
  }
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
  if(_pipe->loop_filter){
    oc_state_loop_filter_frag_rows(&_enc->state,
     _pipe->bounding_values,OC_FRAME_SELF,_pli,
     _pipe->fragy0[_pli]-_sdelay,_pipe->fragy_end[_pli]-_edelay);
  }
  else _sdelay=_edelay=0;
  /*To fill borders, we have an additional two pixel delay, since a fragment
     in the next row could filter its top edge, using two pixels from a
     fragment in this row.
    But there's no reason to delay a full fragment between the two.*/
  oc_state_borders_fill_rows(&_enc->state,
   _enc->state.ref_frame_idx[OC_FRAME_SELF],_pli,
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
 oc_enc_pipeline_state *_pipe,int _pli,ptrdiff_t _fragi,
 unsigned _rd_scale,unsigned _rd_iscale,oc_rd_metric *_mo,
 oc_fr_state *_fr,oc_token_checkpoint **_stack){
  ogg_int16_t            *data;
  ogg_int16_t            *dct;
  ogg_int16_t            *idct;
  oc_qii_state            qs;
  const ogg_uint16_t     *dequant;
  ogg_uint16_t            dequant_dc;
  ptrdiff_t               frag_offs;
  int                     ystride;
  const unsigned char    *src;
  const unsigned char    *ref;
  unsigned char          *dst;
  int                     nonzero;
  unsigned                uncoded_ssd;
  unsigned                coded_ssd;
  oc_token_checkpoint    *checkpoint;
  oc_fragment            *frags;
  int                     mb_mode;
  int                     refi;
  int                     mv_offs[2];
  int                     nmv_offs;
  int                     ac_bits;
  int                     borderi;
  int                     nqis;
  int                     qti;
  int                     qii;
  int                     dc;
  nqis=_enc->state.nqis;
  frags=_enc->state.frags;
  frag_offs=_enc->state.frag_buf_offs[_fragi];
  ystride=_enc->state.ref_ystride[_pli];
  src=_enc->state.ref_frame_data[OC_FRAME_IO]+frag_offs;
  borderi=frags[_fragi].borderi;
  qii=frags[_fragi].qii;
  data=_enc->pipe.dct_data;
  dct=data+64;
  idct=data+128;
  if(qii&~3){
#if !defined(OC_COLLECT_METRICS)
    if(_enc->sp_level>=OC_SP_LEVEL_EARLY_SKIP){
      /*Enable early skip detection.*/
      frags[_fragi].coded=0;
      frags[_fragi].refi=OC_FRAME_NONE;
      oc_fr_skip_block(_fr);
      return 0;
    }
#endif
    /*Try and code this block anyway.*/
    qii&=3;
  }
  refi=frags[_fragi].refi;
  mb_mode=frags[_fragi].mb_mode;
  ref=_enc->state.ref_frame_data[refi]+frag_offs;
  dst=_enc->state.ref_frame_data[OC_FRAME_SELF]+frag_offs;
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
      frag_mvs=_enc->state.frag_mvs;
      nmv_offs=oc_state_get_mv_offsets(&_enc->state,mv_offs,
       _pli,frag_mvs[_fragi]);
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
    unsigned sad;
    unsigned satd;
    switch(nmv_offs){
      case 0:{
        sad=oc_enc_frag_intra_sad(_enc,src,ystride);
        satd=oc_enc_frag_intra_satd(_enc,&dc,src,ystride);
      }break;
      case 1:{
        sad=oc_enc_frag_sad_thresh(_enc,src,ref+mv_offs[0],ystride,UINT_MAX);
        satd=oc_enc_frag_satd(_enc,&dc,src,ref+mv_offs[0],ystride);
        satd+=abs(dc);
      }break;
      default:{
        sad=oc_enc_frag_sad_thresh(_enc,src,dst,ystride,UINT_MAX);
        satd=oc_enc_frag_satd(_enc,&dc,src,dst,ystride);
        satd+=abs(dc);
      }break;
    }
    _enc->frag_sad[_fragi]=sad;
    _enc->frag_satd[_fragi]=satd;
  }
#endif
  /*Transform:*/
  oc_enc_fdct8x8(_enc,dct,data);
  /*Quantize:*/
  qti=mb_mode!=OC_MODE_INTRA;
  dequant=_enc->dequant[_pli][qii][qti];
  nonzero=oc_enc_quantize(_enc,data,dct,dequant,_enc->enquant[_pli][qii][qti]);
  dc=data[0];
  /*Tokenize.*/
  checkpoint=*_stack;
  if(_enc->sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
    ac_bits=oc_enc_tokenize_ac(_enc,_pli,_fragi,idct,data,dequant,dct,
     nonzero+1,_stack,OC_RD_ISCALE(_enc->lambda,_rd_iscale),qti?0:3);
  }
  else{
    ac_bits=oc_enc_tokenize_ac_fast(_enc,_pli,_fragi,idct,data,dequant,dct,
     nonzero+1,_stack,OC_RD_ISCALE(_enc->lambda,_rd_iscale),qti?0:3);
  }
  /*Reconstruct.
    TODO: nonzero may need to be adjusted after tokenization.*/
  dequant_dc=dequant[0];
  if(nonzero==0){
    ogg_int16_t p;
    int         ci;
    int         qi01;
    int         qi12;
    /*We round this dequant product (and not any of the others) because there's
       no iDCT rounding.*/
    p=(ogg_int16_t)(dc*(ogg_int32_t)dequant_dc+15>>5);
    /*LOOP VECTORIZES.*/
    for(ci=0;ci<64;ci++)data[ci]=p;
    /*We didn't code any AC coefficients, so don't change the quantizer.*/
    qi01=_pipe->qs[_pli].qi01;
    qi12=_pipe->qs[_pli].qi12;
    if(qi01>0)qii=1+qi12;
    else if(qi01>=0)qii=0;
  }
  else{
    idct[0]=dc*dequant_dc;
    /*Note: This clears idct[] back to zero for the next block.*/
    oc_idct8x8(&_enc->state,data,idct,nonzero+1);
  }
  frags[_fragi].qii=qii;
  if(nqis>1){
    oc_qii_state_advance(&qs,_pipe->qs+_pli,qii);
    ac_bits+=qs.bits-_pipe->qs[_pli].bits;
  }
  if(!qti)oc_enc_frag_recon_intra(_enc,dst,ystride,data);
  else{
    oc_enc_frag_recon_inter(_enc,dst,
     nmv_offs==1?ref+mv_offs[0]:dst,ystride,data);
  }
  /*If _fr is NULL, then this is an INTRA frame, and we can't skip blocks.*/
#if !defined(OC_COLLECT_METRICS)
  if(_fr!=NULL)
#endif
  {
    /*In retrospect, should we have skipped this block?*/
    if(borderi<0){
      coded_ssd=oc_enc_frag_ssd(_enc,src,dst,ystride);
    }
    else{
      coded_ssd=oc_enc_frag_border_ssd(_enc,src,dst,ystride,
       _enc->state.borders[borderi].mask);
    }
    /*Scale to match DCT domain.*/
    coded_ssd<<=4;
#if defined(OC_COLLECT_METRICS)
    _enc->frag_ssd[_fragi]=coded_ssd;
  }
  if(_fr!=NULL){
#endif
    coded_ssd=OC_RD_SCALE(coded_ssd,_rd_scale);
    uncoded_ssd=_pipe->skip_ssd[_pli][_fragi-_pipe->froffset[_pli]];
    if(uncoded_ssd<UINT_MAX&&
     /*Don't allow luma blocks to be skipped in 4MV mode when VP3 compatibility
        is enabled.*/
     (!_enc->vp3_compatible||mb_mode!=OC_MODE_INTER_MV_FOUR||_pli)){
      int overhead_bits;
      overhead_bits=oc_fr_cost1(_fr);
      /*Although the fragment coding overhead determination is accurate, it is
         greedy, using very coarse-grained local information.
        Allowing it to mildly discourage coding turns out to be beneficial, but
         it's not clear that allowing it to encourage coding through negative
         coding overhead deltas is useful.
        For that reason, we disallow negative coding overheads.*/
      if(overhead_bits<0)overhead_bits=0;
      if(uncoded_ssd<=coded_ssd+(overhead_bits+ac_bits)*_enc->lambda){
        /*Hm, not worth it; roll back.*/
        oc_enc_tokenlog_rollback(_enc,checkpoint,(*_stack)-checkpoint);
        *_stack=checkpoint;
        frags[_fragi].coded=0;
        frags[_fragi].refi=OC_FRAME_NONE;
        oc_fr_skip_block(_fr);
        return 0;
      }
    }
    else _mo->dc_flag=1;
    _mo->uncoded_ac_ssd+=uncoded_ssd;
    _mo->coded_ac_ssd+=coded_ssd;
    _mo->ac_bits+=ac_bits;
    oc_fr_code_block(_fr);
  }
  /*GCC 4.4.4 generates a warning here because it can't tell that
     the init code in the nqis check above will run anytime this
     line runs.*/
  if(nqis>1)*(_pipe->qs+_pli)=*&qs;
  frags[_fragi].dc=dc;
  frags[_fragi].coded=1;
  return 1;
}

static int oc_enc_mb_transform_quantize_inter_luma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,unsigned _mbi,int _mode_overhead,
 const unsigned _rd_scale[4],const unsigned _rd_iscale[4]){
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
  int                  refi;
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
  refi=OC_FRAME_FOR_MODE(mb_mode);
  ncoded=0;
  stackptr=stack;
  memset(&mo,0,sizeof(mo));
  for(bi=0;bi<4;bi++){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frags[fragi].refi=refi;
    frags[fragi].mb_mode=mb_mode;
    if(oc_enc_block_transform_quantize(_enc,_pipe,0,fragi,
     _rd_scale[bi],_rd_iscale[bi],&mo,_pipe->fr+0,&stackptr)){
      coded_fragis[ncoded_fragis++]=fragi;
      ncoded++;
    }
    else *(uncoded_fragis-++nuncoded_fragis)=fragi;
  }
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
          frags[fragi].refi=OC_FRAME_NONE;
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
  _pipe->ncoded_fragis[0]=ncoded_fragis;
  _pipe->nuncoded_fragis[0]=nuncoded_fragis;
  return ncoded;
}

static void oc_enc_sb_transform_quantize_inter_chroma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,int _sbi_start,int _sbi_end){
  const ogg_uint16_t *mcu_rd_scale;
  const ogg_uint16_t *mcu_rd_iscale;
  const oc_sb_map    *sb_maps;
  oc_sb_flags        *sb_flags;
  oc_fr_state        *fr;
  ptrdiff_t          *coded_fragis;
  ptrdiff_t           ncoded_fragis;
  ptrdiff_t          *uncoded_fragis;
  ptrdiff_t           nuncoded_fragis;
  ptrdiff_t           froffset;
  int                 sbi;
  fr=_pipe->fr+_pli;
  mcu_rd_scale=(const ogg_uint16_t *)_enc->mcu_rd_scale;
  mcu_rd_iscale=(const ogg_uint16_t *)_enc->mcu_rd_iscale;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  sb_flags=_enc->state.sb_flags;
  coded_fragis=_pipe->coded_fragis[_pli];
  ncoded_fragis=_pipe->ncoded_fragis[_pli];
  uncoded_fragis=_pipe->uncoded_fragis[_pli];
  nuncoded_fragis=_pipe->nuncoded_fragis[_pli];
  froffset=_pipe->froffset[_pli];
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
        unsigned             rd_scale;
        unsigned             rd_iscale;
        rd_scale=mcu_rd_scale[fragi-froffset];
        rd_iscale=mcu_rd_iscale[fragi-froffset];
        stackptr=stack;
        if(oc_enc_block_transform_quantize(_enc,_pipe,_pli,fragi,
         rd_scale,rd_iscale,&mo,fr,&stackptr)){
          coded_fragis[ncoded_fragis++]=fragi;
        }
        else *(uncoded_fragis-++nuncoded_fragis)=fragi;
      }
    }
    oc_fr_state_flush_sb(fr);
    sb_flags[sbi].coded_fully=fr->sb_full;
    sb_flags[sbi].coded_partially=fr->sb_partial;
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
   dividing them among all the blocks in the run; the latter approach seems
   more theoretically correct, but Monty's testing showed a small improvement
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

static void oc_enc_mode_rd_init(oc_enc_ctx *_enc){
#if !defined(OC_COLLECT_METRICS)
  const
#endif
  oc_mode_rd (*oc_mode_rd_table)[3][2][OC_COMP_BINS]=
   _enc->sp_level<OC_SP_LEVEL_NOSATD?OC_MODE_RD_SATD:OC_MODE_RD_SAD;
  int qii;
#if defined(OC_COLLECT_METRICS)
  oc_enc_mode_metrics_load(_enc);
#endif
  for(qii=0;qii<_enc->state.nqis;qii++){
    int qi;
    int pli;
    qi=_enc->state.qis[qii];
    for(pli=0;pli<3;pli++){
      int qti;
      for(qti=0;qti<2;qti++){
        int log_plq;
        int modeline;
        int bin;
        int dx;
        int dq;
        log_plq=_enc->log_plq[qi][pli][qti];
        /*Find the pair of rows in the mode table that bracket this quantizer.
          If it falls outside the range the table covers, then we just use a
           pair on the edge for linear extrapolation.*/
        for(modeline=0;modeline<OC_LOGQ_BINS-1&&
         OC_MODE_LOGQ[modeline+1][pli][qti]>log_plq;modeline++);
        /*Interpolate a row for this quantizer.*/
        dx=OC_MODE_LOGQ[modeline][pli][qti]-log_plq;
        dq=OC_MODE_LOGQ[modeline][pli][qti]-OC_MODE_LOGQ[modeline+1][pli][qti];
        if(dq==0)dq=1;
        for(bin=0;bin<OC_COMP_BINS;bin++){
          int y0;
          int z0;
          int dy;
          int dz;
          y0=oc_mode_rd_table[modeline][pli][qti][bin].rate;
          z0=oc_mode_rd_table[modeline][pli][qti][bin].rmse;
          dy=oc_mode_rd_table[modeline+1][pli][qti][bin].rate-y0;
          dz=oc_mode_rd_table[modeline+1][pli][qti][bin].rmse-z0;
          _enc->mode_rd[qii][pli][qti][bin].rate=
           (ogg_int16_t)OC_CLAMPI(-32768,y0+(dy*dx+(dq>>1))/dq,32767);
          _enc->mode_rd[qii][pli][qti][bin].rmse=
           (ogg_int16_t)OC_CLAMPI(-32768,z0+(dz*dx+(dq>>1))/dq,32767);
        }
      }
    }
  }
}

/*Estimate the R-D cost of the DCT coefficients given the SATD of a block after
   prediction.*/
static unsigned oc_dct_cost2(oc_enc_ctx *_enc,unsigned *_ssd,
 int _qii,int _pli,int _qti,int _satd){
  unsigned rmse;
  int      shift;
  int      bin;
  int      dx;
  int      y0;
  int      z0;
  int      dy;
  int      dz;
  /*SATD metrics for chroma planes vary much less than luma, so we scale them
     by 4 to distribute them into the mode decision bins more evenly.*/
  _satd<<=_pli+1&2;
  shift=_enc->sp_level<OC_SP_LEVEL_NOSATD?OC_SATD_SHIFT:OC_SAD_SHIFT;
  bin=OC_MINI(_satd>>shift,OC_COMP_BINS-2);
  dx=_satd-(bin<<shift);
  y0=_enc->mode_rd[_qii][_pli][_qti][bin].rate;
  z0=_enc->mode_rd[_qii][_pli][_qti][bin].rmse;
  dy=_enc->mode_rd[_qii][_pli][_qti][bin+1].rate-y0;
  dz=_enc->mode_rd[_qii][_pli][_qti][bin+1].rmse-z0;
  rmse=OC_MAXI(z0+(dz*dx>>shift),0);
  *_ssd=rmse*rmse>>2*OC_RMSE_SCALE-OC_BIT_SCALE;
  return OC_MAXI(y0+(dy*dx>>shift),0);
}

/*activity_avg must be positive, or flat regions could get a zero weight, which
   confounds analysis.
  We set the minimum to this value so that it also avoids the need for divide
   by zero checks in oc_mb_masking().*/
# define OC_ACTIVITY_AVG_MIN (1<<OC_RD_SCALE_BITS)

static unsigned oc_mb_activity(oc_enc_ctx *_enc,unsigned _mbi,
 unsigned _activity[4]){
  const unsigned char *src;
  const ptrdiff_t     *frag_buf_offs;
  const ptrdiff_t     *sb_map;
  unsigned             luma;
  int                  ystride;
  ptrdiff_t            frag_offs;
  ptrdiff_t            fragi;
  int                  bi;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[0];
  luma=0;
  for(bi=0;bi<4;bi++){
    const unsigned char *s;
    unsigned             x;
    unsigned             x2;
    unsigned             act;
    int                  i;
    int                  j;
    fragi=sb_map[bi];
    frag_offs=frag_buf_offs[fragi];
    /*TODO: This could be replaced with SATD^2, since we already have to
       compute SATD.*/
    x=x2=0;
    s=src+frag_offs;
    for(i=0;i<8;i++){
      for(j=0;j<8;j++){
        unsigned c;
        c=s[j];
        x+=c;
        x2+=c*c;
      }
      s+=ystride;
    }
    luma+=x;
    act=(x2<<6)-x*x;
    if(act<8<<12){
      /*The region is flat.*/
      act=OC_MINI(act,5<<12);
    }
    else{
      unsigned e1;
      unsigned e2;
      unsigned e3;
      unsigned e4;
      /*Test for an edge.
        TODO: There are probably much simpler ways to do this (e.g., it could
         probably be combined with the SATD calculation).
        Alternatively, we could split the block around the mean and compute the
         reduction in variance in each half.
        For a Gaussian source the reduction should be
         (1-2/pi) ~= 0.36338022763241865692446494650994.
        Significantly more reduction is a good indication of a bi-level image.
        This has the advantage of identifying, in addition to straight edges,
         small text regions, which would otherwise be classified as "texture".*/
      e1=e2=e3=e4=0;
      s=src+frag_offs-1;
      for(i=0;i<8;i++){
        for(j=0;j<8;j++){
          e1+=abs((s[j+2]-s[j]<<1)+(s-ystride)[j+2]-(s-ystride)[j]
           +(s+ystride)[j+2]-(s+ystride)[j]);
          e2+=abs(((s+ystride)[j+1]-(s-ystride)[j+1]<<1)
           +(s+ystride)[j]-(s-ystride)[j]+(s+ystride)[j+2]-(s-ystride)[j+2]);
          e3+=abs(((s+ystride)[j+2]-(s-ystride)[j]<<1)
           +(s+ystride)[j+1]-s[j]+s[j+2]-(s-ystride)[j+1]);
          e4+=abs(((s+ystride)[j]-(s-ystride)[j+2]<<1)
           +(s+ystride)[j+1]-s[j+2]+s[j]-(s-ystride)[j+1]);
        }
        s+=ystride;
      }
      /*If the largest component of the edge energy is at least 40% of the
         total, then classify the block as an edge block.*/
      if(5*OC_MAXI(OC_MAXI(e1,e2),OC_MAXI(e3,e4))>2*(e1+e2+e3+e4)){
         /*act=act_th*(act/act_th)**0.7
              =exp(log(act_th)+0.7*(log(act)-log(act_th))).
           Here act_th=5.0 and 0x394A=oc_blog32_q10(5<<12).*/
         act=oc_bexp32_q10(0x394A+(7*(oc_blog32_q10(act)-0x394A+5)/10));
      }
    }
    _activity[bi]=act;
  }
  return luma;
}

static void oc_mb_activity_fast(oc_enc_ctx *_enc,unsigned _mbi,
 unsigned _activity[4],const unsigned _intra_satd[12]){
  int bi;
  for(bi=0;bi<4;bi++){
    unsigned act;
    act=(11*_intra_satd[bi]>>8)*_intra_satd[bi];
    if(act<8<<12){
      /*The region is flat.*/
      act=OC_MINI(act,5<<12);
    }
    _activity[bi]=act;
  }
}

/*Compute the masking scales for the blocks in a macro block.
  All masking is computed from the luma blocks.
  We derive scaling factors for the chroma blocks from these, and use the same
   ones for all chroma blocks, regardless of the subsampling.
  It's possible for luma to be perfectly flat and yet have high chroma energy,
   but this is unlikely in non-artificial images, and not a case that has been
   addressed by any research to my knowledge.
  The output of the masking process is two scale factors, which are fed into
   the various R-D optimizations.
  The first, rd_scale, is applied to D in the equation
    D*rd_scale+lambda*R.
  This is the form that must be used to properly combine scores from multiple
   blocks, and can be interpreted as scaling distortions by their visibility.
  The inverse, rd_iscale, is applied to lambda in the equation
    D+rd_iscale*lambda*R.
  This is equivalent to the first form within a single block, but much faster
   to use when evaluating many possible distortions (e.g., during actual
   quantization, where separate distortions are evaluated for every
   coefficient).
  The two macros OC_RD_SCALE(rd_scale,d) and OC_RD_ISCALE(rd_iscale,lambda) are
   used to perform the multiplications with the proper re-scaling for the range
   of the scaling factors.
  Many researchers apply masking values directly to the quantizers used, and
   not to the R-D cost.
  Since we generally use MSE for D, rd_scale must use the square of their
   values to generate an equivalent effect.*/
static unsigned oc_mb_masking(unsigned _rd_scale[5],unsigned _rd_iscale[5],
 const ogg_uint16_t _chroma_rd_scale[2],const unsigned _activity[4],
 unsigned _activity_avg,unsigned _luma,unsigned _luma_avg){
  unsigned activity_sum;
  unsigned la;
  unsigned lb;
  unsigned d;
  int      bi;
  int      bi_min;
  int      bi_min2;
  /*The ratio lb/la is meant to approximate
     ((((_luma-16)/219)*(255/128))**0.649**0.4**2), which is the
     effective luminance masking from~\cite{LKW06} (including the self-masking
     deflator).
    The following actually turns out to be a pretty good approximation for
     _luma>75 or so.
    For smaller values luminance does not really follow Weber's Law anyway, and
     this approximation gives a much less aggressive bitrate boost in this
     region.
    Though some researchers claim that contrast sensitivity actually decreases
     for very low luminance values, in my experience excessive brightness on
     LCDs or buggy color conversions (e.g., treating Y' as full-range instead
     of the CCIR 601 range) make artifacts in such regions extremely visible.
    We substitute _luma_avg for 128 to allow the strength of the masking to
     vary with the actual average image luminance, within certain limits (the
     caller has clamped _luma_avg to the range [90,160], inclusive).
    @ARTICLE{LKW06,
      author="Zhen Liu and Lina J. Karam and Andrew B. Watson",
      title="{JPEG2000} Encoding With Perceptual Distortion Control",
      journal="{IEEE} Transactions on Image Processing",
      volume=15,
      number=7,
      pages="1763--1778",
      month=Jul,
      year=2006
    }*/
#if 0
  la=_luma+4*_luma_avg;
  lb=4*_luma+_luma_avg;
#else
  /*Disable luminance masking.*/
  la=lb=1;
#endif
  activity_sum=0;
  for(bi=0;bi<4;bi++){
    unsigned a;
    unsigned b;
    activity_sum+=_activity[bi];
    /*Apply activity masking.*/
    a=_activity[bi]+4*_activity_avg;
    b=4*_activity[bi]+_activity_avg;
    d=OC_RD_SCALE(b,1);
    /*And luminance masking.*/
    d=(a+(d>>1))/d;
    _rd_scale[bi]=(d*la+(lb>>1))/lb;
    /*And now the inverse.*/
    d=OC_MAXI(OC_RD_ISCALE(a,1),1);
    d=(b+(d>>1))/d;
    _rd_iscale[bi]=(d*lb+(la>>1))/la;
  }
  /*Now compute scaling factors for chroma blocks.
    We start by finding the two smallest iscales from the luma blocks.*/
  bi_min=_rd_iscale[1]<_rd_iscale[0];
  bi_min2=1-bi_min;
  for(bi=2;bi<4;bi++){
    if(_rd_iscale[bi]<_rd_iscale[bi_min]){
      bi_min2=bi_min;
      bi_min=bi;
    }
    else if(_rd_iscale[bi]<_rd_iscale[bi_min2])bi_min2=bi;
  }
  /*If the minimum iscale is less than 1.0, use the second smallest instead,
     and force the value to at least 1.0 (inflating chroma is a waste).*/
  if(_rd_iscale[bi_min]<(1<<OC_RD_ISCALE_BITS))bi_min=bi_min2;
  d=OC_MINI(_rd_scale[bi_min],1<<OC_RD_SCALE_BITS);
  _rd_scale[4]=OC_RD_SCALE(d,_chroma_rd_scale[0]);
  d=OC_MAXI(_rd_iscale[bi_min],1<<OC_RD_ISCALE_BITS);
  _rd_iscale[4]=OC_RD_ISCALE(d,_chroma_rd_scale[1]);
  return activity_sum;
}

static int oc_mb_intra_satd(oc_enc_ctx *_enc,unsigned _mbi,
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
  unsigned               luma;
  int                    dc;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[0];
  luma=0;
  for(bi=0;bi<4;bi++){
    fragi=sb_map[bi];
    frag_offs=frag_buf_offs[fragi];
    _frag_satd[bi]=oc_enc_frag_intra_satd(_enc,&dc,src+frag_offs,ystride);
    luma+=dc;
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
    _frag_satd[mapii]=oc_enc_frag_intra_satd(_enc,&dc,src+frag_offs,ystride);
  }
  return luma;
}

/*Select luma block-level quantizers for a MB in an INTRA frame.*/
static unsigned oc_analyze_intra_mb_luma(oc_enc_ctx *_enc,
 const oc_qii_state *_qs,unsigned _mbi,const unsigned _rd_scale[4]){
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
  int                  dc;
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
  if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
    satd=oc_enc_frag_intra_satd(_enc,&dc,src+frag_offs,ystride);
  }
  else{
    satd=oc_enc_frag_intra_sad(_enc,src+frag_offs,ystride);
  }
  nqis=_enc->state.nqis;
  lambda=_enc->lambda;
  for(qii=0;qii<nqis;qii++){
    oc_qii_state_advance(qs[0]+qii,_qs,qii);
    rate[0][qii]=oc_dct_cost2(_enc,ssd[0]+qii,qii,0,0,satd)
     +(qs[0][qii].bits-_qs->bits<<OC_BIT_SCALE);
    ssd[0][qii]=OC_RD_SCALE(ssd[0][qii],_rd_scale[0]);
    cost[0][qii]=OC_MODE_RD_COST(ssd[0][qii],rate[0][qii],lambda);
  }
  for(bi=1;bi<4;bi++){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frag_offs=frag_buf_offs[fragi];
    if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
      satd=oc_enc_frag_intra_satd(_enc,&dc,src+frag_offs,ystride);
    }
    else{
      satd=oc_enc_frag_intra_sad(_enc,src+frag_offs,ystride);
    }
    for(qii=0;qii<nqis;qii++){
      oc_qii_state qt[3];
      unsigned     cur_ssd;
      unsigned     cur_rate;
      int          best_qij;
      int          qij;
      oc_qii_state_advance(qt+0,qs[bi-1]+0,qii);
      cur_rate=oc_dct_cost2(_enc,&cur_ssd,qii,0,0,satd);
      cur_ssd=OC_RD_SCALE(cur_ssd,_rd_scale[bi]);
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
 const oc_qii_state *_qs,int _pli,ptrdiff_t _fragi,unsigned _rd_scale){
  const unsigned char *src;
  oc_fragment         *frags;
  ptrdiff_t            frag_offs;
  oc_qii_state         qt[3];
  unsigned             cost[3];
  unsigned             satd;
  int                  dc;
  unsigned             best_cost;
  int                  best_qii;
  int                  qii;
  int                  lambda;
  int                  ystride;
  int                  nqis;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ystride=_enc->state.ref_ystride[_pli];
  frag_offs=_enc->state.frag_buf_offs[_fragi];
  if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
    satd=oc_enc_frag_intra_satd(_enc,&dc,src+frag_offs,ystride);
  }
  else{
    satd=oc_enc_frag_intra_sad(_enc,src+frag_offs,ystride);
  }
  /*Most chroma blocks have no AC coefficients to speak of anyway, so it's not
     worth spending the bits to change the AC quantizer.
    TODO: This may be worth revisiting when we separate out DC and AC
     predictions from SATD.*/
#if 0
  nqis=_enc->state.nqis;
#else
  nqis=1;
#endif
  lambda=_enc->lambda;
  best_qii=0;
  for(qii=0;qii<nqis;qii++){
    unsigned cur_rate;
    unsigned cur_ssd;
    oc_qii_state_advance(qt+qii,_qs,qii);
    cur_rate=oc_dct_cost2(_enc,&cur_ssd,qii,_pli,0,satd)
     +(qt[qii].bits-_qs->bits<<OC_BIT_SCALE);
    cur_ssd=OC_RD_SCALE(cur_ssd,_rd_scale);
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

static void oc_enc_mb_transform_quantize_intra_luma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,unsigned _mbi,
 const unsigned _rd_scale[4],const unsigned _rd_iscale[4]){
  /*Worst case token stack usage for 4 fragments.*/
  oc_token_checkpoint  stack[64*4];
  oc_token_checkpoint *stackptr;
  const oc_sb_map     *sb_maps;
  oc_fragment         *frags;
  ptrdiff_t           *coded_fragis;
  ptrdiff_t            ncoded_fragis;
  ptrdiff_t            fragi;
  int                  bi;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  frags=_enc->state.frags;
  coded_fragis=_pipe->coded_fragis[0];
  ncoded_fragis=_pipe->ncoded_fragis[0];
  stackptr=stack;
  for(bi=0;bi<4;bi++){
    fragi=sb_maps[_mbi>>2][_mbi&3][bi];
    frags[fragi].refi=OC_FRAME_SELF;
    frags[fragi].mb_mode=OC_MODE_INTRA;
    oc_enc_block_transform_quantize(_enc,_pipe,0,fragi,
     _rd_scale[bi],_rd_iscale[bi],NULL,NULL,&stackptr);
    coded_fragis[ncoded_fragis++]=fragi;
  }
  _pipe->ncoded_fragis[0]=ncoded_fragis;
}

static void oc_enc_sb_transform_quantize_intra_chroma(oc_enc_ctx *_enc,
 oc_enc_pipeline_state *_pipe,int _pli,int _sbi_start,int _sbi_end){
  const ogg_uint16_t *mcu_rd_scale;
  const ogg_uint16_t *mcu_rd_iscale;
  const oc_sb_map    *sb_maps;
  ptrdiff_t          *coded_fragis;
  ptrdiff_t           ncoded_fragis;
  ptrdiff_t           froffset;
  int                 sbi;
  mcu_rd_scale=(const ogg_uint16_t *)_enc->mcu_rd_scale;
  mcu_rd_iscale=(const ogg_uint16_t *)_enc->mcu_rd_iscale;
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  coded_fragis=_pipe->coded_fragis[_pli];
  ncoded_fragis=_pipe->ncoded_fragis[_pli];
  froffset=_pipe->froffset[_pli];
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
        unsigned             rd_scale;
        unsigned             rd_iscale;
        rd_scale=mcu_rd_scale[fragi-froffset];
        rd_iscale=mcu_rd_iscale[fragi-froffset];
        oc_analyze_intra_chroma_block(_enc,_pipe->qs+_pli,_pli,fragi,rd_scale);
        stackptr=stack;
        oc_enc_block_transform_quantize(_enc,_pipe,_pli,fragi,
         rd_scale,rd_iscale,NULL,NULL,&stackptr);
        coded_fragis[ncoded_fragis++]=fragi;
      }
    }
  }
  _pipe->ncoded_fragis[_pli]=ncoded_fragis;
}

/*Analysis stage for an INTRA frame.*/
void oc_enc_analyze_intra(oc_enc_ctx *_enc,int _recode){
  ogg_int64_t             activity_sum;
  ogg_int64_t             luma_sum;
  unsigned                activity_avg;
  unsigned                luma_avg;
  const ogg_uint16_t     *chroma_rd_scale;
  ogg_uint16_t           *mcu_rd_scale;
  ogg_uint16_t           *mcu_rd_iscale;
  const unsigned char    *map_idxs;
  int                     nmap_idxs;
  oc_sb_flags            *sb_flags;
  signed char            *mb_modes;
  const oc_mb_map        *mb_maps;
  const oc_sb_map        *sb_maps;
  oc_fragment            *frags;
  unsigned                stripe_sby;
  unsigned                mcu_nvsbs;
  int                     notstart;
  int                     notdone;
  int                     refi;
  int                     pli;
  _enc->state.frame_type=OC_INTRA_FRAME;
  oc_enc_tokenize_start(_enc);
  oc_enc_pipeline_init(_enc,&_enc->pipe);
  oc_enc_mode_rd_init(_enc);
  activity_sum=luma_sum=0;
  activity_avg=_enc->activity_avg;
  luma_avg=OC_CLAMPI(90<<8,_enc->luma_avg,160<<8);
  chroma_rd_scale=_enc->chroma_rd_scale[OC_INTRA_FRAME][_enc->state.qis[0]];
  mcu_rd_scale=_enc->mcu_rd_scale;
  mcu_rd_iscale=_enc->mcu_rd_iscale;
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
  sb_maps=(const oc_sb_map *)_enc->state.sb_maps;
  frags=_enc->state.frags;
  notstart=0;
  notdone=1;
  mcu_nvsbs=_enc->mcu_nvsbs;
  for(stripe_sby=0;notdone;stripe_sby+=mcu_nvsbs){
    ptrdiff_t cfroffset;
    unsigned  sbi;
    unsigned  sbi_end;
    notdone=oc_enc_pipeline_set_stripe(_enc,&_enc->pipe,stripe_sby);
    sbi_end=_enc->pipe.sbi_end[0];
    cfroffset=_enc->pipe.froffset[1];
    for(sbi=_enc->pipe.sbi0[0];sbi<sbi_end;sbi++){
      int quadi;
      /*Mode addressing is through Y plane, always 4 MB per SB.*/
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        unsigned  activity[4];
        unsigned  rd_scale[5];
        unsigned  rd_iscale[5];
        unsigned  luma;
        unsigned  mbi;
        int       mapii;
        int       mapi;
        int       bi;
        ptrdiff_t fragi;
        mbi=sbi<<2|quadi;
        /*Activity masking.*/
        if(_enc->sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
          luma=oc_mb_activity(_enc,mbi,activity);
        }
        else{
          unsigned intra_satd[12];
          luma=oc_mb_intra_satd(_enc,mbi,intra_satd);
          oc_mb_activity_fast(_enc,mbi,activity,intra_satd);
          for(bi=0;bi<4;bi++)frags[sb_maps[mbi>>2][mbi&3][bi]].qii=0;
        }
        activity_sum+=oc_mb_masking(rd_scale,rd_iscale,
         chroma_rd_scale,activity,activity_avg,luma,luma_avg);
        luma_sum+=luma;
        /*Motion estimation:
          We do a basic 1MV search for all macroblocks, coded or not,
           keyframe or not, unless we aren't using motion estimation at all.*/
        if(!_recode&&_enc->state.curframe_num>0&&
         _enc->sp_level<OC_SP_LEVEL_NOMC&&_enc->keyframe_frequency_force>1){
          oc_mcenc_search(_enc,mbi);
        }
        if(_enc->sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
          oc_analyze_intra_mb_luma(_enc,_enc->pipe.qs+0,mbi,rd_scale);
        }
        mb_modes[mbi]=OC_MODE_INTRA;
        oc_enc_mb_transform_quantize_intra_luma(_enc,&_enc->pipe,
         mbi,rd_scale,rd_iscale);
        /*Propagate final MB mode and MVs to the chroma blocks.*/
        for(mapii=4;mapii<nmap_idxs;mapii++){
          mapi=map_idxs[mapii];
          pli=mapi>>2;
          bi=mapi&3;
          fragi=mb_maps[mbi][pli][bi];
          frags[fragi].refi=OC_FRAME_SELF;
          frags[fragi].mb_mode=OC_MODE_INTRA;
        }
        /*Save masking scale factors for chroma blocks.*/
        for(mapii=4;mapii<(nmap_idxs-4>>1)+4;mapii++){
          mapi=map_idxs[mapii];
          bi=mapi&3;
          fragi=mb_maps[mbi][1][bi];
          mcu_rd_scale[fragi-cfroffset]=(ogg_uint16_t)rd_scale[4];
          mcu_rd_iscale[fragi-cfroffset]=(ogg_uint16_t)rd_iscale[4];
        }
      }
    }
    oc_enc_pipeline_finish_mcu_plane(_enc,&_enc->pipe,0,notstart,notdone);
    /*Code chroma planes.*/
    for(pli=1;pli<3;pli++){
      oc_enc_sb_transform_quantize_intra_chroma(_enc,&_enc->pipe,
       pli,_enc->pipe.sbi0[pli],_enc->pipe.sbi_end[pli]);
      oc_enc_pipeline_finish_mcu_plane(_enc,&_enc->pipe,pli,notstart,notdone);
    }
    notstart=1;
  }
  /*Compute the average block activity and MB luma score for the frame.*/
  _enc->activity_avg=OC_MAXI(OC_ACTIVITY_AVG_MIN,
   (unsigned)((activity_sum+(_enc->state.fplanes[0].nfrags>>1))/
   _enc->state.fplanes[0].nfrags));
  _enc->luma_avg=(unsigned)((luma_sum+(_enc->state.nmbs>>1))/_enc->state.nmbs);
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
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12],
 const unsigned _rd_scale[4],int _qti){
  oc_fr_state  fr;
  oc_qii_state qs;
  unsigned     ssd;
  unsigned     rate;
  unsigned     satd;
  unsigned     best_ssd;
  unsigned     best_rate;
  int          best_fri;
  int          best_qii;
  int          lambda;
  int          nqis;
  int          nskipped;
  int          bi;
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
  ssd=rate=nskipped=0;
  for(bi=0;bi<4;bi++){
    oc_fr_state  ft[2];
    oc_qii_state qt[3];
    unsigned     best_cost;
    unsigned     cur_cost;
    unsigned     cur_ssd;
    unsigned     cur_rate;
    unsigned     cur_overhead;
    int          qii;
    satd=_frag_satd[bi];
    *(ft+0)=*&fr;
    oc_fr_code_block(ft+0);
    cur_overhead=ft[0].bits-fr.bits;
    best_rate=oc_dct_cost2(_enc,&best_ssd,0,0,_qti,satd)
     +(cur_overhead<<OC_BIT_SCALE);
    if(nqis>1){
      oc_qii_state_advance(qt+0,&qs,0);
      best_rate+=qt[0].bits-qs.bits<<OC_BIT_SCALE;
    }
    best_ssd=OC_RD_SCALE(best_ssd,_rd_scale[bi]);
    best_cost=OC_MODE_RD_COST(ssd+best_ssd,rate+best_rate,lambda);
    best_fri=0;
    best_qii=0;
    for(qii=1;qii<nqis;qii++){
      oc_qii_state_advance(qt+qii,&qs,qii);
      cur_rate=oc_dct_cost2(_enc,&cur_ssd,qii,0,_qti,satd)
       +(cur_overhead+qt[qii].bits-qs.bits<<OC_BIT_SCALE);
      cur_ssd=OC_RD_SCALE(cur_ssd,_rd_scale[bi]);
      cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_rate,lambda);
      if(cur_cost<best_cost){
        best_cost=cur_cost;
        best_ssd=cur_ssd;
        best_rate=cur_rate;
        best_qii=qii;
      }
    }
    if(_skip_ssd[bi]<(UINT_MAX>>OC_BIT_SCALE+2)&&nskipped<3){
      *(ft+1)=*&fr;
      oc_fr_skip_block(ft+1);
      cur_overhead=ft[1].bits-fr.bits<<OC_BIT_SCALE;
      cur_ssd=_skip_ssd[bi]<<OC_BIT_SCALE;
      cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_overhead,lambda);
      if(cur_cost<=best_cost){
        best_ssd=cur_ssd;
        best_rate=cur_overhead;
        best_fri=1;
        best_qii+=4;
      }
    }
    rate+=best_rate;
    ssd+=best_ssd;
    *&fr=*(ft+best_fri);
    if(best_fri==0)*&qs=*(qt+best_qii);
    else nskipped++;
    _modec->qii[bi]=best_qii;
  }
  _modec->ssd=ssd;
  _modec->rate=rate;
}

static void oc_analyze_mb_mode_chroma(oc_enc_ctx *_enc,
 oc_mode_choice *_modec,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12],
 unsigned _rd_scale,int _qti){
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
  /*Most chroma blocks have no AC coefficients to speak of anyway, so it's not
     worth spending the bits to change the AC quantizer.
    TODO: This may be worth revisiting when we separate out DC and AC
     predictions from SATD.*/
#if 0
  nqis=_enc->state.nqis;
#else
  nqis=1;
#endif
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
      best_rate=oc_dct_cost2(_enc,&best_ssd,0,pli,_qti,satd)
       +OC_CHROMA_QII_RATE;
      best_ssd=OC_RD_SCALE(best_ssd,_rd_scale);
      best_cost=OC_MODE_RD_COST(ssd+best_ssd,rate+best_rate,lambda);
      best_qii=0;
      for(qii=1;qii<nqis;qii++){
        cur_rate=oc_dct_cost2(_enc,&cur_ssd,qii,pli,_qti,satd)
         +OC_CHROMA_QII_RATE;
        cur_ssd=OC_RD_SCALE(cur_ssd,_rd_scale);
        cur_cost=OC_MODE_RD_COST(ssd+cur_ssd,rate+cur_rate,lambda);
        if(cur_cost<best_cost){
          best_cost=cur_cost;
          best_ssd=cur_ssd;
          best_rate=cur_rate;
          best_qii=qii;
        }
      }
      if(_skip_ssd[bi]<(UINT_MAX>>OC_BIT_SCALE+2)){
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
 unsigned _mbi,const unsigned _rd_scale[4],unsigned _ssd[12]){
  const unsigned char   *src;
  const unsigned char   *ref;
  int                    ystride;
  const oc_fragment     *frags;
  const ptrdiff_t       *frag_buf_offs;
  const ptrdiff_t       *sb_map;
  const oc_mb_map_plane *mb_map;
  const unsigned char   *map_idxs;
  oc_mv                 *mvs;
  int                    map_nidxs;
  unsigned               uncoded_ssd;
  int                    mapii;
  int                    mapi;
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  int                    borderi;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[OC_FRAME_PREV];
  ystride=_enc->state.ref_ystride[0];
  frags=_enc->state.frags;
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  mvs=_enc->mb_info[_mbi].block_mv;
  for(bi=0;bi<4;bi++){
    fragi=sb_map[bi];
    borderi=frags[fragi].borderi;
    frag_offs=frag_buf_offs[fragi];
    if(borderi<0){
      uncoded_ssd=oc_enc_frag_ssd(_enc,src+frag_offs,ref+frag_offs,ystride);
    }
    else{
      uncoded_ssd=oc_enc_frag_border_ssd(_enc,
       src+frag_offs,ref+frag_offs,ystride,_enc->state.borders[borderi].mask);
    }
    /*Scale to match DCT domain and RD.*/
    uncoded_ssd=OC_RD_SKIP_SCALE(uncoded_ssd,_rd_scale[bi]);
    /*Motion is a special case; if there is more than a full-pixel motion
       against the prior frame, penalize skipping.
      TODO: The factor of two here is a kludge, but it tested out better than a
       hard limit.*/
    if(mvs[bi]!=0)uncoded_ssd*=2;
    _pipe->skip_ssd[0][fragi-_pipe->froffset[0]]=_ssd[bi]=uncoded_ssd;
  }
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=(map_nidxs-4>>1)+4;
  mapii=4;
  mvs=_enc->mb_info[_mbi].unref_mv;
  for(pli=1;pli<3;pli++){
    ystride=_enc->state.ref_ystride[pli];
    for(;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      borderi=frags[fragi].borderi;
      frag_offs=frag_buf_offs[fragi];
      if(borderi<0){
        uncoded_ssd=oc_enc_frag_ssd(_enc,src+frag_offs,ref+frag_offs,ystride);
      }
      else{
        uncoded_ssd=oc_enc_frag_border_ssd(_enc,
         src+frag_offs,ref+frag_offs,ystride,_enc->state.borders[borderi].mask);
      }
      /*Scale to match DCT domain and RD.*/
      uncoded_ssd=OC_RD_SKIP_SCALE(uncoded_ssd,_rd_scale[4]);
      /*Motion is a special case; if there is more than a full-pixel motion
         against the prior frame, penalize skipping.
        TODO: The factor of two here is a kludge, but it tested out better than
         a hard limit*/
      if(mvs[OC_FRAME_PREV]!=0)uncoded_ssd*=2;
      _pipe->skip_ssd[pli][fragi-_pipe->froffset[pli]]=_ssd[mapii]=uncoded_ssd;
    }
    map_nidxs=(map_nidxs-4<<1)+4;
  }
}


static void oc_cost_intra(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _frag_satd[12],const unsigned _skip_ssd[12],
 const unsigned _rd_scale[5]){
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,_frag_satd,_skip_ssd,_rd_scale,0);
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,
   _frag_satd,_skip_ssd,_rd_scale[4],0);
  _modec->overhead=
   oc_mode_scheme_chooser_cost(&_enc->chooser,OC_MODE_INTRA)<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

static void oc_cost_inter(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,oc_mv _mv,
 const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _skip_ssd[12],const unsigned _rd_scale[5]){
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
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  int                    dc;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[OC_FRAME_FOR_MODE(_mb_mode)];
  ystride=_enc->state.ref_ystride[0];
  frag_buf_offs=_enc->state.frag_buf_offs;
  sb_map=_enc->state.sb_maps[_mbi>>2][_mbi&3];
  _modec->rate=_modec->ssd=0;
  if(oc_state_get_mv_offsets(&_enc->state,mv_offs,0,_mv)>1){
    for(bi=0;bi<4;bi++){
      fragi=sb_map[bi];
      frag_offs=frag_buf_offs[fragi];
      if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
        frag_satd[bi]=oc_enc_frag_satd2(_enc,&dc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride);
        frag_satd[bi]+=abs(dc);
      }
      else{
        frag_satd[bi]=oc_enc_frag_sad2_thresh(_enc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
      }
    }
  }
  else{
    for(bi=0;bi<4;bi++){
      fragi=sb_map[bi];
      frag_offs=frag_buf_offs[fragi];
      if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
        frag_satd[bi]=oc_enc_frag_satd(_enc,&dc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ystride);
        frag_satd[bi]+=abs(dc);
      }
      else{
        frag_satd[bi]=oc_enc_frag_sad(_enc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ystride);
      }
    }
  }
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  map_nidxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
  /*Note: This assumes ref_ystride[1]==ref_ystride[2].*/
  ystride=_enc->state.ref_ystride[1];
  if(oc_state_get_mv_offsets(&_enc->state,mv_offs,1,_mv)>1){
    for(mapii=4;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      pli=mapi>>2;
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      frag_offs=frag_buf_offs[fragi];
      if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
        frag_satd[mapii]=oc_enc_frag_satd2(_enc,&dc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride);
        frag_satd[mapii]+=abs(dc);
      }
      else{
        frag_satd[mapii]=oc_enc_frag_sad2_thresh(_enc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride,UINT_MAX);
      }
    }
  }
  else{
    for(mapii=4;mapii<map_nidxs;mapii++){
      mapi=map_idxs[mapii];
      pli=mapi>>2;
      bi=mapi&3;
      fragi=mb_map[pli][bi];
      frag_offs=frag_buf_offs[fragi];
      if(_enc->sp_level<OC_SP_LEVEL_NOSATD){
        frag_satd[mapii]=oc_enc_frag_satd(_enc,&dc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ystride);
        frag_satd[mapii]+=abs(dc);
      }
      else{
        frag_satd[mapii]=oc_enc_frag_sad(_enc,src+frag_offs,
         ref+frag_offs+mv_offs[0],ystride);
      }
    }
  }
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,frag_satd,_skip_ssd,_rd_scale,1);
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,
   frag_satd,_skip_ssd,_rd_scale[4],1);
  _modec->overhead=
   oc_mode_scheme_chooser_cost(&_enc->chooser,_mb_mode)<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

static void oc_cost_inter_nomv(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,const oc_fr_state *_fr,const oc_qii_state *_qs,
 const unsigned _skip_ssd[12],const unsigned _rd_scale[5]){
  oc_cost_inter(_enc,_modec,_mbi,_mb_mode,0,_fr,_qs,_skip_ssd,_rd_scale);
}

static int oc_cost_inter1mv(oc_enc_ctx *_enc,oc_mode_choice *_modec,
 unsigned _mbi,int _mb_mode,oc_mv _mv,
 const oc_fr_state *_fr,const oc_qii_state *_qs,const unsigned _skip_ssd[12],
 const unsigned _rd_scale[5]){
  int bits0;
  oc_cost_inter(_enc,_modec,_mbi,_mb_mode,_mv,_fr,_qs,_skip_ssd,_rd_scale);
  bits0=OC_MV_BITS[0][OC_MV_X(_mv)+31]+OC_MV_BITS[0][OC_MV_Y(_mv)+31];
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
 const unsigned _skip_ssd[12],const unsigned _rd_scale[5]){
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
  int                    pli;
  int                    bi;
  ptrdiff_t              fragi;
  ptrdiff_t              frag_offs;
  int                    bits0;
  int                    bits1;
  unsigned               satd;
  int                    dc;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[OC_FRAME_PREV];
  ystride=_enc->state.ref_ystride[0];
  frag_buf_offs=_enc->state.frag_buf_offs;
  frag_mvs=_enc->state.frag_mvs;
  mb_map=(const oc_mb_map_plane *)_enc->state.mb_maps[_mbi];
  _modec->rate=_modec->ssd=0;
  for(bi=0;bi<4;bi++){
    fragi=mb_map[0][bi];
    /*Save the block MVs as the current ones while we're here; we'll replace
       them if we don't ultimately choose 4MV mode.*/
    frag_mvs[fragi]=_mv[bi];
    frag_offs=frag_buf_offs[fragi];
    if(oc_state_get_mv_offsets(&_enc->state,mv_offs,0,_mv[bi])>1){
      satd=oc_enc_frag_satd2(_enc,&dc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride);
    }
    else{
      satd=oc_enc_frag_satd(_enc,&dc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride);
    }
    frag_satd[OC_MB_PHASE[_mbi&3][bi]]=satd+abs(dc);
  }
  oc_analyze_mb_mode_luma(_enc,_modec,_fr,_qs,frag_satd,
   _enc->vp3_compatible?OC_NOSKIP:_skip_ssd,_rd_scale,1);
  /*Figure out which blocks are being skipped and give them (0,0) MVs.*/
  bits0=0;
  bits1=0;
  nqis=_enc->state.nqis;
  for(bi=0;bi<4;bi++){
    if(_modec->qii[OC_MB_PHASE[_mbi&3][bi]]>=nqis)lbmvs[bi]=0;
    else{
      lbmvs[bi]=_mv[bi];
      bits0+=OC_MV_BITS[0][OC_MV_X(_mv[bi])+31]
       +OC_MV_BITS[0][OC_MV_Y(_mv[bi])+31];
      bits1+=12;
    }
  }
  (*OC_SET_CHROMA_MVS_TABLE[_enc->state.info.pixel_fmt])(cbmvs,lbmvs);
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
    /*TODO: We could save half these calls by re-using the results for the Cb
       and Cr planes; is it worth it?*/
    if(oc_state_get_mv_offsets(&_enc->state,mv_offs,pli,cbmvs[bi])>1){
      satd=oc_enc_frag_satd2(_enc,&dc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ref+frag_offs+mv_offs[1],ystride);
    }
    else{
      satd=oc_enc_frag_satd(_enc,&dc,src+frag_offs,
       ref+frag_offs+mv_offs[0],ystride);
    }
    frag_satd[mapii]=satd+abs(dc);
  }
  oc_analyze_mb_mode_chroma(_enc,_modec,_fr,_qs,
   frag_satd,_skip_ssd,_rd_scale[4],1);
  _modec->overhead=
   oc_mode_scheme_chooser_cost(&_enc->chooser,OC_MODE_INTER_MV_FOUR)
   +OC_MINI(_enc->mv_bits[0]+bits0,_enc->mv_bits[1]+bits1)
   -OC_MINI(_enc->mv_bits[0],_enc->mv_bits[1])<<OC_BIT_SCALE;
  oc_mode_set_cost(_modec,_enc->lambda);
}

int oc_enc_analyze_inter(oc_enc_ctx *_enc,int _allow_keyframe,int _recode){
  oc_set_chroma_mvs_func  set_chroma_mvs;
  oc_qii_state            intra_luma_qs;
  oc_mv                   last_mv;
  oc_mv                   prior_mv;
  ogg_int64_t             interbits;
  ogg_int64_t             intrabits;
  ogg_int64_t             activity_sum;
  ogg_int64_t             luma_sum;
  unsigned                activity_avg;
  unsigned                luma_avg;
  const ogg_uint16_t     *chroma_rd_scale;
  ogg_uint16_t           *mcu_rd_scale;
  ogg_uint16_t           *mcu_rd_iscale;
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
  unsigned                stripe_sby;
  unsigned                mcu_nvsbs;
  int                     notstart;
  int                     notdone;
  unsigned                sbi;
  unsigned                sbi_end;
  int                     refi;
  int                     pli;
  int                     sp_level;
  sp_level=_enc->sp_level;
  set_chroma_mvs=OC_SET_CHROMA_MVS_TABLE[_enc->state.info.pixel_fmt];
  _enc->state.frame_type=OC_INTER_FRAME;
  oc_mode_scheme_chooser_reset(&_enc->chooser);
  oc_enc_tokenize_start(_enc);
  oc_enc_pipeline_init(_enc,&_enc->pipe);
  oc_enc_mode_rd_init(_enc);
  if(_allow_keyframe)oc_qii_state_init(&intra_luma_qs);
  _enc->mv_bits[0]=_enc->mv_bits[1]=0;
  interbits=intrabits=0;
  activity_sum=luma_sum=0;
  activity_avg=_enc->activity_avg;
  luma_avg=OC_CLAMPI(90<<8,_enc->luma_avg,160<<8);
  chroma_rd_scale=_enc->chroma_rd_scale[OC_INTER_FRAME][_enc->state.qis[0]];
  mcu_rd_scale=_enc->mcu_rd_scale;
  mcu_rd_iscale=_enc->mcu_rd_iscale;
  last_mv=prior_mv=0;
  /*Choose MVs and MB modes and quantize and code luma.
    Must be done in Hilbert order.*/
  map_idxs=OC_MB_MAP_IDXS[_enc->state.info.pixel_fmt];
  nmap_idxs=OC_MB_MAP_NIDXS[_enc->state.info.pixel_fmt];
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
  notstart=0;
  notdone=1;
  mcu_nvsbs=_enc->mcu_nvsbs;
  for(stripe_sby=0;notdone;stripe_sby+=mcu_nvsbs){
    ptrdiff_t cfroffset;
    notdone=oc_enc_pipeline_set_stripe(_enc,&_enc->pipe,stripe_sby);
    sbi_end=_enc->pipe.sbi_end[0];
    cfroffset=_enc->pipe.froffset[1];
    for(sbi=_enc->pipe.sbi0[0];sbi<sbi_end;sbi++){
      int quadi;
      /*Mode addressing is through Y plane, always 4 MB per SB.*/
      for(quadi=0;quadi<4;quadi++)if(sb_flags[sbi].quad_valid&1<<quadi){
        oc_mode_choice modes[8];
        unsigned       activity[4];
        unsigned       rd_scale[5];
        unsigned       rd_iscale[5];
        unsigned       skip_ssd[12];
        unsigned       intra_satd[12];
        unsigned       luma;
        int            mb_mv_bits_0;
        int            mb_gmv_bits_0;
        int            inter_mv_pref;
        int            mb_mode;
        int            refi;
        int            mv;
        unsigned       mbi;
        int            mapii;
        int            mapi;
        int            bi;
        ptrdiff_t      fragi;
        mbi=sbi<<2|quadi;
        luma=oc_mb_intra_satd(_enc,mbi,intra_satd);
        /*Activity masking.*/
        if(sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
          oc_mb_activity(_enc,mbi,activity);
        }
        else oc_mb_activity_fast(_enc,mbi,activity,intra_satd);
        luma_sum+=luma;
        activity_sum+=oc_mb_masking(rd_scale,rd_iscale,
         chroma_rd_scale,activity,activity_avg,luma,luma_avg);
        /*Motion estimation:
          We always do a basic 1MV search for all macroblocks, coded or not,
           keyframe or not.*/
        if(!_recode&&sp_level<OC_SP_LEVEL_NOMC)oc_mcenc_search(_enc,mbi);
        mv=0;
        /*Find the block choice with the lowest estimated coding cost.
          If a Cb or Cr block is coded but no Y' block from a macro block then
           the mode MUST be OC_MODE_INTER_NOMV.
          This is the default state to which the mode data structure is
           initialised in encoder and decoder at the start of each frame.*/
        /*Block coding cost is estimated from correlated SATD metrics.*/
        /*At this point, all blocks that are in frame are still marked coded.*/
        if(!_recode){
          embs[mbi].unref_mv[OC_FRAME_GOLD]=
           embs[mbi].analysis_mv[0][OC_FRAME_GOLD];
          embs[mbi].unref_mv[OC_FRAME_PREV]=
           embs[mbi].analysis_mv[0][OC_FRAME_PREV];
          embs[mbi].refined=0;
        }
        /*Estimate the cost of coding this MB in a keyframe.*/
        if(_allow_keyframe){
          oc_cost_intra(_enc,modes+OC_MODE_INTRA,mbi,
           _enc->pipe.fr+0,&intra_luma_qs,intra_satd,OC_NOSKIP,rd_scale);
          intrabits+=modes[OC_MODE_INTRA].rate;
          for(bi=0;bi<4;bi++){
            oc_qii_state_advance(&intra_luma_qs,&intra_luma_qs,
             modes[OC_MODE_INTRA].qii[bi]);
          }
        }
        /*Estimate the cost in a delta frame for various modes.*/
        oc_skip_cost(_enc,&_enc->pipe,mbi,rd_scale,skip_ssd);
        if(sp_level<OC_SP_LEVEL_NOMC){
          oc_cost_inter_nomv(_enc,modes+OC_MODE_INTER_NOMV,mbi,
           OC_MODE_INTER_NOMV,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
          oc_cost_intra(_enc,modes+OC_MODE_INTRA,mbi,
           _enc->pipe.fr+0,_enc->pipe.qs+0,intra_satd,skip_ssd,rd_scale);
          mb_mv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_INTER_MV,mbi,
           OC_MODE_INTER_MV,embs[mbi].unref_mv[OC_FRAME_PREV],
           _enc->pipe.fr+0,_enc->pipe.qs+0,skip_ssd,rd_scale);
          oc_cost_inter(_enc,modes+OC_MODE_INTER_MV_LAST,mbi,
           OC_MODE_INTER_MV_LAST,last_mv,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
          oc_cost_inter(_enc,modes+OC_MODE_INTER_MV_LAST2,mbi,
           OC_MODE_INTER_MV_LAST2,prior_mv,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
          oc_cost_inter_nomv(_enc,modes+OC_MODE_GOLDEN_NOMV,mbi,
           OC_MODE_GOLDEN_NOMV,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
          mb_gmv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_GOLDEN_MV,mbi,
           OC_MODE_GOLDEN_MV,embs[mbi].unref_mv[OC_FRAME_GOLD],
           _enc->pipe.fr+0,_enc->pipe.qs+0,skip_ssd,rd_scale);
          /*The explicit MV modes (2,6,7) have not yet gone through halfpel
             refinement.
            We choose the explicit MV mode that's already furthest ahead on
             R-D cost and refine only that one.
            We have to be careful to remember which ones we've refined so that
             we don't refine it again if we re-encode this frame.*/
          inter_mv_pref=_enc->lambda*3;
          if(sp_level<OC_SP_LEVEL_FAST_ANALYSIS){
            oc_cost_inter4mv(_enc,modes+OC_MODE_INTER_MV_FOUR,mbi,
             embs[mbi].block_mv,_enc->pipe.fr+0,_enc->pipe.qs+0,
             skip_ssd,rd_scale);
          }
          else{
            modes[OC_MODE_INTER_MV_FOUR].cost=UINT_MAX;
          }
          if(modes[OC_MODE_INTER_MV_FOUR].cost<modes[OC_MODE_INTER_MV].cost&&
           modes[OC_MODE_INTER_MV_FOUR].cost<modes[OC_MODE_GOLDEN_MV].cost){
            if(!(embs[mbi].refined&0x80)){
              oc_mcenc_refine4mv(_enc,mbi);
              embs[mbi].refined|=0x80;
            }
            oc_cost_inter4mv(_enc,modes+OC_MODE_INTER_MV_FOUR,mbi,
             embs[mbi].ref_mv,_enc->pipe.fr+0,_enc->pipe.qs+0,
             skip_ssd,rd_scale);
          }
          else if(modes[OC_MODE_GOLDEN_MV].cost+inter_mv_pref<
           modes[OC_MODE_INTER_MV].cost){
            if(!(embs[mbi].refined&0x40)){
              oc_mcenc_refine1mv(_enc,mbi,OC_FRAME_GOLD);
              embs[mbi].refined|=0x40;
            }
            mb_gmv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_GOLDEN_MV,mbi,
             OC_MODE_GOLDEN_MV,embs[mbi].analysis_mv[0][OC_FRAME_GOLD],
             _enc->pipe.fr+0,_enc->pipe.qs+0,skip_ssd,rd_scale);
          }
          if(!(embs[mbi].refined&0x04)){
            oc_mcenc_refine1mv(_enc,mbi,OC_FRAME_PREV);
            embs[mbi].refined|=0x04;
          }
          mb_mv_bits_0=oc_cost_inter1mv(_enc,modes+OC_MODE_INTER_MV,mbi,
           OC_MODE_INTER_MV,embs[mbi].analysis_mv[0][OC_FRAME_PREV],
           _enc->pipe.fr+0,_enc->pipe.qs+0,skip_ssd,rd_scale);
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
          oc_cost_inter_nomv(_enc,modes+OC_MODE_INTER_NOMV,mbi,
           OC_MODE_INTER_NOMV,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
          oc_cost_intra(_enc,modes+OC_MODE_INTRA,mbi,
           _enc->pipe.fr+0,_enc->pipe.qs+0,intra_satd,skip_ssd,rd_scale);
          oc_cost_inter_nomv(_enc,modes+OC_MODE_GOLDEN_NOMV,mbi,
           OC_MODE_GOLDEN_NOMV,_enc->pipe.fr+0,_enc->pipe.qs+0,
           skip_ssd,rd_scale);
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
              mv=embs[mbi].analysis_mv[0][OC_FRAME_PREV];
            }break;
            case OC_MODE_INTER_MV_LAST:mv=last_mv;break;
            case OC_MODE_INTER_MV_LAST2:mv=prior_mv;break;
            case OC_MODE_GOLDEN_MV:{
              mv=embs[mbi].analysis_mv[0][OC_FRAME_GOLD];
            }break;
          }
          for(bi=0;bi<4;bi++){
            fragi=mb_maps[mbi][0][bi];
            frag_mvs[fragi]=mv;
          }
        }
        for(bi=0;bi<4;bi++){
          fragi=sb_maps[mbi>>2][mbi&3][bi];
          frags[fragi].qii=modes[mb_mode].qii[bi];
        }
        if(oc_enc_mb_transform_quantize_inter_luma(_enc,&_enc->pipe,mbi,
         modes[mb_mode].overhead>>OC_BIT_SCALE,rd_scale,rd_iscale)>0){
          int orig_mb_mode;
          orig_mb_mode=mb_mode;
          mb_mode=mb_modes[mbi];
          refi=OC_FRAME_FOR_MODE(mb_mode);
          switch(mb_mode){
            case OC_MODE_INTER_MV:{
              prior_mv=last_mv;
              /*If we're backing out from 4MV, find the MV we're actually
                 using.*/
              if(orig_mb_mode==OC_MODE_INTER_MV_FOUR){
                for(bi=0;;bi++){
                  fragi=mb_maps[mbi][0][bi];
                  if(frags[fragi].coded){
                    mv=last_mv=frag_mvs[fragi];
                    break;
                  }
                }
                mb_mv_bits_0=OC_MV_BITS[0][OC_MV_X(mv)+31]
                 +OC_MV_BITS[0][OC_MV_Y(mv)+31];
              }
              /*Otherwise we used the original analysis MV.*/
              else last_mv=embs[mbi].analysis_mv[0][OC_FRAME_PREV];
              _enc->mv_bits[0]+=mb_mv_bits_0;
              _enc->mv_bits[1]+=12;
            }break;
            case OC_MODE_INTER_MV_LAST2:{
              oc_mv tmp_mv;
              tmp_mv=prior_mv;
              prior_mv=last_mv;
              last_mv=tmp_mv;
            }break;
            case OC_MODE_GOLDEN_MV:{
              _enc->mv_bits[0]+=mb_gmv_bits_0;
              _enc->mv_bits[1]+=12;
            }break;
            case OC_MODE_INTER_MV_FOUR:{
              oc_mv lbmvs[4];
              oc_mv cbmvs[4];
              prior_mv=last_mv;
              for(bi=0;bi<4;bi++){
                fragi=mb_maps[mbi][0][bi];
                if(frags[fragi].coded){
                  lbmvs[bi]=last_mv=frag_mvs[fragi];
                  _enc->mv_bits[0]+=OC_MV_BITS[0][OC_MV_X(last_mv)+31]
                   +OC_MV_BITS[0][OC_MV_Y(last_mv)+31];
                  _enc->mv_bits[1]+=12;
                }
                /*Replace the block MVs for not-coded blocks with (0,0).*/
                else lbmvs[bi]=0;
              }
              (*set_chroma_mvs)(cbmvs,lbmvs);
              for(mapii=4;mapii<nmap_idxs;mapii++){
                mapi=map_idxs[mapii];
                pli=mapi>>2;
                bi=mapi&3;
                fragi=mb_maps[mbi][pli][bi];
                frags[fragi].qii=modes[OC_MODE_INTER_MV_FOUR].qii[mapii];
                frags[fragi].refi=refi;
                frags[fragi].mb_mode=mb_mode;
                frag_mvs[fragi]=cbmvs[bi];
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
          refi=OC_FRAME_PREV;
          mv=0;
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
            /*If we switched from 4MV mode to INTER_MV mode, then the qii
               values won't have been chosen with the right MV, but it's
               probably not worth re-estimating them.*/
            frags[fragi].qii=modes[mb_mode].qii[mapii];
            frags[fragi].refi=refi;
            frags[fragi].mb_mode=mb_mode;
            frag_mvs[fragi]=mv;
          }
        }
        /*Save masking scale factors for chroma blocks.*/
        for(mapii=4;mapii<(nmap_idxs-4>>1)+4;mapii++){
          mapi=map_idxs[mapii];
          bi=mapi&3;
          fragi=mb_maps[mbi][1][bi];
          mcu_rd_scale[fragi-cfroffset]=(ogg_uint16_t)rd_scale[4];
          mcu_rd_iscale[fragi-cfroffset]=(ogg_uint16_t)rd_iscale[4];
        }
      }
      oc_fr_state_flush_sb(_enc->pipe.fr+0);
      sb_flags[sbi].coded_fully=_enc->pipe.fr[0].sb_full;
      sb_flags[sbi].coded_partially=_enc->pipe.fr[0].sb_partial;
    }
    oc_enc_pipeline_finish_mcu_plane(_enc,&_enc->pipe,0,notstart,notdone);
    /*Code chroma planes.*/
    for(pli=1;pli<3;pli++){
      oc_enc_sb_transform_quantize_inter_chroma(_enc,&_enc->pipe,
       pli,_enc->pipe.sbi0[pli],_enc->pipe.sbi_end[pli]);
      oc_enc_pipeline_finish_mcu_plane(_enc,&_enc->pipe,pli,notstart,notdone);
    }
    notstart=1;
  }
  /*Update the average block activity and MB luma score for the frame.
    We could use a Bessel follower here, but fast reaction is probably almost
     always best.*/
  _enc->activity_avg=OC_MAXI(OC_ACTIVITY_AVG_MIN,
   (unsigned)((activity_sum+(_enc->state.fplanes[0].nfrags>>1))/
   _enc->state.fplanes[0].nfrags));
  _enc->luma_avg=(unsigned)((luma_sum+(_enc->state.nmbs>>1))/_enc->state.nmbs);
  /*Finish filling in the reference frame borders.*/
  refi=_enc->state.ref_frame_idx[OC_FRAME_SELF];
  for(pli=0;pli<3;pli++)oc_state_borders_fill_caps(&_enc->state,refi,pli);
  /*Finish adding flagging overhead costs to inter bit counts to determine if
     we should have coded a key frame instead.*/
  if(_allow_keyframe){
    /*Technically the chroma plane counts are over-estimations, because they
       don't account for continuing runs from the luma planes, but the
       inaccuracy is small.
      We don't need to add the luma plane coding flag costs, because they are
       already included in the MB rate estimates.*/
    for(pli=1;pli<3;pli++)interbits+=_enc->pipe.fr[pli].bits<<OC_BIT_SCALE;
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
