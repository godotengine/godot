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

  function:
  last mod: $Id: tokenize.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/
#include <stdlib.h>
#include <string.h>
#include "encint.h"



static int oc_make_eob_token(int _run_count){
  if(_run_count<4)return OC_DCT_EOB1_TOKEN+_run_count-1;
  else{
    int cat;
    cat=OC_ILOGNZ_32(_run_count)-3;
    cat=OC_MINI(cat,3);
    return OC_DCT_REPEAT_RUN0_TOKEN+cat;
  }
}

static int oc_make_eob_token_full(int _run_count,int *_eb){
  if(_run_count<4){
    *_eb=0;
    return OC_DCT_EOB1_TOKEN+_run_count-1;
  }
  else{
    int cat;
    cat=OC_ILOGNZ_32(_run_count)-3;
    cat=OC_MINI(cat,3);
    *_eb=_run_count-OC_BYTE_TABLE32(4,8,16,0,cat);
    return OC_DCT_REPEAT_RUN0_TOKEN+cat;
  }
}

/*Returns the number of blocks ended by an EOB token.*/
static int oc_decode_eob_token(int _token,int _eb){
  return (0x20820C41U>>_token*5&0x1F)+_eb;
}

/*TODO: This is now only used during DCT tokenization, and never for runs; it
   should be simplified.*/
static int oc_make_dct_token_full(int _zzi,int _zzj,int _val,int *_eb){
  int neg;
  int zero_run;
  int token;
  int eb;
  neg=_val<0;
  _val=abs(_val);
  zero_run=_zzj-_zzi;
  if(zero_run>0){
    int adj;
    /*Implement a minor restriction on stack 1 so that we know during DC fixups
       that extending a dctrun token from stack 1 will never overflow.*/
    adj=_zzi!=1;
    if(_val<2&&zero_run<17+adj){
      if(zero_run<6){
        token=OC_DCT_RUN_CAT1A+zero_run-1;
        eb=neg;
      }
      else if(zero_run<10){
        token=OC_DCT_RUN_CAT1B;
        eb=zero_run-6+(neg<<2);
      }
      else{
        token=OC_DCT_RUN_CAT1C;
        eb=zero_run-10+(neg<<3);
      }
    }
    else if(_val<4&&zero_run<3+adj){
      if(zero_run<2){
        token=OC_DCT_RUN_CAT2A;
        eb=_val-2+(neg<<1);
      }
      else{
        token=OC_DCT_RUN_CAT2B;
        eb=zero_run-2+(_val-2<<1)+(neg<<2);
      }
    }
    else{
      if(zero_run<9)token=OC_DCT_SHORT_ZRL_TOKEN;
      else token=OC_DCT_ZRL_TOKEN;
      eb=zero_run-1;
    }
  }
  else if(_val<3){
    token=OC_ONE_TOKEN+(_val-1<<1)+neg;
    eb=0;
  }
  else if(_val<7){
    token=OC_DCT_VAL_CAT2+_val-3;
    eb=neg;
  }
  else if(_val<9){
    token=OC_DCT_VAL_CAT3;
    eb=_val-7+(neg<<1);
  }
  else if(_val<13){
    token=OC_DCT_VAL_CAT4;
    eb=_val-9+(neg<<2);
  }
  else if(_val<21){
    token=OC_DCT_VAL_CAT5;
    eb=_val-13+(neg<<3);
  }
  else if(_val<37){
    token=OC_DCT_VAL_CAT6;
    eb=_val-21+(neg<<4);
  }
  else if(_val<69){
    token=OC_DCT_VAL_CAT7;
    eb=_val-37+(neg<<5);
  }
  else{
    token=OC_DCT_VAL_CAT8;
    eb=_val-69+(neg<<9);
  }
  *_eb=eb;
  return token;
}

/*Token logging to allow a few fragments of efficient rollback.
  Late SKIP analysis is tied up in the tokenization process, so we need to be
   able to undo a fragment's tokens on a whim.*/

static const unsigned char OC_ZZI_HUFF_OFFSET[64]={
   0,16,16,16,16,16,32,32,
  32,32,32,32,32,32,32,48,
  48,48,48,48,48,48,48,48,
  48,48,48,48,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64
};

static int oc_token_bits(oc_enc_ctx *_enc,int _huffi,int _zzi,int _token){
  return _enc->huff_codes[_huffi+OC_ZZI_HUFF_OFFSET[_zzi]][_token].nbits
   +OC_DCT_TOKEN_EXTRA_BITS[_token];
}

static void oc_enc_tokenlog_checkpoint(oc_enc_ctx *_enc,
 oc_token_checkpoint *_cp,int _pli,int _zzi){
  _cp->pli=_pli;
  _cp->zzi=_zzi;
  _cp->eob_run=_enc->eob_run[_pli][_zzi];
  _cp->ndct_tokens=_enc->ndct_tokens[_pli][_zzi];
}

void oc_enc_tokenlog_rollback(oc_enc_ctx *_enc,
 const oc_token_checkpoint *_stack,int _n){
  int i;
  for(i=_n;i-->0;){
    int pli;
    int zzi;
    pli=_stack[i].pli;
    zzi=_stack[i].zzi;
    _enc->eob_run[pli][zzi]=_stack[i].eob_run;
    _enc->ndct_tokens[pli][zzi]=_stack[i].ndct_tokens;
  }
}

static void oc_enc_token_log(oc_enc_ctx *_enc,
 int _pli,int _zzi,int _token,int _eb){
  ptrdiff_t ti;
  ti=_enc->ndct_tokens[_pli][_zzi]++;
  _enc->dct_tokens[_pli][_zzi][ti]=(unsigned char)_token;
  _enc->extra_bits[_pli][_zzi][ti]=(ogg_uint16_t)_eb;
}

static void oc_enc_eob_log(oc_enc_ctx *_enc,
 int _pli,int _zzi,int _run_count){
  int token;
  int eb;
  token=oc_make_eob_token_full(_run_count,&eb);
  oc_enc_token_log(_enc,_pli,_zzi,token,eb);
}


void oc_enc_tokenize_start(oc_enc_ctx *_enc){
  memset(_enc->ndct_tokens,0,sizeof(_enc->ndct_tokens));
  memset(_enc->eob_run,0,sizeof(_enc->eob_run));
  memset(_enc->dct_token_offs,0,sizeof(_enc->dct_token_offs));
  memset(_enc->dc_pred_last,0,sizeof(_enc->dc_pred_last));
}

typedef struct oc_quant_token oc_quant_token;

/*A single node in the Viterbi trellis.
  We maintain up to 2 of these per coefficient:
    - A token to code if the value is zero (EOB, zero run, or combo token).
    - A token to code if the value is not zero (DCT value token).*/
struct oc_quant_token{
  unsigned char next;
  signed char   token;
  ogg_int16_t   eb;
  ogg_uint32_t  cost;
  int           bits;
  int           qc;
};

/*Tokenizes the AC coefficients, possibly adjusting the quantization, and then
   dequantizes and de-zig-zags the result.
  The DC coefficient is not preserved; it should be restored by the caller.*/
int oc_enc_tokenize_ac(oc_enc_ctx *_enc,int _pli,ptrdiff_t _fragi,
 ogg_int16_t *_qdct,const ogg_uint16_t *_dequant,const ogg_int16_t *_dct,
 int _zzi,oc_token_checkpoint **_stack,int _acmin){
  oc_token_checkpoint *stack;
  ogg_int64_t          zflags;
  ogg_int64_t          nzflags;
  ogg_int64_t          best_flags;
  ogg_uint32_t         d2_accum[64];
  oc_quant_token       tokens[64][2];
  ogg_uint16_t        *eob_run;
  const unsigned char *dct_fzig_zag;
  ogg_uint32_t         cost;
  int                  bits;
  int                  eob;
  int                  token;
  int                  eb;
  int                  next;
  int                  huffi;
  int                  zzi;
  int                  ti;
  int                  zzj;
  int                  qc;
  huffi=_enc->huff_idxs[_enc->state.frame_type][1][_pli+1>>1];
  eob_run=_enc->eob_run[_pli];
  memset(tokens[0],0,sizeof(tokens[0]));
  best_flags=nzflags=0;
  zflags=1;
  d2_accum[0]=0;
  zzj=64;
  for(zzi=OC_MINI(_zzi,63);zzi>0;zzi--){
    ogg_int32_t  lambda;
    ogg_uint32_t best_cost;
    int          best_bits=best_bits;
    int          best_next=best_next;
    int          best_token=best_token;
    int          best_eb=best_eb;
    int          best_qc=best_qc;
    int          flush_bits;
    ogg_uint32_t d2;
    int          dq;
    int          e;
    int          c;
    int          s;
    int          tj;
    lambda=_enc->lambda;
    qc=_qdct[zzi];
    s=-(qc<0);
    qc=qc+s^s;
    c=_dct[OC_FZIG_ZAG[zzi]];
    if(qc<=1){
      ogg_uint32_t sum_d2;
      int          nzeros;
      int          dc_reserve;
      /*The hard case: try a zero run.*/
      if(!qc){
        /*Skip runs that are already quantized to zeros.
          If we considered each zero coefficient in turn, we might
           theoretically find a better way to partition long zero runs (e.g.,
           a run of > 17 zeros followed by a 1 might be better coded as a short
           zero run followed by a combo token, rather than the longer zero
           token followed by a 1 value token), but zeros are so common that
           this becomes very computationally expensive (quadratic instead of
           linear in the number of coefficients), for a marginal gain.*/
        while(zzi>1&&!_qdct[zzi-1])zzi--;
        /*The distortion of coefficients originally quantized to zero is
           treated as zero (since we'll never quantize them to anything else).*/
        d2=0;
      }
      else{
        c=c+s^s;
        d2=c*(ogg_int32_t)c;
      }
      eob=eob_run[zzi];
      nzeros=zzj-zzi;
      zzj&=63;
      sum_d2=d2+d2_accum[zzj];
      d2_accum[zzi]=sum_d2;
      flush_bits=eob>0?oc_token_bits(_enc,huffi,zzi,oc_make_eob_token(eob)):0;
      /*We reserve 1 spot for combo run tokens that start in the 1st AC stack
         to ensure they can be extended to include the DC coefficient if
         necessary; this greatly simplifies stack-rewriting later on.*/
      dc_reserve=zzi+62>>6;
      best_cost=0xFFFFFFFF;
      for(;;){
        if(nzflags>>zzj&1){
          int cat;
          int val;
          int val_s;
          int zzk;
          int tk;
          next=tokens[zzj][1].next;
          tk=next&1;
          zzk=next>>1;
          /*Try a pure zero run to this point.*/
          cat=nzeros+55>>6;
          token=OC_DCT_SHORT_ZRL_TOKEN+cat;
          bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
          d2=sum_d2-d2_accum[zzj];
          cost=d2+lambda*bits+tokens[zzj][1].cost;
          if(cost<=best_cost){
            best_next=(zzj<<1)+1;
            best_token=token;
            best_eb=nzeros-1;
            best_cost=cost;
            best_bits=bits+tokens[zzj][1].bits;
            best_qc=0;
          }
          if(nzeros<16+dc_reserve){
            val=_qdct[zzj];
            val_s=-(val<0);
            val=val+val_s^val_s;
            if(val<=2){
              /*Try a +/- 1 combo token.*/
              if(nzeros<6){
                token=OC_DCT_RUN_CAT1A+nzeros-1;
                eb=-val_s;
              }
              else{
                cat=nzeros+54>>6;
                token=OC_DCT_RUN_CAT1B+cat;
                eb=(-val_s<<cat+2)+nzeros-6-(cat<<2);
              }
              e=(_dct[OC_FZIG_ZAG[zzj]]+val_s^val_s)-_dequant[zzj];
              d2=e*(ogg_int32_t)e+sum_d2-d2_accum[zzj];
              bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
              cost=d2+lambda*bits+tokens[zzk][tk].cost;
              if(cost<=best_cost){
                best_next=next;
                best_token=token;
                best_eb=eb;
                best_cost=cost;
                best_bits=bits+tokens[zzk][tk].bits;
                best_qc=1+val_s^val_s;
              }
            }
            if(nzeros<2+dc_reserve&&2<=val&&val<=4){
              /*Try a +/- 2/3 combo token.*/
              cat=nzeros>>1;
              token=OC_DCT_RUN_CAT2A+cat;
              bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
              val=2+((val+val_s^val_s)>2);
              e=(_dct[OC_FZIG_ZAG[zzj]]+val_s^val_s)-_dequant[zzj]*val;
              d2=e*(ogg_int32_t)e+sum_d2-d2_accum[zzj];
              cost=d2+lambda*bits+tokens[zzk][tk].cost;
              if(cost<=best_cost){
                best_cost=cost;
                best_bits=bits+tokens[zzk][tk].bits;
                best_next=next;
                best_token=token;
                best_eb=(-val_s<<1+cat)+(val-2<<cat)+(nzeros-1>>1);
                best_qc=val+val_s^val_s;
              }
            }
          }
          /*zzj can't be coded as a zero, so stop trying to extend the run.*/
          if(!(zflags>>zzj&1))break;
        }
        /*We could try to consider _all_ potentially non-zero coefficients, but
           if we already found a bunch of them not worth coding, it's fairly
           unlikely they would now be worth coding from this position; skipping
           them saves a lot of work.*/
        zzj=(tokens[zzj][0].next>>1)-(tokens[zzj][0].qc!=0)&63;
        if(zzj==0){
          /*We made it all the way to the end of the block; try an EOB token.*/
          if(eob<4095){
            bits=oc_token_bits(_enc,huffi,zzi,oc_make_eob_token(eob+1))
             -flush_bits;
          }
          else bits=oc_token_bits(_enc,huffi,zzi,OC_DCT_EOB1_TOKEN);
          cost=sum_d2+bits*lambda;
          /*If the best route so far is still a pure zero run to the end of the
             block, force coding it as an EOB.
            Even if it's not optimal for this block, it has a good chance of
             getting combined with an EOB token from subsequent blocks, saving
             bits overall.*/
          if(cost<=best_cost||best_token<=OC_DCT_ZRL_TOKEN&&zzi+best_eb==63){
            best_next=0;
            /*This token is just a marker; in reality we may not emit any
               tokens, but update eob_run[] instead.*/
            best_token=OC_DCT_EOB1_TOKEN;
            best_eb=0;
            best_cost=cost;
            best_bits=bits;
            best_qc=0;
          }
          break;
        }
        nzeros=zzj-zzi;
      }
      tokens[zzi][0].next=(unsigned char)best_next;
      tokens[zzi][0].token=(signed char)best_token;
      tokens[zzi][0].eb=(ogg_int16_t)best_eb;
      tokens[zzi][0].cost=best_cost;
      tokens[zzi][0].bits=best_bits;
      tokens[zzi][0].qc=best_qc;
      zflags|=(ogg_int64_t)1<<zzi;
      if(qc){
        dq=_dequant[zzi];
        if(zzi<_acmin)lambda=0;
        e=dq-c;
        d2=e*(ogg_int32_t)e;
        token=OC_ONE_TOKEN-s;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        zzj=zzi+1&63;
        tj=best_flags>>zzj&1;
        next=(zzj<<1)+tj;
        tokens[zzi][1].next=(unsigned char)next;
        tokens[zzi][1].token=(signed char)token;
        tokens[zzi][1].eb=0;
        tokens[zzi][1].cost=d2+lambda*bits+tokens[zzj][tj].cost;
        tokens[zzi][1].bits=bits+tokens[zzj][tj].bits;
        tokens[zzi][1].qc=1+s^s;
        nzflags|=(ogg_int64_t)1<<zzi;
        best_flags|=
         (ogg_int64_t)(tokens[zzi][1].cost<tokens[zzi][0].cost)<<zzi;
      }
    }
    else{
      eob=eob_run[zzi];
      if(zzi<_acmin)lambda=0;
      c=c+s^s;
      dq=_dequant[zzi];
      /*No zero run can extend past this point.*/
      d2_accum[zzi]=0;
      flush_bits=eob>0?oc_token_bits(_enc,huffi,zzi,oc_make_eob_token(eob)):0;
      if(qc<=2){
        e=2*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_TWO_TOKEN-s;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e-=dq;
        d2=e*(ogg_int32_t)e;
        token=OC_ONE_TOKEN-s;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_bits=bits;
          best_cost=cost;
          qc--;
        }
        best_eb=0;
      }
      else if(qc<=3){
        e=3*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT2;
        best_eb=-s;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e-=dq;
        d2=e*(ogg_int32_t)e;
        token=OC_TWO_TOKEN-s;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_eb=0;
          best_bits=bits;
          best_cost=cost;
          qc--;
        }
      }
      else if(qc<=6){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT2+qc-3;
        best_eb=-s;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e-=dq;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_bits=bits;
          best_cost=cost;
          qc--;
        }
      }
      else if(qc<=8){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT3;
        best_eb=(-s<<1)+qc-7;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=6*dq-c;
        d2=e*(ogg_int32_t)e;
        token=OC_DCT_VAL_CAT2+3;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_eb=-s;
          best_bits=bits;
          best_cost=cost;
          qc=6;
        }
      }
      else if(qc<=12){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT4;
        best_eb=(-s<<2)+qc-9;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=8*dq-c;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_eb=(-s<<1)+1;
          best_bits=bits;
          best_cost=cost;
          qc=8;
        }
      }
      else if(qc<=20){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT5;
        best_eb=(-s<<3)+qc-13;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=12*dq-c;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_eb=(-s<<2)+3;
          best_bits=bits;
          best_cost=cost;
          qc=12;
        }
      }
      else if(qc<=36){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT6;
        best_eb=(-s<<4)+qc-21;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=20*dq-c;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<=best_cost){
          best_token=token;
          best_eb=(-s<<3)+7;
          best_bits=bits;
          best_cost=cost;
          qc=20;
        }
      }
      else if(qc<=68){
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT7;
        best_eb=(-s<<5)+qc-37;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=36*dq-c;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<best_cost){
          best_token=token;
          best_eb=(-s<<4)+15;
          best_bits=bits;
          best_cost=cost;
          qc=36;
        }
      }
      else{
        e=qc*dq-c;
        d2=e*(ogg_int32_t)e;
        best_token=OC_DCT_VAL_CAT8;
        best_eb=(-s<<9)+qc-69;
        best_bits=flush_bits+oc_token_bits(_enc,huffi,zzi,best_token);
        best_cost=d2+lambda*best_bits;
        e=68*dq-c;
        d2=e*(ogg_int32_t)e;
        token=best_token-1;
        bits=flush_bits+oc_token_bits(_enc,huffi,zzi,token);
        cost=d2+lambda*bits;
        if(cost<best_cost){
          best_token=token;
          best_eb=(-s<<5)+31;
          best_bits=bits;
          best_cost=cost;
          qc=68;
        }
      }
      zzj=zzi+1&63;
      tj=best_flags>>zzj&1;
      next=(zzj<<1)+tj;
      tokens[zzi][1].next=(unsigned char)next;
      tokens[zzi][1].token=(signed char)best_token;
      tokens[zzi][1].eb=best_eb;
      tokens[zzi][1].cost=best_cost+tokens[zzj][tj].cost;
      tokens[zzi][1].bits=best_bits+tokens[zzj][tj].bits;
      tokens[zzi][1].qc=qc+s^s;
      nzflags|=(ogg_int64_t)1<<zzi;
      best_flags|=(ogg_int64_t)1<<zzi;
    }
    zzj=zzi;
  }
  /*Emit the tokens from the best path through the trellis.*/
  stack=*_stack;
  /*We blow away the first entry here so that things vectorize better.
    The DC coefficient is not actually stored in the array yet.*/
  for(zzi=0;zzi<64;zzi++)_qdct[zzi]=0;
  dct_fzig_zag=_enc->state.opt_data.dct_fzig_zag;
  zzi=1;
  ti=best_flags>>1&1;
  bits=tokens[zzi][ti].bits;
  do{
    oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzi);
    eob=eob_run[zzi];
    if(tokens[zzi][ti].token<OC_NDCT_EOB_TOKEN_MAX){
      if(++eob>=4095){
        oc_enc_eob_log(_enc,_pli,zzi,eob);
        eob=0;
      }
      eob_run[zzi]=eob;
      /*We don't include the actual EOB cost for this block in the return value.
        It will be paid for by the fragment that terminates the EOB run.*/
      bits-=tokens[zzi][ti].bits;
      zzi=_zzi;
      break;
    }
    /*Emit pending EOB run if any.*/
    if(eob>0){
      oc_enc_eob_log(_enc,_pli,zzi,eob);
      eob_run[zzi]=0;
    }
    oc_enc_token_log(_enc,_pli,zzi,tokens[zzi][ti].token,tokens[zzi][ti].eb);
    next=tokens[zzi][ti].next;
    qc=tokens[zzi][ti].qc;
    zzj=(next>>1)-1&63;
    /*TODO: It may be worth saving the dequantized coefficient in the trellis
       above; we had to compute it to measure the error anyway.*/
    _qdct[dct_fzig_zag[zzj]]=(ogg_int16_t)(qc*(int)_dequant[zzj]);
    zzi=next>>1;
    ti=next&1;
  }
  while(zzi);
  *_stack=stack;
  return bits;
}

void oc_enc_pred_dc_frag_rows(oc_enc_ctx *_enc,
 int _pli,int _fragy0,int _frag_yend){
  const oc_fragment_plane *fplane;
  const oc_fragment       *frags;
  ogg_int16_t             *frag_dc;
  ptrdiff_t                fragi;
  int                     *pred_last;
  int                      nhfrags;
  int                      fragx;
  int                      fragy;
  fplane=_enc->state.fplanes+_pli;
  frags=_enc->state.frags;
  frag_dc=_enc->frag_dc;
  pred_last=_enc->dc_pred_last[_pli];
  nhfrags=fplane->nhfrags;
  fragi=fplane->froffset+_fragy0*nhfrags;
  for(fragy=_fragy0;fragy<_frag_yend;fragy++){
    if(fragy==0){
      /*For the first row, all of the cases reduce to just using the previous
         predictor for the same reference frame.*/
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        if(frags[fragi].coded){
          int ref;
          ref=OC_FRAME_FOR_MODE(frags[fragi].mb_mode);
          frag_dc[fragi]=(ogg_int16_t)(frags[fragi].dc-pred_last[ref]);
          pred_last[ref]=frags[fragi].dc;
        }
      }
    }
    else{
      const oc_fragment *u_frags;
      int                l_ref;
      int                ul_ref;
      int                u_ref;
      u_frags=frags-nhfrags;
      l_ref=-1;
      ul_ref=-1;
      u_ref=u_frags[fragi].coded?OC_FRAME_FOR_MODE(u_frags[fragi].mb_mode):-1;
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        int ur_ref;
        if(fragx+1>=nhfrags)ur_ref=-1;
        else{
          ur_ref=u_frags[fragi+1].coded?
           OC_FRAME_FOR_MODE(u_frags[fragi+1].mb_mode):-1;
        }
        if(frags[fragi].coded){
          int pred;
          int ref;
          ref=OC_FRAME_FOR_MODE(frags[fragi].mb_mode);
          /*We break out a separate case based on which of our neighbors use
             the same reference frames.
            This is somewhat faster than trying to make a generic case which
             handles all of them, since it reduces lots of poorly predicted
             jumps to one switch statement, and also lets a number of the
             multiplications be optimized out by strength reduction.*/
          switch((l_ref==ref)|(ul_ref==ref)<<1|
           (u_ref==ref)<<2|(ur_ref==ref)<<3){
            default:pred=pred_last[ref];break;
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
          frag_dc[fragi]=(ogg_int16_t)(frags[fragi].dc-pred);
          pred_last[ref]=frags[fragi].dc;
          l_ref=ref;
        }
        else l_ref=-1;
        ul_ref=u_ref;
        u_ref=ur_ref;
      }
    }
  }
}

void oc_enc_tokenize_dc_frag_list(oc_enc_ctx *_enc,int _pli,
 const ptrdiff_t *_coded_fragis,ptrdiff_t _ncoded_fragis,
 int _prev_ndct_tokens1,int _prev_eob_run1){
  const ogg_int16_t *frag_dc;
  ptrdiff_t          fragii;
  unsigned char     *dct_tokens0;
  unsigned char     *dct_tokens1;
  ogg_uint16_t      *extra_bits0;
  ogg_uint16_t      *extra_bits1;
  ptrdiff_t          ti0;
  ptrdiff_t          ti1r;
  ptrdiff_t          ti1w;
  int                eob_run0;
  int                eob_run1;
  int                neobs1;
  int                token;
  int                eb;
  int                token1=token1;
  int                eb1=eb1;
  /*Return immediately if there are no coded fragments; otherwise we'd flush
     any trailing EOB run into the AC 1 list and never read it back out.*/
  if(_ncoded_fragis<=0)return;
  frag_dc=_enc->frag_dc;
  dct_tokens0=_enc->dct_tokens[_pli][0];
  dct_tokens1=_enc->dct_tokens[_pli][1];
  extra_bits0=_enc->extra_bits[_pli][0];
  extra_bits1=_enc->extra_bits[_pli][1];
  ti0=_enc->ndct_tokens[_pli][0];
  ti1w=ti1r=_prev_ndct_tokens1;
  eob_run0=_enc->eob_run[_pli][0];
  /*Flush any trailing EOB run for the 1st AC coefficient.
    This is needed to allow us to track tokens to the end of the list.*/
  eob_run1=_enc->eob_run[_pli][1];
  if(eob_run1>0)oc_enc_eob_log(_enc,_pli,1,eob_run1);
  /*If there was an active EOB run at the start of the 1st AC stack, read it
     in and decode it.*/
  if(_prev_eob_run1>0){
    token1=dct_tokens1[ti1r];
    eb1=extra_bits1[ti1r];
    ti1r++;
    eob_run1=oc_decode_eob_token(token1,eb1);
    /*Consume the portion of the run that came before these fragments.*/
    neobs1=eob_run1-_prev_eob_run1;
  }
  else eob_run1=neobs1=0;
  for(fragii=0;fragii<_ncoded_fragis;fragii++){
    int val;
    /*All tokens in the 1st AC coefficient stack are regenerated as the DC
       coefficients are produced.
      This can be done in-place; stack 1 cannot get larger.*/
    if(!neobs1){
      /*There's no active EOB run in stack 1; read the next token.*/
      token1=dct_tokens1[ti1r];
      eb1=extra_bits1[ti1r];
      ti1r++;
      if(token1<OC_NDCT_EOB_TOKEN_MAX){
        neobs1=oc_decode_eob_token(token1,eb1);
        /*It's an EOB run; add it to the current (inactive) one.
          Because we may have moved entries to stack 0, we may have an
           opportunity to merge two EOB runs in stack 1.*/
        eob_run1+=neobs1;
      }
    }
    val=frag_dc[_coded_fragis[fragii]];
    if(val){
      /*There was a non-zero DC value, so there's no alteration to stack 1
         for this fragment; just code the stack 0 token.*/
      /*Flush any pending EOB run.*/
      if(eob_run0>0){
        token=oc_make_eob_token_full(eob_run0,&eb);
        dct_tokens0[ti0]=(unsigned char)token;
        extra_bits0[ti0]=(ogg_uint16_t)eb;
        ti0++;
        eob_run0=0;
      }
      token=oc_make_dct_token_full(0,0,val,&eb);
      dct_tokens0[ti0]=(unsigned char)token;
      extra_bits0[ti0]=(ogg_uint16_t)eb;
      ti0++;
    }
    else{
      /*Zero DC value; that means the entry in stack 1 might need to be coded
         from stack 0.
        This requires a stack 1 fixup.*/
      if(neobs1>0){
        /*We're in the middle of an active EOB run in stack 1.
          Move it to stack 0.*/
        if(++eob_run0>=4095){
          token=oc_make_eob_token_full(eob_run0,&eb);
          dct_tokens0[ti0]=(unsigned char)token;
          extra_bits0[ti0]=(ogg_uint16_t)eb;
          ti0++;
          eob_run0=0;
        }
        eob_run1--;
      }
      else{
        /*No active EOB run in stack 1, so we can't extend one in stack 0.
          Flush it if we've got it.*/
        if(eob_run0>0){
          token=oc_make_eob_token_full(eob_run0,&eb);
          dct_tokens0[ti0]=(unsigned char)token;
          extra_bits0[ti0]=(ogg_uint16_t)eb;
          ti0++;
          eob_run0=0;
        }
        /*Stack 1 token is one of: a pure zero run token, a single
           coefficient token, or a zero run/coefficient combo token.
          A zero run token is expanded and moved to token stack 0, and the
           stack 1 entry dropped.
          A single coefficient value may be transformed into combo token that
           is moved to stack 0, or if it cannot be combined, it is left alone
           and a single length-1 zero run is emitted in stack 0.
          A combo token is extended and moved to stack 0.
          During AC coding, we restrict the run lengths on combo tokens for
           stack 1 to guarantee we can extend them.*/
        switch(token1){
          case OC_DCT_SHORT_ZRL_TOKEN:{
            if(eb1<7){
              dct_tokens0[ti0]=OC_DCT_SHORT_ZRL_TOKEN;
              extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
              ti0++;
              /*Don't write the AC coefficient back out.*/
              continue;
            }
            /*Fall through.*/
          }
          case OC_DCT_ZRL_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_ZRL_TOKEN;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_ONE_TOKEN:
          case OC_MINUS_ONE_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1A;
            extra_bits0[ti0]=(ogg_uint16_t)(token1-OC_ONE_TOKEN);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_TWO_TOKEN:
          case OC_MINUS_TWO_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2A;
            extra_bits0[ti0]=(ogg_uint16_t)(token1-OC_TWO_TOKEN<<1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_VAL_CAT2:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2A;
            extra_bits0[ti0]=(ogg_uint16_t)((eb1<<1)+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1A:
          case OC_DCT_RUN_CAT1A+1:
          case OC_DCT_RUN_CAT1A+2:
          case OC_DCT_RUN_CAT1A+3:{
            dct_tokens0[ti0]=(unsigned char)(token1+1);
            extra_bits0[ti0]=(ogg_uint16_t)eb1;
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1A+4:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1B;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1<<2);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1B:{
            if((eb1&3)<3){
              dct_tokens0[ti0]=OC_DCT_RUN_CAT1B;
              extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
              ti0++;
              /*Don't write the AC coefficient back out.*/
              continue;
            }
            eb1=((eb1&4)<<1)-1;
            /*Fall through.*/
          }
          case OC_DCT_RUN_CAT1C:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1C;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT2A:{
            eb1=(eb1<<1)-1;
            /*Fall through.*/
          }
          case OC_DCT_RUN_CAT2B:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2B;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
        }
        /*We can't merge tokens, write a short zero run and keep going.*/
        dct_tokens0[ti0]=OC_DCT_SHORT_ZRL_TOKEN;
        extra_bits0[ti0]=0;
        ti0++;
      }
    }
    if(!neobs1){
      /*Flush any (inactive) EOB run.*/
      if(eob_run1>0){
        token=oc_make_eob_token_full(eob_run1,&eb);
        dct_tokens1[ti1w]=(unsigned char)token;
        extra_bits1[ti1w]=(ogg_uint16_t)eb;
        ti1w++;
        eob_run1=0;
      }
      /*There's no active EOB run, so log the current token.*/
      dct_tokens1[ti1w]=(unsigned char)token1;
      extra_bits1[ti1w]=(ogg_uint16_t)eb1;
      ti1w++;
    }
    else{
      /*Otherwise consume one EOB from the current run.*/
      neobs1--;
      /*If we have more than 4095 EOBs outstanding in stack1, flush the run.*/
      if(eob_run1-neobs1>=4095){
        token=oc_make_eob_token_full(4095,&eb);
        dct_tokens1[ti1w]=(unsigned char)token;
        extra_bits1[ti1w]=(ogg_uint16_t)eb;
        ti1w++;
        eob_run1-=4095;
      }
    }
  }
  /*Save the current state.*/
  _enc->ndct_tokens[_pli][0]=ti0;
  _enc->ndct_tokens[_pli][1]=ti1w;
  _enc->eob_run[_pli][0]=eob_run0;
  _enc->eob_run[_pli][1]=eob_run1;
}

/*Final EOB run welding.*/
void oc_enc_tokenize_finish(oc_enc_ctx *_enc){
  int pli;
  int zzi;
  /*Emit final EOB runs.*/
  for(pli=0;pli<3;pli++)for(zzi=0;zzi<64;zzi++){
    int eob_run;
    eob_run=_enc->eob_run[pli][zzi];
    if(eob_run>0)oc_enc_eob_log(_enc,pli,zzi,eob_run);
  }
  /*Merge the final EOB run of one token list with the start of the next, if
     possible.*/
  for(zzi=0;zzi<64;zzi++)for(pli=0;pli<3;pli++){
    int       old_tok1;
    int       old_tok2;
    int       old_eb1;
    int       old_eb2;
    int       new_tok;
    int       new_eb;
    int       zzj;
    int       plj;
    ptrdiff_t ti=ti;
    int       run_count;
    /*Make sure this coefficient has tokens at all.*/
    if(_enc->ndct_tokens[pli][zzi]<=0)continue;
    /*Ensure the first token is an EOB run.*/
    old_tok2=_enc->dct_tokens[pli][zzi][0];
    if(old_tok2>=OC_NDCT_EOB_TOKEN_MAX)continue;
    /*Search for a previous coefficient that has any tokens at all.*/
    old_tok1=OC_NDCT_EOB_TOKEN_MAX;
    for(zzj=zzi,plj=pli;zzj>=0;zzj--){
      while(plj-->0){
        ti=_enc->ndct_tokens[plj][zzj]-1;
        if(ti>=_enc->dct_token_offs[plj][zzj]){
          old_tok1=_enc->dct_tokens[plj][zzj][ti];
          break;
        }
      }
      if(plj>=0)break;
      plj=3;
    }
    /*Ensure its last token was an EOB run.*/
    if(old_tok1>=OC_NDCT_EOB_TOKEN_MAX)continue;
    /*Pull off the associated extra bits, if any, and decode the runs.*/
    old_eb1=_enc->extra_bits[plj][zzj][ti];
    old_eb2=_enc->extra_bits[pli][zzi][0];
    run_count=oc_decode_eob_token(old_tok1,old_eb1)
     +oc_decode_eob_token(old_tok2,old_eb2);
    /*We can't possibly combine these into one run.
      It might be possible to split them more optimally, but we'll just leave
       them as-is.*/
    if(run_count>=4096)continue;
    /*We CAN combine them into one run.*/
    new_tok=oc_make_eob_token_full(run_count,&new_eb);
    _enc->dct_tokens[plj][zzj][ti]=(unsigned char)new_tok;
    _enc->extra_bits[plj][zzj][ti]=(ogg_uint16_t)new_eb;
    _enc->dct_token_offs[pli][zzi]++;
  }
}
