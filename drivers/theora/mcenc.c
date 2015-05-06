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
  last mod: $Id$

 ********************************************************************/
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "encint.h"



typedef struct oc_mcenc_ctx           oc_mcenc_ctx;



/*Temporary state used for motion estimation.*/
struct oc_mcenc_ctx{
  /*The candidate motion vectors.*/
  int                candidates[13][2];
  /*The start of the Set B candidates.*/
  int                setb0;
  /*The total number of candidates.*/
  int                ncandidates;
};



/*The maximum Y plane SAD value for accepting the median predictor.*/
#define OC_YSAD_THRESH1            (256)
/*The amount to right shift the minimum error by when inflating it for
   computing the second maximum Y plane SAD threshold.*/
#define OC_YSAD_THRESH2_SCALE_BITS (4)
/*The amount to add to the second maximum Y plane threshold when inflating
   it.*/
#define OC_YSAD_THRESH2_OFFSET     (64)

/*The vector offsets in the X direction for each search site in the square
   pattern.*/
static const int OC_SQUARE_DX[9]={-1,0,1,-1,0,1,-1,0,1};
/*The vector offsets in the Y direction for each search site in the square
   pattern.*/
static const int OC_SQUARE_DY[9]={-1,-1,-1,0,0,0,1,1,1};
/*The number of sites to search for each boundary condition in the square
   pattern.
  Bit flags for the boundary conditions are as follows:
  1: -16==dx
  2:      dx==15(.5)
  4: -16==dy
  8:      dy==15(.5)*/
static const int OC_SQUARE_NSITES[11]={8,5,5,0,5,3,3,0,5,3,3};
/*The list of sites to search for each boundary condition in the square
   pattern.*/
static const int OC_SQUARE_SITES[11][8]={
  /* -15.5<dx<31,       -15.5<dy<15(.5)*/
  {0,1,2,3,5,6,7,8},
  /*-15.5==dx,          -15.5<dy<15(.5)*/
  {1,2,5,7,8},
  /*     dx==15(.5),    -15.5<dy<15(.5)*/
  {0,1,3,6,7},
  /*-15.5==dx==15(.5),  -15.5<dy<15(.5)*/
  {-1},
  /* -15.5<dx<15(.5),  -15.5==dy*/
  {3,5,6,7,8},
  /*-15.5==dx,         -15.5==dy*/
  {5,7,8},
  /*     dx==15(.5),   -15.5==dy*/
  {3,6,7},
  /*-15.5==dx==15(.5), -15.5==dy*/
  {-1},
  /*-15.5dx<15(.5),           dy==15(.5)*/
  {0,1,2,3,5},
  /*-15.5==dx,                dy==15(.5)*/
  {1,2,5},
  /*       dx==15(.5),        dy==15(.5)*/
  {0,1,3}
};


static void oc_mcenc_find_candidates(oc_enc_ctx *_enc,oc_mcenc_ctx *_mcenc,
 int _accum[2],int _mbi,int _frame){
  oc_mb_enc_info *embs;
  int             a[3][2];
  int             ncandidates;
  unsigned        nmbi;
  int             i;
  embs=_enc->mb_info;
  /*Skip a position to store the median predictor in.*/
  ncandidates=1;
  if(embs[_mbi].ncneighbors>0){
    /*Fill in the first part of set A: the vectors from adjacent blocks.*/
    for(i=0;i<embs[_mbi].ncneighbors;i++){
      nmbi=embs[_mbi].cneighbors[i];
      _mcenc->candidates[ncandidates][0]=embs[nmbi].analysis_mv[0][_frame][0];
      _mcenc->candidates[ncandidates][1]=embs[nmbi].analysis_mv[0][_frame][1];
      ncandidates++;
    }
  }
  /*Add a few additional vectors to set A: the vectors used in the previous
     frames and the (0,0) vector.*/
  _mcenc->candidates[ncandidates][0]=OC_CLAMPI(-31,_accum[0],31);
  _mcenc->candidates[ncandidates][1]=OC_CLAMPI(-31,_accum[1],31);
  ncandidates++;
  _mcenc->candidates[ncandidates][0]=OC_CLAMPI(-31,
   embs[_mbi].analysis_mv[1][_frame][0]+_accum[0],31);
  _mcenc->candidates[ncandidates][1]=OC_CLAMPI(-31,
   embs[_mbi].analysis_mv[1][_frame][1]+_accum[1],31);
  ncandidates++;
  _mcenc->candidates[ncandidates][0]=0;
  _mcenc->candidates[ncandidates][1]=0;
  ncandidates++;
  /*Use the first three vectors of set A to find our best predictor: their
     median.*/
  memcpy(a,_mcenc->candidates+1,sizeof(a));
  OC_SORT2I(a[0][0],a[1][0]);
  OC_SORT2I(a[0][1],a[1][1]);
  OC_SORT2I(a[1][0],a[2][0]);
  OC_SORT2I(a[1][1],a[2][1]);
  OC_SORT2I(a[0][0],a[1][0]);
  OC_SORT2I(a[0][1],a[1][1]);
  _mcenc->candidates[0][0]=a[1][0];
  _mcenc->candidates[0][1]=a[1][1];
  /*Fill in set B: accelerated predictors for this and adjacent macro blocks.*/
  _mcenc->setb0=ncandidates;
  /*The first time through the loop use the current macro block.*/
  nmbi=_mbi;
  for(i=0;;i++){
    _mcenc->candidates[ncandidates][0]=OC_CLAMPI(-31,
     2*embs[_mbi].analysis_mv[1][_frame][0]
     -embs[_mbi].analysis_mv[2][_frame][0]+_accum[0],31);
    _mcenc->candidates[ncandidates][1]=OC_CLAMPI(-31,
     2*embs[_mbi].analysis_mv[1][_frame][1]
     -embs[_mbi].analysis_mv[2][_frame][1]+_accum[1],31);
    ncandidates++;
    if(i>=embs[_mbi].npneighbors)break;
    nmbi=embs[_mbi].pneighbors[i];
  }
  /*Truncate to full-pel positions.*/
  for(i=0;i<ncandidates;i++){
    _mcenc->candidates[i][0]=OC_DIV2(_mcenc->candidates[i][0]);
    _mcenc->candidates[i][1]=OC_DIV2(_mcenc->candidates[i][1]);
  }
  _mcenc->ncandidates=ncandidates;
}

#if 0
static unsigned oc_sad16_halfpel(const oc_enc_ctx *_enc,
 const ptrdiff_t *_frag_buf_offs,const ptrdiff_t _fragis[4],
 int _mvoffset0,int _mvoffset1,const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _best_err){
  unsigned err;
  int      bi;
  err=0;
  for(bi=0;bi<4;bi++){
    ptrdiff_t frag_offs;
    frag_offs=_frag_buf_offs[_fragis[bi]];
    err+=oc_enc_frag_sad2_thresh(_enc,_src+frag_offs,_ref+frag_offs+_mvoffset0,
     _ref+frag_offs+_mvoffset1,_ystride,_best_err-err);
  }
  return err;
}
#endif

static unsigned oc_satd16_halfpel(const oc_enc_ctx *_enc,
 const ptrdiff_t *_frag_buf_offs,const ptrdiff_t _fragis[4],
 int _mvoffset0,int _mvoffset1,const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _best_err){
  unsigned err;
  int      bi;
  err=0;
  for(bi=0;bi<4;bi++){
    ptrdiff_t frag_offs;
    frag_offs=_frag_buf_offs[_fragis[bi]];
    err+=oc_enc_frag_satd2_thresh(_enc,_src+frag_offs,_ref+frag_offs+_mvoffset0,
     _ref+frag_offs+_mvoffset1,_ystride,_best_err-err);
  }
  return err;
}

static unsigned oc_mcenc_ysad_check_mbcandidate_fullpel(const oc_enc_ctx *_enc,
 const ptrdiff_t *_frag_buf_offs,const ptrdiff_t _fragis[4],int _dx,int _dy,
 const unsigned char *_src,const unsigned char *_ref,int _ystride,
 unsigned _block_err[4]){
  unsigned err;
  int      mvoffset;
  int      bi;
  mvoffset=_dx+_dy*_ystride;
  err=0;
  for(bi=0;bi<4;bi++){
    ptrdiff_t frag_offs;
    unsigned  block_err;
    frag_offs=_frag_buf_offs[_fragis[bi]];
    block_err=oc_enc_frag_sad(_enc,
     _src+frag_offs,_ref+frag_offs+mvoffset,_ystride);
    _block_err[bi]=block_err;
    err+=block_err;
  }
  return err;
}

static int oc_mcenc_ysatd_check_mbcandidate_fullpel(const oc_enc_ctx *_enc,
 const ptrdiff_t *_frag_buf_offs,const ptrdiff_t _fragis[4],int _dx,int _dy,
 const unsigned char *_src,const unsigned char *_ref,int _ystride){
  int mvoffset;
  int err;
  int bi;
  mvoffset=_dx+_dy*_ystride;
  err=0;
  for(bi=0;bi<4;bi++){
    ptrdiff_t frag_offs;
    frag_offs=_frag_buf_offs[_fragis[bi]];
    err+=oc_enc_frag_satd_thresh(_enc,
     _src+frag_offs,_ref+frag_offs+mvoffset,_ystride,UINT_MAX);
  }
  return err;
}

static unsigned oc_mcenc_ysatd_check_bcandidate_fullpel(const oc_enc_ctx *_enc,
 ptrdiff_t _frag_offs,int _dx,int _dy,
 const unsigned char *_src,const unsigned char *_ref,int _ystride){
  return oc_enc_frag_satd_thresh(_enc,
   _src+_frag_offs,_ref+_frag_offs+_dx+_dy*_ystride,_ystride,UINT_MAX);
}

/*Perform a motion vector search for this macro block against a single
   reference frame.
  As a bonus, individual block motion vectors are computed as well, as much of
   the work can be shared.
  The actual motion vector is stored in the appropriate place in the
   oc_mb_enc_info structure.
  _mcenc:    The motion compensation context.
  _accum:    Drop frame/golden MV accumulators.
  _mbi:      The macro block index.
  _frame:    The frame to search, either OC_FRAME_PREV or OC_FRAME_GOLD.*/
void oc_mcenc_search_frame(oc_enc_ctx *_enc,int _accum[2],int _mbi,int _frame){
  /*Note: Traditionally this search is done using a rate-distortion objective
     function of the form D+lambda*R.
    However, xiphmont tested this and found it produced a small degredation,
     while requiring extra computation.
    This is most likely due to Theora's peculiar MV encoding scheme: MVs are
     not coded relative to a predictor, and the only truly cheap way to use a
     MV is in the LAST or LAST2 MB modes, which are not being considered here.
    Therefore if we use the MV found here, it's only because both LAST and
     LAST2 performed poorly, and therefore the MB is not likely to be uniform
     or suffer from the aperture problem.
    Furthermore we would like to re-use the MV found here for as many MBs as
     possible, so picking a slightly sub-optimal vector to save a bit or two
     may cause increased degredation in many blocks to come.
    We could artificially reduce lambda to compensate, but it's faster to just
     disable it entirely, and use D (the distortion) as the sole criterion.*/
  oc_mcenc_ctx         mcenc;
  const ptrdiff_t     *frag_buf_offs;
  const ptrdiff_t     *fragis;
  const unsigned char *src;
  const unsigned char *ref;
  int                  ystride;
  oc_mb_enc_info      *embs;
  ogg_int32_t          hit_cache[31];
  ogg_int32_t          hitbit;
  unsigned             best_block_err[4];
  unsigned             block_err[4];
  unsigned             best_err;
  int                  best_vec[2];
  int                  best_block_vec[4][2];
  int                  candx;
  int                  candy;
  int                  bi;
  embs=_enc->mb_info;
  /*Find some candidate motion vectors.*/
  oc_mcenc_find_candidates(_enc,&mcenc,_accum,_mbi,_frame);
  /*Clear the cache of locations we've examined.*/
  memset(hit_cache,0,sizeof(hit_cache));
  /*Start with the median predictor.*/
  candx=mcenc.candidates[0][0];
  candy=mcenc.candidates[0][1];
  hit_cache[candy+15]|=(ogg_int32_t)1<<candx+15;
  frag_buf_offs=_enc->state.frag_buf_offs;
  fragis=_enc->state.mb_maps[_mbi][0];
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[_frame]];
  ystride=_enc->state.ref_ystride[0];
  /*TODO: customize error function for speed/(quality+size) tradeoff.*/
  best_err=oc_mcenc_ysad_check_mbcandidate_fullpel(_enc,
   frag_buf_offs,fragis,candx,candy,src,ref,ystride,block_err);
  best_vec[0]=candx;
  best_vec[1]=candy;
  if(_frame==OC_FRAME_PREV){
    for(bi=0;bi<4;bi++){
      best_block_err[bi]=block_err[bi];
      best_block_vec[bi][0]=candx;
      best_block_vec[bi][1]=candy;
    }
  }
  /*If this predictor fails, move on to set A.*/
  if(best_err>OC_YSAD_THRESH1){
    unsigned err;
    unsigned t2;
    int      ncs;
    int      ci;
    /*Compute the early termination threshold for set A.*/
    t2=embs[_mbi].error[_frame];
    ncs=OC_MINI(3,embs[_mbi].ncneighbors);
    for(ci=0;ci<ncs;ci++){
      t2=OC_MAXI(t2,embs[embs[_mbi].cneighbors[ci]].error[_frame]);
    }
    t2+=(t2>>OC_YSAD_THRESH2_SCALE_BITS)+OC_YSAD_THRESH2_OFFSET;
    /*Examine the candidates in set A.*/
    for(ci=1;ci<mcenc.setb0;ci++){
      candx=mcenc.candidates[ci][0];
      candy=mcenc.candidates[ci][1];
      /*If we've already examined this vector, then we would be using it if it
         was better than what we are using.*/
      hitbit=(ogg_int32_t)1<<candx+15;
      if(hit_cache[candy+15]&hitbit)continue;
      hit_cache[candy+15]|=hitbit;
      err=oc_mcenc_ysad_check_mbcandidate_fullpel(_enc,
       frag_buf_offs,fragis,candx,candy,src,ref,ystride,block_err);
      if(err<best_err){
        best_err=err;
        best_vec[0]=candx;
        best_vec[1]=candy;
      }
      if(_frame==OC_FRAME_PREV){
        for(bi=0;bi<4;bi++)if(block_err[bi]<best_block_err[bi]){
          best_block_err[bi]=block_err[bi];
          best_block_vec[bi][0]=candx;
          best_block_vec[bi][1]=candy;
        }
      }
    }
    if(best_err>t2){
      /*Examine the candidates in set B.*/
      for(;ci<mcenc.ncandidates;ci++){
        candx=mcenc.candidates[ci][0];
        candy=mcenc.candidates[ci][1];
        hitbit=(ogg_int32_t)1<<candx+15;
        if(hit_cache[candy+15]&hitbit)continue;
        hit_cache[candy+15]|=hitbit;
        err=oc_mcenc_ysad_check_mbcandidate_fullpel(_enc,
         frag_buf_offs,fragis,candx,candy,src,ref,ystride,block_err);
        if(err<best_err){
          best_err=err;
          best_vec[0]=candx;
          best_vec[1]=candy;
        }
        if(_frame==OC_FRAME_PREV){
          for(bi=0;bi<4;bi++)if(block_err[bi]<best_block_err[bi]){
            best_block_err[bi]=block_err[bi];
            best_block_vec[bi][0]=candx;
            best_block_vec[bi][1]=candy;
          }
        }
      }
      /*Use the same threshold for set B as in set A.*/
      if(best_err>t2){
        int best_site;
        int nsites;
        int sitei;
        int site;
        int b;
        /*Square pattern search.*/
        for(;;){
          best_site=4;
          /*Compose the bit flags for boundary conditions.*/
          b=OC_DIV16(-best_vec[0]+1)|OC_DIV16(best_vec[0]+1)<<1|
           OC_DIV16(-best_vec[1]+1)<<2|OC_DIV16(best_vec[1]+1)<<3;
          nsites=OC_SQUARE_NSITES[b];
          for(sitei=0;sitei<nsites;sitei++){
            site=OC_SQUARE_SITES[b][sitei];
            candx=best_vec[0]+OC_SQUARE_DX[site];
            candy=best_vec[1]+OC_SQUARE_DY[site];
            hitbit=(ogg_int32_t)1<<candx+15;
            if(hit_cache[candy+15]&hitbit)continue;
            hit_cache[candy+15]|=hitbit;
            err=oc_mcenc_ysad_check_mbcandidate_fullpel(_enc,
             frag_buf_offs,fragis,candx,candy,src,ref,ystride,block_err);
            if(err<best_err){
              best_err=err;
              best_site=site;
            }
            if(_frame==OC_FRAME_PREV){
              for(bi=0;bi<4;bi++)if(block_err[bi]<best_block_err[bi]){
                best_block_err[bi]=block_err[bi];
                best_block_vec[bi][0]=candx;
                best_block_vec[bi][1]=candy;
              }
            }
          }
          if(best_site==4)break;
          best_vec[0]+=OC_SQUARE_DX[best_site];
          best_vec[1]+=OC_SQUARE_DY[best_site];
        }
        /*Final 4-MV search.*/
        /*Simply use 1/4 of the macro block set A and B threshold as the
           individual block threshold.*/
        if(_frame==OC_FRAME_PREV){
          t2>>=2;
          for(bi=0;bi<4;bi++){
            if(best_block_err[bi]>t2){
              /*Square pattern search.
                We do this in a slightly interesting manner.
                We continue to check the SAD of all four blocks in the
                 macro block.
                This gives us two things:
                 1) We can continue to use the hit_cache to avoid duplicate
                     checks.
                    Otherwise we could continue to read it, but not write to it
                     without saving and restoring it for each block.
                    Note that we could still eliminate a large number of
                     duplicate checks by taking into account the site we came
                     from when choosing the site list.
                    We can still do that to avoid extra hit_cache queries, and
                     it might even be a speed win.
                 2) It gives us a slightly better chance of escaping local
                     minima.
                    We would not be here if we weren't doing a fairly bad job
                     in finding a good vector, and checking these vectors can
                     save us from 100 to several thousand points off our SAD 1
                     in 15 times.
                TODO: Is this a good idea?
                Who knows.
                It needs more testing.*/
              for(;;){
                int bestx;
                int besty;
                int bj;
                bestx=best_block_vec[bi][0];
                besty=best_block_vec[bi][1];
                /*Compose the bit flags for boundary conditions.*/
                b=OC_DIV16(-bestx+1)|OC_DIV16(bestx+1)<<1|
                 OC_DIV16(-besty+1)<<2|OC_DIV16(besty+1)<<3;
                nsites=OC_SQUARE_NSITES[b];
                for(sitei=0;sitei<nsites;sitei++){
                  site=OC_SQUARE_SITES[b][sitei];
                  candx=bestx+OC_SQUARE_DX[site];
                  candy=besty+OC_SQUARE_DY[site];
                  hitbit=(ogg_int32_t)1<<candx+15;
                  if(hit_cache[candy+15]&hitbit)continue;
                  hit_cache[candy+15]|=hitbit;
                  err=oc_mcenc_ysad_check_mbcandidate_fullpel(_enc,
                   frag_buf_offs,fragis,candx,candy,src,ref,ystride,block_err);
                  if(err<best_err){
                    best_err=err;
                    best_vec[0]=candx;
                    best_vec[1]=candy;
                  }
                  for(bj=0;bj<4;bj++)if(block_err[bj]<best_block_err[bj]){
                    best_block_err[bj]=block_err[bj];
                    best_block_vec[bj][0]=candx;
                    best_block_vec[bj][1]=candy;
                  }
                }
                if(best_block_vec[bi][0]==bestx&&best_block_vec[bi][1]==besty){
                  break;
                }
              }
            }
          }
        }
      }
    }
  }
  embs[_mbi].error[_frame]=(ogg_uint16_t)best_err;
  candx=best_vec[0];
  candy=best_vec[1];
  embs[_mbi].satd[_frame]=oc_mcenc_ysatd_check_mbcandidate_fullpel(_enc,
   frag_buf_offs,fragis,candx,candy,src,ref,ystride);
  embs[_mbi].analysis_mv[0][_frame][0]=(signed char)(candx<<1);
  embs[_mbi].analysis_mv[0][_frame][1]=(signed char)(candy<<1);
  if(_frame==OC_FRAME_PREV){
    for(bi=0;bi<4;bi++){
      candx=best_block_vec[bi][0];
      candy=best_block_vec[bi][1];
      embs[_mbi].block_satd[bi]=oc_mcenc_ysatd_check_bcandidate_fullpel(_enc,
       frag_buf_offs[fragis[bi]],candx,candy,src,ref,ystride);
      embs[_mbi].block_mv[bi][0]=(signed char)(candx<<1);
      embs[_mbi].block_mv[bi][1]=(signed char)(candy<<1);
    }
  }
}

void oc_mcenc_search(oc_enc_ctx *_enc,int _mbi){
  oc_mv2         *mvs;
  int             accum_p[2];
  int             accum_g[2];
  mvs=_enc->mb_info[_mbi].analysis_mv;
  if(_enc->prevframe_dropped){
    accum_p[0]=mvs[0][OC_FRAME_PREV][0];
    accum_p[1]=mvs[0][OC_FRAME_PREV][1];
  }
  else accum_p[1]=accum_p[0]=0;
  accum_g[0]=mvs[2][OC_FRAME_GOLD][0];
  accum_g[1]=mvs[2][OC_FRAME_GOLD][1];
  mvs[0][OC_FRAME_PREV][0]-=mvs[2][OC_FRAME_PREV][0];
  mvs[0][OC_FRAME_PREV][1]-=mvs[2][OC_FRAME_PREV][1];
  /*Move the motion vector predictors back a frame.*/
  memmove(mvs+1,mvs,2*sizeof(*mvs));
  /*Search the last frame.*/
  oc_mcenc_search_frame(_enc,accum_p,_mbi,OC_FRAME_PREV);
  mvs[2][OC_FRAME_PREV][0]=accum_p[0];
  mvs[2][OC_FRAME_PREV][1]=accum_p[1];
  /*GOLDEN MVs are different from PREV MVs in that they're each absolute
     offsets from some frame in the past rather than relative offsets from the
     frame before.
    For predictor calculation to make sense, we need them to be in the same
     form as PREV MVs.*/
  mvs[1][OC_FRAME_GOLD][0]-=mvs[2][OC_FRAME_GOLD][0];
  mvs[1][OC_FRAME_GOLD][1]-=mvs[2][OC_FRAME_GOLD][1];
  mvs[2][OC_FRAME_GOLD][0]-=accum_g[0];
  mvs[2][OC_FRAME_GOLD][1]-=accum_g[1];
  /*Search the golden frame.*/
  oc_mcenc_search_frame(_enc,accum_g,_mbi,OC_FRAME_GOLD);
  /*Put GOLDEN MVs back into absolute offset form.
    The newest MV is already an absolute offset.*/
  mvs[2][OC_FRAME_GOLD][0]+=accum_g[0];
  mvs[2][OC_FRAME_GOLD][1]+=accum_g[1];
  mvs[1][OC_FRAME_GOLD][0]+=mvs[2][OC_FRAME_GOLD][0];
  mvs[1][OC_FRAME_GOLD][1]+=mvs[2][OC_FRAME_GOLD][1];
}

#if 0
static int oc_mcenc_ysad_halfpel_mbrefine(const oc_enc_ctx *_enc,int _mbi,
 int _vec[2],int _best_err,int _frame){
  const unsigned char *src;
  const unsigned char *ref;
  const ptrdiff_t     *frag_buf_offs;
  const ptrdiff_t     *fragis;
  int                  offset_y[9];
  int                  ystride;
  int                  mvoffset_base;
  int                  best_site;
  int                  sitei;
  int                  err;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[_framei]];
  frag_buf_offs=_enc->state.frag_buf_offs;
  fragis=_enc->state.mb_maps[_mbi][0];
  ystride=_enc->state.ref_ystride[0];
  mvoffset_base=_vec[0]+_vec[1]*ystride;
  offset_y[0]=offset_y[1]=offset_y[2]=-ystride;
  offset_y[3]=offset_y[5]=0;
  offset_y[6]=offset_y[7]=offset_y[8]=ystride;
  best_site=4;
  for(sitei=0;sitei<8;sitei++){
    int site;
    int xmask;
    int ymask;
    int dx;
    int dy;
    int mvoffset0;
    int mvoffset1;
    site=OC_SQUARE_SITES[0][sitei];
    dx=OC_SQUARE_DX[site];
    dy=OC_SQUARE_DY[site];
    /*The following code SHOULD be equivalent to
        oc_state_get_mv_offsets(&_mcenc->enc.state,&mvoffset0,&mvoffset1,
         (_vec[0]<<1)+dx,(_vec[1]<<1)+dy,ref_ystride,0);
      However, it should also be much faster, as it involves no multiplies and
       doesn't have to handle chroma vectors.*/
    xmask=OC_SIGNMASK(((_vec[0]<<1)+dx)^dx);
    ymask=OC_SIGNMASK(((_vec[1]<<1)+dy)^dy);
    mvoffset0=mvoffset_base+(dx&xmask)+(offset_y[site]&ymask);
    mvoffset1=mvoffset_base+(dx&~xmask)+(offset_y[site]&~ymask);
    err=oc_sad16_halfpel(_enc,frag_buf_offs,fragis,
     mvoffset0,mvoffset1,src,ref,ystride,_best_err);
    if(err<_best_err){
      _best_err=err;
      best_site=site;
    }
  }
  _vec[0]=(_vec[0]<<1)+OC_SQUARE_DX[best_site];
  _vec[1]=(_vec[1]<<1)+OC_SQUARE_DY[best_site];
  return _best_err;
}
#endif

static unsigned oc_mcenc_ysatd_halfpel_mbrefine(const oc_enc_ctx *_enc,
 int _mbi,int _vec[2],unsigned _best_err,int _frame){
  const unsigned char *src;
  const unsigned char *ref;
  const ptrdiff_t     *frag_buf_offs;
  const ptrdiff_t     *fragis;
  int                  offset_y[9];
  int                  ystride;
  int                  mvoffset_base;
  int                  best_site;
  int                  sitei;
  int                  err;
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[_frame]];
  frag_buf_offs=_enc->state.frag_buf_offs;
  fragis=_enc->state.mb_maps[_mbi][0];
  ystride=_enc->state.ref_ystride[0];
  mvoffset_base=_vec[0]+_vec[1]*ystride;
  offset_y[0]=offset_y[1]=offset_y[2]=-ystride;
  offset_y[3]=offset_y[5]=0;
  offset_y[6]=offset_y[7]=offset_y[8]=ystride;
  best_site=4;
  for(sitei=0;sitei<8;sitei++){
    int site;
    int xmask;
    int ymask;
    int dx;
    int dy;
    int mvoffset0;
    int mvoffset1;
    site=OC_SQUARE_SITES[0][sitei];
    dx=OC_SQUARE_DX[site];
    dy=OC_SQUARE_DY[site];
    /*The following code SHOULD be equivalent to
        oc_state_get_mv_offsets(&_mcenc->enc.state,&mvoffset0,&mvoffset1,
         (_vec[0]<<1)+dx,(_vec[1]<<1)+dy,ref_ystride,0);
      However, it should also be much faster, as it involves no multiplies and
       doesn't have to handle chroma vectors.*/
    xmask=OC_SIGNMASK(((_vec[0]<<1)+dx)^dx);
    ymask=OC_SIGNMASK(((_vec[1]<<1)+dy)^dy);
    mvoffset0=mvoffset_base+(dx&xmask)+(offset_y[site]&ymask);
    mvoffset1=mvoffset_base+(dx&~xmask)+(offset_y[site]&~ymask);
    err=oc_satd16_halfpel(_enc,frag_buf_offs,fragis,
     mvoffset0,mvoffset1,src,ref,ystride,_best_err);
    if(err<_best_err){
      _best_err=err;
      best_site=site;
    }
  }
  _vec[0]=(_vec[0]<<1)+OC_SQUARE_DX[best_site];
  _vec[1]=(_vec[1]<<1)+OC_SQUARE_DY[best_site];
  return _best_err;
}

void oc_mcenc_refine1mv(oc_enc_ctx *_enc,int _mbi,int _frame){
  oc_mb_enc_info *embs;
  int             vec[2];
  embs=_enc->mb_info;
  vec[0]=OC_DIV2(embs[_mbi].analysis_mv[0][_frame][0]);
  vec[1]=OC_DIV2(embs[_mbi].analysis_mv[0][_frame][1]);
  embs[_mbi].satd[_frame]=oc_mcenc_ysatd_halfpel_mbrefine(_enc,
   _mbi,vec,embs[_mbi].satd[_frame],_frame);
  embs[_mbi].analysis_mv[0][_frame][0]=(signed char)vec[0];
  embs[_mbi].analysis_mv[0][_frame][1]=(signed char)vec[1];
}

#if 0
static int oc_mcenc_ysad_halfpel_brefine(const oc_enc_ctx *_enc,
 int _vec[2],const unsigned char *_src,const unsigned char *_ref,int _ystride,
 int _offset_y[9],unsigned _best_err){
  int mvoffset_base;
  int best_site;
  int sitei;
  mvoffset_base=_vec[0]+_vec[1]*_ystride;
  best_site=4;
  for(sitei=0;sitei<8;sitei++){
    unsigned err;
    int      site;
    int      xmask;
    int      ymask;
    int      dx;
    int      dy;
    int      mvoffset0;
    int      mvoffset1;
    site=OC_SQUARE_SITES[0][sitei];
    dx=OC_SQUARE_DX[site];
    dy=OC_SQUARE_DY[site];
    /*The following code SHOULD be equivalent to
        oc_state_get_mv_offsets(&_mcenc->enc.state,&mvoffset0,&mvoffset1,
         (_vec[0]<<1)+dx,(_vec[1]<<1)+dy,ref_ystride,0);
      However, it should also be much faster, as it involves no multiplies and
       doesn't have to handle chroma vectors.*/
    xmask=OC_SIGNMASK(((_vec[0]<<1)+dx)^dx);
    ymask=OC_SIGNMASK(((_vec[1]<<1)+dy)^dy);
    mvoffset0=mvoffset_base+(dx&xmask)+(_offset_y[site]&ymask);
    mvoffset1=mvoffset_base+(dx&~xmask)+(_offset_y[site]&~ymask);
    err=oc_enc_frag_sad2_thresh(_enc,_src,
     _ref+mvoffset0,_ref+mvoffset1,ystride,_best_err);
    if(err<_best_err){
      _best_err=err;
      best_site=site;
    }
  }
  _vec[0]=(_vec[0]<<1)+OC_SQUARE_DX[best_site];
  _vec[1]=(_vec[1]<<1)+OC_SQUARE_DY[best_site];
  return _best_err;
}
#endif

static unsigned oc_mcenc_ysatd_halfpel_brefine(const oc_enc_ctx *_enc,
 int _vec[2],const unsigned char *_src,const unsigned char *_ref,int _ystride,
 int _offset_y[9],unsigned _best_err){
  int mvoffset_base;
  int best_site;
  int sitei;
  mvoffset_base=_vec[0]+_vec[1]*_ystride;
  best_site=4;
  for(sitei=0;sitei<8;sitei++){
    unsigned err;
    int      site;
    int      xmask;
    int      ymask;
    int      dx;
    int      dy;
    int      mvoffset0;
    int      mvoffset1;
    site=OC_SQUARE_SITES[0][sitei];
    dx=OC_SQUARE_DX[site];
    dy=OC_SQUARE_DY[site];
    /*The following code SHOULD be equivalent to
        oc_state_get_mv_offsets(&_enc->state,&mvoffsets,0,
         (_vec[0]<<1)+dx,(_vec[1]<<1)+dy);
      However, it should also be much faster, as it involves no multiplies and
       doesn't have to handle chroma vectors.*/
    xmask=OC_SIGNMASK(((_vec[0]<<1)+dx)^dx);
    ymask=OC_SIGNMASK(((_vec[1]<<1)+dy)^dy);
    mvoffset0=mvoffset_base+(dx&xmask)+(_offset_y[site]&ymask);
    mvoffset1=mvoffset_base+(dx&~xmask)+(_offset_y[site]&~ymask);
    err=oc_enc_frag_satd2_thresh(_enc,_src,
     _ref+mvoffset0,_ref+mvoffset1,_ystride,_best_err);
    if(err<_best_err){
      _best_err=err;
      best_site=site;
    }
  }
  _vec[0]=(_vec[0]<<1)+OC_SQUARE_DX[best_site];
  _vec[1]=(_vec[1]<<1)+OC_SQUARE_DY[best_site];
  return _best_err;
}

void oc_mcenc_refine4mv(oc_enc_ctx *_enc,int _mbi){
  oc_mb_enc_info      *embs;
  const ptrdiff_t     *frag_buf_offs;
  const ptrdiff_t     *fragis;
  const unsigned char *src;
  const unsigned char *ref;
  int                  offset_y[9];
  int                  ystride;
  int                  bi;
  ystride=_enc->state.ref_ystride[0];
  frag_buf_offs=_enc->state.frag_buf_offs;
  fragis=_enc->state.mb_maps[_mbi][0];
  src=_enc->state.ref_frame_data[OC_FRAME_IO];
  ref=_enc->state.ref_frame_data[_enc->state.ref_frame_idx[OC_FRAME_PREV]];
  offset_y[0]=offset_y[1]=offset_y[2]=-ystride;
  offset_y[3]=offset_y[5]=0;
  offset_y[6]=offset_y[7]=offset_y[8]=ystride;
  embs=_enc->mb_info;
  for(bi=0;bi<4;bi++){
    ptrdiff_t frag_offs;
    int       vec[2];
    frag_offs=frag_buf_offs[fragis[bi]];
    vec[0]=OC_DIV2(embs[_mbi].block_mv[bi][0]);
    vec[1]=OC_DIV2(embs[_mbi].block_mv[bi][1]);
    embs[_mbi].block_satd[bi]=oc_mcenc_ysatd_halfpel_brefine(_enc,vec,
     src+frag_offs,ref+frag_offs,ystride,offset_y,embs[_mbi].block_satd[bi]);
    embs[_mbi].ref_mv[bi][0]=(signed char)vec[0];
    embs[_mbi].ref_mv[bi][1]=(signed char)vec[1];
  }
}
