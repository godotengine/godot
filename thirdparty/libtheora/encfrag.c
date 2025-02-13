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
#include <string.h>
#include "encint.h"


void oc_enc_frag_sub_c(ogg_int16_t _diff[64],const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  int i;
  for(i=0;i<8;i++){
    int j;
    for(j=0;j<8;j++)_diff[i*8+j]=(ogg_int16_t)(_src[j]-_ref[j]);
    _src+=_ystride;
    _ref+=_ystride;
  }
}

void oc_enc_frag_sub_128_c(ogg_int16_t *_diff,
 const unsigned char *_src,int _ystride){
  int i;
  for(i=0;i<8;i++){
    int j;
    for(j=0;j<8;j++)_diff[i*8+j]=(ogg_int16_t)(_src[j]-128);
    _src+=_ystride;
  }
}

unsigned oc_enc_frag_sad_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  unsigned sad;
  int      i;
  sad=0;
  for(i=8;i-->0;){
    int j;
    for(j=0;j<8;j++)sad+=abs(_src[j]-_ref[j]);
    _src+=_ystride;
    _ref+=_ystride;
  }
  return sad;
}

unsigned oc_enc_frag_sad_thresh_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _thresh){
  unsigned sad;
  int      i;
  sad=0;
  for(i=8;i-->0;){
    int j;
    for(j=0;j<8;j++)sad+=abs(_src[j]-_ref[j]);
    if(sad>_thresh)break;
    _src+=_ystride;
    _ref+=_ystride;
  }
  return sad;
}

unsigned oc_enc_frag_sad2_thresh_c(const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
 unsigned _thresh){
  unsigned sad;
  int      i;
  sad=0;
  for(i=8;i-->0;){
    int j;
    for(j=0;j<8;j++)sad+=abs(_src[j]-(_ref1[j]+_ref2[j]>>1));
    if(sad>_thresh)break;
    _src+=_ystride;
    _ref1+=_ystride;
    _ref2+=_ystride;
  }
  return sad;
}

unsigned oc_enc_frag_intra_sad_c(const unsigned char *_src, int _ystride){
  const unsigned char *src = _src;
  unsigned dc;
  unsigned sad;
  int      i;
  dc=0;
  for(i=8;i-->0;){
    int j;
    for(j=0;j<8;j++)dc+=src[j];
    src+=_ystride;
  }
  dc=dc+32>>6;
  sad=0;
  for(i=8;i-->0;){
    int j;
    for(j=0;j<8;j++)sad+=abs(_src[j]-dc);
    _src+=_ystride;
  }
  return sad;
}

static void oc_diff_hadamard(ogg_int16_t _buf[64],const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  int i;
  for(i=0;i<8;i++){
    int t0;
    int t1;
    int t2;
    int t3;
    int t4;
    int t5;
    int t6;
    int t7;
    int r;
    /*Hadamard stage 1:*/
    t0=_src[0]-_ref[0]+_src[4]-_ref[4];
    t4=_src[0]-_ref[0]-_src[4]+_ref[4];
    t1=_src[1]-_ref[1]+_src[5]-_ref[5];
    t5=_src[1]-_ref[1]-_src[5]+_ref[5];
    t2=_src[2]-_ref[2]+_src[6]-_ref[6];
    t6=_src[2]-_ref[2]-_src[6]+_ref[6];
    t3=_src[3]-_ref[3]+_src[7]-_ref[7];
    t7=_src[3]-_ref[3]-_src[7]+_ref[7];
    /*Hadamard stage 2:*/
    r=t0;
    t0+=t2;
    t2=r-t2;
    r=t1;
    t1+=t3;
    t3=r-t3;
    r=t4;
    t4+=t6;
    t6=r-t6;
    r=t5;
    t5+=t7;
    t7=r-t7;
    /*Hadamard stage 3:*/
    _buf[0*8+i]=(ogg_int16_t)(t0+t1);
    _buf[1*8+i]=(ogg_int16_t)(t0-t1);
    _buf[2*8+i]=(ogg_int16_t)(t2+t3);
    _buf[3*8+i]=(ogg_int16_t)(t2-t3);
    _buf[4*8+i]=(ogg_int16_t)(t4+t5);
    _buf[5*8+i]=(ogg_int16_t)(t4-t5);
    _buf[6*8+i]=(ogg_int16_t)(t6+t7);
    _buf[7*8+i]=(ogg_int16_t)(t6-t7);
    _src+=_ystride;
    _ref+=_ystride;
  }
}

static void oc_diff_hadamard2(ogg_int16_t _buf[64],const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride){
  int i;
  for(i=0;i<8;i++){
    int t0;
    int t1;
    int t2;
    int t3;
    int t4;
    int t5;
    int t6;
    int t7;
    int r;
    /*Hadamard stage 1:*/
    r=_ref1[0]+_ref2[0]>>1;
    t4=_ref1[4]+_ref2[4]>>1;
    t0=_src[0]-r+_src[4]-t4;
    t4=_src[0]-r-_src[4]+t4;
    r=_ref1[1]+_ref2[1]>>1;
    t5=_ref1[5]+_ref2[5]>>1;
    t1=_src[1]-r+_src[5]-t5;
    t5=_src[1]-r-_src[5]+t5;
    r=_ref1[2]+_ref2[2]>>1;
    t6=_ref1[6]+_ref2[6]>>1;
    t2=_src[2]-r+_src[6]-t6;
    t6=_src[2]-r-_src[6]+t6;
    r=_ref1[3]+_ref2[3]>>1;
    t7=_ref1[7]+_ref2[7]>>1;
    t3=_src[3]-r+_src[7]-t7;
    t7=_src[3]-r-_src[7]+t7;
    /*Hadamard stage 2:*/
    r=t0;
    t0+=t2;
    t2=r-t2;
    r=t1;
    t1+=t3;
    t3=r-t3;
    r=t4;
    t4+=t6;
    t6=r-t6;
    r=t5;
    t5+=t7;
    t7=r-t7;
    /*Hadamard stage 3:*/
    _buf[0*8+i]=(ogg_int16_t)(t0+t1);
    _buf[1*8+i]=(ogg_int16_t)(t0-t1);
    _buf[2*8+i]=(ogg_int16_t)(t2+t3);
    _buf[3*8+i]=(ogg_int16_t)(t2-t3);
    _buf[4*8+i]=(ogg_int16_t)(t4+t5);
    _buf[5*8+i]=(ogg_int16_t)(t4-t5);
    _buf[6*8+i]=(ogg_int16_t)(t6+t7);
    _buf[7*8+i]=(ogg_int16_t)(t6-t7);
    _src+=_ystride;
    _ref1+=_ystride;
    _ref2+=_ystride;
  }
}

static void oc_intra_hadamard(ogg_int16_t _buf[64],const unsigned char *_src,
 int _ystride){
  int i;
  for(i=0;i<8;i++){
    int t0;
    int t1;
    int t2;
    int t3;
    int t4;
    int t5;
    int t6;
    int t7;
    int r;
    /*Hadamard stage 1:*/
    t0=_src[0]+_src[4];
    t4=_src[0]-_src[4];
    t1=_src[1]+_src[5];
    t5=_src[1]-_src[5];
    t2=_src[2]+_src[6];
    t6=_src[2]-_src[6];
    t3=_src[3]+_src[7];
    t7=_src[3]-_src[7];
    /*Hadamard stage 2:*/
    r=t0;
    t0+=t2;
    t2=r-t2;
    r=t1;
    t1+=t3;
    t3=r-t3;
    r=t4;
    t4+=t6;
    t6=r-t6;
    r=t5;
    t5+=t7;
    t7=r-t7;
    /*Hadamard stage 3:*/
    _buf[0*8+i]=(ogg_int16_t)(t0+t1);
    _buf[1*8+i]=(ogg_int16_t)(t0-t1);
    _buf[2*8+i]=(ogg_int16_t)(t2+t3);
    _buf[3*8+i]=(ogg_int16_t)(t2-t3);
    _buf[4*8+i]=(ogg_int16_t)(t4+t5);
    _buf[5*8+i]=(ogg_int16_t)(t4-t5);
    _buf[6*8+i]=(ogg_int16_t)(t6+t7);
    _buf[7*8+i]=(ogg_int16_t)(t6-t7);
    _src+=_ystride;
  }
}

unsigned oc_hadamard_sad(int *_dc,const ogg_int16_t _buf[64]){
  unsigned sad;
  int      dc;
  int      t0;
  int      t1;
  int      t2;
  int      t3;
  int      t4;
  int      t5;
  int      t6;
  int      t7;
  int      r;
  int      i;
  sad=dc=0;
  for(i=0;i<8;i++){
    /*Hadamard stage 1:*/
    t0=_buf[i*8+0]+_buf[i*8+4];
    t4=_buf[i*8+0]-_buf[i*8+4];
    t1=_buf[i*8+1]+_buf[i*8+5];
    t5=_buf[i*8+1]-_buf[i*8+5];
    t2=_buf[i*8+2]+_buf[i*8+6];
    t6=_buf[i*8+2]-_buf[i*8+6];
    t3=_buf[i*8+3]+_buf[i*8+7];
    t7=_buf[i*8+3]-_buf[i*8+7];
    /*Hadamard stage 2:*/
    r=t0;
    t0+=t2;
    t2=r-t2;
    r=t1;
    t1+=t3;
    t3=r-t3;
    r=t4;
    t4+=t6;
    t6=r-t6;
    r=t5;
    t5+=t7;
    t7=r-t7;
    /*Hadamard stage 3:*/
    r=abs(t0+t1)&-(i>0);
    r+=abs(t0-t1);
    r+=abs(t2+t3);
    r+=abs(t2-t3);
    r+=abs(t4+t5);
    r+=abs(t4-t5);
    r+=abs(t6+t7);
    r+=abs(t6-t7);
    sad+=r;
  }
  dc=_buf[0]+_buf[1]+_buf[2]+_buf[3]+_buf[4]+_buf[5]+_buf[6]+_buf[7];
  *_dc=dc;
  return sad;
}

unsigned oc_enc_frag_satd_c(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  ogg_int16_t buf[64];
  oc_diff_hadamard(buf,_src,_ref,_ystride);
  return oc_hadamard_sad(_dc,buf);
}

unsigned oc_enc_frag_satd2_c(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride){
  ogg_int16_t buf[64];
  oc_diff_hadamard2(buf,_src,_ref1,_ref2,_ystride);
  return oc_hadamard_sad(_dc,buf);
}

unsigned oc_enc_frag_intra_satd_c(int *_dc,
 const unsigned char *_src,int _ystride){
  ogg_int16_t buf[64];
  oc_intra_hadamard(buf,_src,_ystride);
  return oc_hadamard_sad(_dc,buf);
}

unsigned oc_enc_frag_ssd_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  unsigned ret;
  int      y;
  int      x;
  ret=0;
  for(y=0;y<8;y++){
    for(x=0;x<8;x++)ret+=(_src[x]-_ref[x])*(_src[x]-_ref[x]);
    _src+=_ystride;
    _ref+=_ystride;
  }
  return ret;
}

unsigned oc_enc_frag_border_ssd_c(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,ogg_int64_t _mask){
  unsigned ret;
  int      y;
  int      x;
  ret=0;
  for(y=0;y<8;y++){
    for(x=0;x<8;x++,_mask>>=1){
      if(_mask&1)ret+=(_src[x]-_ref[x])*(_src[x]-_ref[x]);
    }
    _src+=_ystride;
    _ref+=_ystride;
  }
  return ret;
}

void oc_enc_frag_copy2_c(unsigned char *_dst,
 const unsigned char *_src1,const unsigned char *_src2,int _ystride){
  int i;
  int j;
  for(i=8;i-->0;){
    for(j=0;j<8;j++)_dst[j]=_src1[j]+_src2[j]>>1;
    _dst+=_ystride;
    _src1+=_ystride;
    _src2+=_ystride;
  }
}
