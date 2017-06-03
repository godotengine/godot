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
    last mod: $Id: internal.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "internal.h"



/*A map from the index in the zig zag scan to the coefficient number in a
   block.
  All zig zag indices beyond 63 are sent to coefficient 64, so that zero runs
   past the end of a block in bogus streams get mapped to a known location.*/
const unsigned char OC_FZIG_ZAG[128]={
   0, 1, 8,16, 9, 2, 3,10,
  17,24,32,25,18,11, 4, 5,
  12,19,26,33,40,48,41,34,
  27,20,13, 6, 7,14,21,28,
  35,42,49,56,57,50,43,36,
  29,22,15,23,30,37,44,51,
  58,59,52,45,38,31,39,46,
  53,60,61,54,47,55,62,63,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64
};

/*A map from the coefficient number in a block to its index in the zig zag
   scan.*/
const unsigned char OC_IZIG_ZAG[64]={
   0, 1, 5, 6,14,15,27,28,
   2, 4, 7,13,16,26,29,42,
   3, 8,12,17,25,30,41,43,
   9,11,18,24,31,40,44,53,
  10,19,23,32,39,45,52,54,
  20,22,33,38,46,51,55,60,
  21,34,37,47,50,56,59,61,
  35,36,48,49,57,58,62,63
};

/*A map from physical macro block ordering to bitstream macro block
   ordering within a super block.*/
const unsigned char OC_MB_MAP[2][2]={{0,3},{1,2}};

/*A list of the indices in the oc_mb.map array that can be valid for each of
   the various chroma decimation types.*/
const unsigned char OC_MB_MAP_IDXS[TH_PF_NFORMATS][12]={
  {0,1,2,3,4,8},
  {0,1,2,3,4,5,8,9},
  {0,1,2,3,4,6,8,10},
  {0,1,2,3,4,5,6,7,8,9,10,11}
};

/*The number of indices in the oc_mb.map array that can be valid for each of
   the various chroma decimation types.*/
const unsigned char OC_MB_MAP_NIDXS[TH_PF_NFORMATS]={6,8,8,12};

/*The number of extra bits that are coded with each of the DCT tokens.
  Each DCT token has some fixed number of additional bits (possibly 0) stored
   after the token itself, containing, for example, coefficient magnitude,
   sign bits, etc.*/
const unsigned char OC_DCT_TOKEN_EXTRA_BITS[TH_NDCT_TOKENS]={
  0,0,0,2,3,4,12,3,6,
  0,0,0,0,
  1,1,1,1,2,3,4,5,6,10,
  1,1,1,1,1,3,4,
  2,3
};



int oc_ilog(unsigned _v){
  int ret;
  for(ret=0;_v;ret++)_v>>=1;
  return ret;
}



/*The function used to fill in the chroma plane motion vectors for a macro
   block when 4 different motion vectors are specified in the luma plane.
  This version is for use with chroma decimated in the X and Y directions
   (4:2:0).
  _cbmvs: The chroma block-level motion vectors to fill in.
  _lbmvs: The luma block-level motion vectors.*/
static void oc_set_chroma_mvs00(oc_mv _cbmvs[4],const oc_mv _lbmvs[4]){
  int dx;
  int dy;
  dx=_lbmvs[0][0]+_lbmvs[1][0]+_lbmvs[2][0]+_lbmvs[3][0];
  dy=_lbmvs[0][1]+_lbmvs[1][1]+_lbmvs[2][1]+_lbmvs[3][1];
  _cbmvs[0][0]=(signed char)OC_DIV_ROUND_POW2(dx,2,2);
  _cbmvs[0][1]=(signed char)OC_DIV_ROUND_POW2(dy,2,2);
}

/*The function used to fill in the chroma plane motion vectors for a macro
   block when 4 different motion vectors are specified in the luma plane.
  This version is for use with chroma decimated in the Y direction.
  _cbmvs: The chroma block-level motion vectors to fill in.
  _lbmvs: The luma block-level motion vectors.*/
static void oc_set_chroma_mvs01(oc_mv _cbmvs[4],const oc_mv _lbmvs[4]){
  int dx;
  int dy;
  dx=_lbmvs[0][0]+_lbmvs[2][0];
  dy=_lbmvs[0][1]+_lbmvs[2][1];
  _cbmvs[0][0]=(signed char)OC_DIV_ROUND_POW2(dx,1,1);
  _cbmvs[0][1]=(signed char)OC_DIV_ROUND_POW2(dy,1,1);
  dx=_lbmvs[1][0]+_lbmvs[3][0];
  dy=_lbmvs[1][1]+_lbmvs[3][1];
  _cbmvs[1][0]=(signed char)OC_DIV_ROUND_POW2(dx,1,1);
  _cbmvs[1][1]=(signed char)OC_DIV_ROUND_POW2(dy,1,1);
}

/*The function used to fill in the chroma plane motion vectors for a macro
   block when 4 different motion vectors are specified in the luma plane.
  This version is for use with chroma decimated in the X direction (4:2:2).
  _cbmvs: The chroma block-level motion vectors to fill in.
  _lbmvs: The luma block-level motion vectors.*/
static void oc_set_chroma_mvs10(oc_mv _cbmvs[4],const oc_mv _lbmvs[4]){
  int dx;
  int dy;
  dx=_lbmvs[0][0]+_lbmvs[1][0];
  dy=_lbmvs[0][1]+_lbmvs[1][1];
  _cbmvs[0][0]=(signed char)OC_DIV_ROUND_POW2(dx,1,1);
  _cbmvs[0][1]=(signed char)OC_DIV_ROUND_POW2(dy,1,1);
  dx=_lbmvs[2][0]+_lbmvs[3][0];
  dy=_lbmvs[2][1]+_lbmvs[3][1];
  _cbmvs[2][0]=(signed char)OC_DIV_ROUND_POW2(dx,1,1);
  _cbmvs[2][1]=(signed char)OC_DIV_ROUND_POW2(dy,1,1);
}

/*The function used to fill in the chroma plane motion vectors for a macro
   block when 4 different motion vectors are specified in the luma plane.
  This version is for use with no chroma decimation (4:4:4).
  _cbmvs: The chroma block-level motion vectors to fill in.
  _lmbmv: The luma macro-block level motion vector to fill in for use in
           prediction.
  _lbmvs: The luma block-level motion vectors.*/
static void oc_set_chroma_mvs11(oc_mv _cbmvs[4],const oc_mv _lbmvs[4]){
  memcpy(_cbmvs,_lbmvs,4*sizeof(_lbmvs[0]));
}

/*A table of functions used to fill in the chroma plane motion vectors for a
   macro block when 4 different motion vectors are specified in the luma
   plane.*/
const oc_set_chroma_mvs_func OC_SET_CHROMA_MVS_TABLE[TH_PF_NFORMATS]={
  (oc_set_chroma_mvs_func)oc_set_chroma_mvs00,
  (oc_set_chroma_mvs_func)oc_set_chroma_mvs01,
  (oc_set_chroma_mvs_func)oc_set_chroma_mvs10,
  (oc_set_chroma_mvs_func)oc_set_chroma_mvs11
};



void **oc_malloc_2d(size_t _height,size_t _width,size_t _sz){
  size_t  rowsz;
  size_t  colsz;
  size_t  datsz;
  char   *ret;
  colsz=_height*sizeof(void *);
  rowsz=_sz*_width;
  datsz=rowsz*_height;
  /*Alloc array and row pointers.*/
  ret=(char *)_ogg_malloc(datsz+colsz);
  if(ret==NULL)return NULL;
  /*Initialize the array.*/
  if(ret!=NULL){
    size_t   i;
    void   **p;
    char    *datptr;
    p=(void **)ret;
    i=_height;
    for(datptr=ret+colsz;i-->0;p++,datptr+=rowsz)*p=(void *)datptr;
  }
  return (void **)ret;
}

void **oc_calloc_2d(size_t _height,size_t _width,size_t _sz){
  size_t  colsz;
  size_t  rowsz;
  size_t  datsz;
  char   *ret;
  colsz=_height*sizeof(void *);
  rowsz=_sz*_width;
  datsz=rowsz*_height;
  /*Alloc array and row pointers.*/
  ret=(char *)_ogg_calloc(datsz+colsz,1);
  if(ret==NULL)return NULL;
  /*Initialize the array.*/
  if(ret!=NULL){
    size_t   i;
    void   **p;
    char    *datptr;
    p=(void **)ret;
    i=_height;
    for(datptr=ret+colsz;i-->0;p++,datptr+=rowsz)*p=(void *)datptr;
  }
  return (void **)ret;
}

void oc_free_2d(void *_ptr){
  _ogg_free(_ptr);
}

/*Fills in a Y'CbCr buffer with a pointer to the image data in the first
   buffer, but with the opposite vertical orientation.
  _dst: The destination buffer.
        This can be the same as _src.
  _src: The source buffer.*/
void oc_ycbcr_buffer_flip(th_ycbcr_buffer _dst,
 const th_ycbcr_buffer _src){
  int pli;
  for(pli=0;pli<3;pli++){
    _dst[pli].width=_src[pli].width;
    _dst[pli].height=_src[pli].height;
    _dst[pli].stride=-_src[pli].stride;
    _dst[pli].data=_src[pli].data
     +(1-_dst[pli].height)*(ptrdiff_t)_dst[pli].stride;
  }
}

const char *th_version_string(void){
  return OC_VENDOR_STRING;
}

ogg_uint32_t th_version_number(void){
  return (TH_VERSION_MAJOR<<16)+(TH_VERSION_MINOR<<8)+TH_VERSION_SUB;
}

/*Determines the packet type.
  Note that this correctly interprets a 0-byte packet as a video data packet.
  Return: 1 for a header packet, 0 for a data packet.*/
int th_packet_isheader(ogg_packet *_op){
  return _op->bytes>0?_op->packet[0]>>7:0;
}

/*Determines the frame type of a video data packet.
  Note that this correctly interprets a 0-byte packet as a delta frame.
  Return: 1 for a key frame, 0 for a delta frame, and -1 for a header
           packet.*/
int th_packet_iskeyframe(ogg_packet *_op){
  return _op->bytes<=0?0:_op->packet[0]&0x80?-1:!(_op->packet[0]&0x40);
}
