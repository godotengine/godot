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



void *oc_aligned_malloc(size_t _sz,size_t _align){
  unsigned char *p;
  if(_align-1>UCHAR_MAX||(_align&_align-1)||_sz>~(size_t)0-_align)return NULL;
  p=(unsigned char *)_ogg_malloc(_sz+_align);
  if(p!=NULL){
    int offs;
    offs=((p-(unsigned char *)0)-1&_align-1);
    p[offs]=offs;
    p+=offs+1;
  }
  return p;
}

void oc_aligned_free(void *_ptr){
  unsigned char *p;
  p=(unsigned char *)_ptr;
  if(p!=NULL){
    int offs;
    offs=*--p;
    _ogg_free(p-offs);
  }
}


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
