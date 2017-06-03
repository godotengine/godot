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
  last mod: $Id: encoder_disabled.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/
#include "apiwrapper.h"
#include "encint.h"

th_enc_ctx *th_encode_alloc(const th_info *_info){
  return NULL;
}

void th_encode_free(th_enc_ctx *_enc){}


int th_encode_ctl(th_enc_ctx *_enc,int _req,void *_buf,size_t _buf_sz){
  return OC_DISABLED;
}

int th_encode_flushheader(th_enc_ctx *_enc,th_comment *_tc,ogg_packet *_op){
  return OC_DISABLED;
}

int th_encode_ycbcr_in(th_enc_ctx *_enc,th_ycbcr_buffer _img){
  return OC_DISABLED;
}

int th_encode_packetout(th_enc_ctx *_enc,int _last_p,ogg_packet *_op){
  return OC_DISABLED;
}



int theora_encode_init(theora_state *_te,theora_info *_ci){
  return OC_DISABLED;
}

int theora_encode_YUVin(theora_state *_te,yuv_buffer *_yuv){
  return OC_DISABLED;
}

int theora_encode_packetout(theora_state *_te,int _last_p,ogg_packet *_op){
  return OC_DISABLED;
}

int theora_encode_header(theora_state *_te,ogg_packet *_op){
  return OC_DISABLED;
}

int theora_encode_comment(theora_comment *_tc,ogg_packet *_op){
  return OC_DISABLED;
}

int theora_encode_tables(theora_state *_te,ogg_packet *_op){
  return OC_DISABLED;
}
