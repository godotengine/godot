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

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "apiwrapper.h"
#include "decint.h"
#include "theora/theoradec.h"

static void th_dec_api_clear(th_api_wrapper *_api){
  if(_api->setup)th_setup_free(_api->setup);
  if(_api->decode)th_decode_free(_api->decode);
  memset(_api,0,sizeof(*_api));
}

static void theora_decode_clear(theora_state *_td){
  if(_td->i!=NULL)theora_info_clear(_td->i);
  memset(_td,0,sizeof(*_td));
}

static int theora_decode_control(theora_state *_td,int _req,
 void *_buf,size_t _buf_sz){
  return th_decode_ctl(((th_api_wrapper *)_td->i->codec_setup)->decode,
   _req,_buf,_buf_sz);
}

static ogg_int64_t theora_decode_granule_frame(theora_state *_td,
 ogg_int64_t _gp){
  return th_granule_frame(((th_api_wrapper *)_td->i->codec_setup)->decode,_gp);
}

static double theora_decode_granule_time(theora_state *_td,ogg_int64_t _gp){
  return th_granule_time(((th_api_wrapper *)_td->i->codec_setup)->decode,_gp);
}

static const oc_state_dispatch_vtable OC_DEC_DISPATCH_VTBL={
  (oc_state_clear_func)theora_decode_clear,
  (oc_state_control_func)theora_decode_control,
  (oc_state_granule_frame_func)theora_decode_granule_frame,
  (oc_state_granule_time_func)theora_decode_granule_time,
};

static void th_info2theora_info(theora_info *_ci,const th_info *_info){
  _ci->version_major=_info->version_major;
  _ci->version_minor=_info->version_minor;
  _ci->version_subminor=_info->version_subminor;
  _ci->width=_info->frame_width;
  _ci->height=_info->frame_height;
  _ci->frame_width=_info->pic_width;
  _ci->frame_height=_info->pic_height;
  _ci->offset_x=_info->pic_x;
  _ci->offset_y=_info->pic_y;
  _ci->fps_numerator=_info->fps_numerator;
  _ci->fps_denominator=_info->fps_denominator;
  _ci->aspect_numerator=_info->aspect_numerator;
  _ci->aspect_denominator=_info->aspect_denominator;
  switch(_info->colorspace){
    case TH_CS_ITU_REC_470M:_ci->colorspace=OC_CS_ITU_REC_470M;break;
    case TH_CS_ITU_REC_470BG:_ci->colorspace=OC_CS_ITU_REC_470BG;break;
    default:_ci->colorspace=OC_CS_UNSPECIFIED;break;
  }
  switch(_info->pixel_fmt){
    case TH_PF_420:_ci->pixelformat=OC_PF_420;break;
    case TH_PF_422:_ci->pixelformat=OC_PF_422;break;
    case TH_PF_444:_ci->pixelformat=OC_PF_444;break;
    default:_ci->pixelformat=OC_PF_RSVD;
  }
  _ci->target_bitrate=_info->target_bitrate;
  _ci->quality=_info->quality;
  _ci->keyframe_frequency_force=1<<_info->keyframe_granule_shift;
}

int theora_decode_init(theora_state *_td,theora_info *_ci){
  th_api_info    *apiinfo;
  th_api_wrapper *api;
  th_info         info;
  api=(th_api_wrapper *)_ci->codec_setup;
  /*Allocate our own combined API wrapper/theora_info struct.
    We put them both in one malloc'd block so that when the API wrapper is
     freed, the info struct goes with it.
    This avoids having to figure out whether or not we need to free the info
     struct in either theora_info_clear() or theora_clear().*/
  apiinfo=(th_api_info *)_ogg_calloc(1,sizeof(*apiinfo));
  if(apiinfo==NULL)return OC_FAULT;
  /*Make our own copy of the info struct, since its lifetime should be
     independent of the one we were passed in.*/
  *&apiinfo->info=*_ci;
  /*Convert the info struct now instead of saving the the one we decoded with
     theora_decode_header(), since the user might have modified values (i.e.,
     color space, aspect ratio, etc. can be specified from a higher level).
    The user also might be doing something "clever" with the header packets if
     they are not using an Ogg encapsulation.*/
  oc_theora_info2th_info(&info,_ci);
  /*Don't bother to copy the setup info; th_decode_alloc() makes its own copy
     of the stuff it needs.*/
  apiinfo->api.decode=th_decode_alloc(&info,api->setup);
  if(apiinfo->api.decode==NULL){
    _ogg_free(apiinfo);
    return OC_EINVAL;
  }
  apiinfo->api.clear=(oc_setup_clear_func)th_dec_api_clear;
  _td->internal_encode=NULL;
  /*Provide entry points for ABI compatibility with old decoder shared libs.*/
  _td->internal_decode=(void *)&OC_DEC_DISPATCH_VTBL;
  _td->granulepos=0;
  _td->i=&apiinfo->info;
  _td->i->codec_setup=&apiinfo->api;
  return 0;
}

int theora_decode_header(theora_info *_ci,theora_comment *_cc,ogg_packet *_op){
  th_api_wrapper *api;
  th_info         info;
  int             ret;
  api=(th_api_wrapper *)_ci->codec_setup;
  /*Allocate an API wrapper struct on demand, since it will not also include a
     theora_info struct like the ones that are used in a theora_state struct.*/
  if(api==NULL){
    _ci->codec_setup=_ogg_calloc(1,sizeof(*api));
    if(_ci->codec_setup==NULL)return OC_FAULT;
    api=(th_api_wrapper *)_ci->codec_setup;
    api->clear=(oc_setup_clear_func)th_dec_api_clear;
  }
  /*Convert from the theora_info struct instead of saving our own th_info
     struct between calls.
    The user might be doing something "clever" with the header packets if they
     are not using an Ogg encapsulation, and we don't want to break this.*/
  oc_theora_info2th_info(&info,_ci);
  /*We rely on the fact that theora_comment and th_comment structures are
     actually identical.
    Take care not to change this fact unless you change the code here as
     well!*/
  ret=th_decode_headerin(&info,(th_comment *)_cc,&api->setup,_op);
  /*We also rely on the fact that the error return code values are the same,
    and that the implementations of these two functions return the same set of
    them.
   Note that theora_decode_header() really can return OC_NOTFORMAT, even
    though it is not currently documented to do so.*/
  if(ret<0)return ret;
  th_info2theora_info(_ci,&info);
  return 0;
}

int theora_decode_packetin(theora_state *_td,ogg_packet *_op){
  th_api_wrapper *api;
  ogg_int64_t     gp;
  int             ret;
  if(!_td||!_td->i||!_td->i->codec_setup)return OC_FAULT;
  api=(th_api_wrapper *)_td->i->codec_setup;
  ret=th_decode_packetin(api->decode,_op,&gp);
  if(ret<0)return OC_BADPACKET;
  _td->granulepos=gp;
  return 0;
}

int theora_decode_YUVout(theora_state *_td,yuv_buffer *_yuv){
  th_api_wrapper  *api;
  th_dec_ctx      *decode;
  th_ycbcr_buffer  buf;
  int              ret;
  if(!_td||!_td->i||!_td->i->codec_setup)return OC_FAULT;
  api=(th_api_wrapper *)_td->i->codec_setup;
  decode=(th_dec_ctx *)api->decode;
  if(!decode)return OC_FAULT;
  ret=th_decode_ycbcr_out(decode,buf);
  if(ret>=0){
    _yuv->y_width=buf[0].width;
    _yuv->y_height=buf[0].height;
    _yuv->y_stride=buf[0].stride;
    _yuv->uv_width=buf[1].width;
    _yuv->uv_height=buf[1].height;
    _yuv->uv_stride=buf[1].stride;
    _yuv->y=buf[0].data;
    _yuv->u=buf[1].data;
    _yuv->v=buf[2].data;
  }
  return ret;
}
