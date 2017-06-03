#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "apiwrapper.h"
#include "encint.h"
#include "theora/theoraenc.h"



static void th_enc_api_clear(th_api_wrapper *_api){
  if(_api->encode)th_encode_free(_api->encode);
  memset(_api,0,sizeof(*_api));
}

static void theora_encode_clear(theora_state *_te){
  if(_te->i!=NULL)theora_info_clear(_te->i);
  memset(_te,0,sizeof(*_te));
}

static int theora_encode_control(theora_state *_te,int _req,
 void *_buf,size_t _buf_sz){
  return th_encode_ctl(((th_api_wrapper *)_te->i->codec_setup)->encode,
   _req,_buf,_buf_sz);
}

static ogg_int64_t theora_encode_granule_frame(theora_state *_te,
 ogg_int64_t _gp){
  return th_granule_frame(((th_api_wrapper *)_te->i->codec_setup)->encode,_gp);
}

static double theora_encode_granule_time(theora_state *_te,ogg_int64_t _gp){
  return th_granule_time(((th_api_wrapper *)_te->i->codec_setup)->encode,_gp);
}

static const oc_state_dispatch_vtable OC_ENC_DISPATCH_VTBL={
  (oc_state_clear_func)theora_encode_clear,
  (oc_state_control_func)theora_encode_control,
  (oc_state_granule_frame_func)theora_encode_granule_frame,
  (oc_state_granule_time_func)theora_encode_granule_time,
};

int theora_encode_init(theora_state *_te,theora_info *_ci){
  th_api_info *apiinfo;
  th_info      info;
  ogg_uint32_t keyframe_frequency_force;
  /*Allocate our own combined API wrapper/theora_info struct.
    We put them both in one malloc'd block so that when the API wrapper is
     freed, the info struct goes with it.
    This avoids having to figure out whether or not we need to free the info
     struct in either theora_info_clear() or theora_clear().*/
  apiinfo=(th_api_info *)_ogg_malloc(sizeof(*apiinfo));
  if(apiinfo==NULL)return TH_EFAULT;
  /*Make our own copy of the info struct, since its lifetime should be
     independent of the one we were passed in.*/
  *&apiinfo->info=*_ci;
  oc_theora_info2th_info(&info,_ci);
  apiinfo->api.encode=th_encode_alloc(&info);
  if(apiinfo->api.encode==NULL){
    _ogg_free(apiinfo);
    return OC_EINVAL;
  }
  apiinfo->api.clear=(oc_setup_clear_func)th_enc_api_clear;
  /*Provide entry points for ABI compatibility with old decoder shared libs.*/
  _te->internal_encode=(void *)&OC_ENC_DISPATCH_VTBL;
  _te->internal_decode=NULL;
  _te->granulepos=0;
  _te->i=&apiinfo->info;
  _te->i->codec_setup=&apiinfo->api;
  /*Set the precise requested keyframe frequency.*/
  keyframe_frequency_force=_ci->keyframe_auto_p?
   _ci->keyframe_frequency_force:_ci->keyframe_frequency;
  th_encode_ctl(apiinfo->api.encode,
   TH_ENCCTL_SET_KEYFRAME_FREQUENCY_FORCE,
   &keyframe_frequency_force,sizeof(keyframe_frequency_force));
  /*TODO: Additional codec setup using the extra fields in theora_info.*/
  return 0;
}

int theora_encode_YUVin(theora_state *_te,yuv_buffer *_yuv){
  th_api_wrapper  *api;
  th_ycbcr_buffer  buf;
  int              ret;
  api=(th_api_wrapper *)_te->i->codec_setup;
  buf[0].width=_yuv->y_width;
  buf[0].height=_yuv->y_height;
  buf[0].stride=_yuv->y_stride;
  buf[0].data=_yuv->y;
  buf[1].width=_yuv->uv_width;
  buf[1].height=_yuv->uv_height;
  buf[1].stride=_yuv->uv_stride;
  buf[1].data=_yuv->u;
  buf[2].width=_yuv->uv_width;
  buf[2].height=_yuv->uv_height;
  buf[2].stride=_yuv->uv_stride;
  buf[2].data=_yuv->v;
  ret=th_encode_ycbcr_in(api->encode,buf);
  if(ret<0)return ret;
  _te->granulepos=api->encode->state.granpos;
  return ret;
}

int theora_encode_packetout(theora_state *_te,int _last_p,ogg_packet *_op){
  th_api_wrapper *api;
  api=(th_api_wrapper *)_te->i->codec_setup;
  return th_encode_packetout(api->encode,_last_p,_op);
}

int theora_encode_header(theora_state *_te,ogg_packet *_op){
  oc_enc_ctx     *enc;
  th_api_wrapper *api;
  int             ret;
  api=(th_api_wrapper *)_te->i->codec_setup;
  enc=api->encode;
  /*If we've already started encoding, fail.*/
  if(enc->packet_state>OC_PACKET_EMPTY||enc->state.granpos!=0){
    return TH_EINVAL;
  }
  /*Reset the state to make sure we output an info packet.*/
  enc->packet_state=OC_PACKET_INFO_HDR;
  ret=th_encode_flushheader(api->encode,NULL,_op);
  return ret>=0?0:ret;
}

int theora_encode_comment(theora_comment *_tc,ogg_packet *_op){
  oggpack_buffer  opb;
  void           *buf;
  int             packet_state;
  int             ret;
  packet_state=OC_PACKET_COMMENT_HDR;
  oggpackB_writeinit(&opb);
  ret=oc_state_flushheader(NULL,&packet_state,&opb,NULL,NULL,
   th_version_string(),(th_comment *)_tc,_op);
  if(ret>=0){
    /*The oggpack_buffer's lifetime ends with this function, so we have to
       copy out the packet contents.
      Presumably the application knows it is supposed to free this.
      This part works nothing like the Vorbis API, and the documentation on it
       has been wrong for some time, claiming libtheora owned the memory.*/
    buf=_ogg_malloc(_op->bytes);
    if(buf==NULL){
      _op->packet=NULL;
      ret=TH_EFAULT;
    }
    else{
      memcpy(buf,_op->packet,_op->bytes);
      _op->packet=buf;
      ret=0;
    }
  }
  oggpack_writeclear(&opb);
  return ret;
}

int theora_encode_tables(theora_state *_te,ogg_packet *_op){
  oc_enc_ctx     *enc;
  th_api_wrapper *api;
  int             ret;
  api=(th_api_wrapper *)_te->i->codec_setup;
  enc=api->encode;
  /*If we've already started encoding, fail.*/
  if(enc->packet_state>OC_PACKET_EMPTY||enc->state.granpos!=0){
    return TH_EINVAL;
  }
  /*Reset the state to make sure we output a setup packet.*/
  enc->packet_state=OC_PACKET_SETUP_HDR;
  ret=th_encode_flushheader(api->encode,NULL,_op);
  return ret>=0?0:ret;
}
