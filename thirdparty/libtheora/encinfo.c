#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "enquant.h"
#include "huffenc.h"



/*Packs a series of octets from a given byte array into the pack buffer.
  _opb: The pack buffer to store the octets in.
  _buf: The byte array containing the bytes to pack.
  _len: The number of octets to pack.*/
static void oc_pack_octets(oggpack_buffer *_opb,const char *_buf,int _len){
  int i;
  for(i=0;i<_len;i++)oggpackB_write(_opb,_buf[i],8);
}



int oc_state_flushheader(oc_theora_state *_state,int *_packet_state,
 oggpack_buffer *_opb,const th_quant_info *_qinfo,
 const th_huff_code _codes[TH_NHUFFMAN_TABLES][TH_NDCT_TOKENS],
 const char *_vendor,th_comment *_tc,ogg_packet *_op){
  unsigned char *packet;
  int            b_o_s;
  if(_op==NULL)return TH_EFAULT;
  switch(*_packet_state){
    /*Codec info header.*/
    case OC_PACKET_INFO_HDR:{
      if(_state==NULL)return TH_EFAULT;
      oggpackB_reset(_opb);
      /*Mark this packet as the info header.*/
      oggpackB_write(_opb,0x80,8);
      /*Write the codec string.*/
      oc_pack_octets(_opb,"theora",6);
      /*Write the codec bitstream version.*/
      oggpackB_write(_opb,TH_VERSION_MAJOR,8);
      oggpackB_write(_opb,TH_VERSION_MINOR,8);
      oggpackB_write(_opb,TH_VERSION_SUB,8);
      /*Describe the encoded frame.*/
      oggpackB_write(_opb,_state->info.frame_width>>4,16);
      oggpackB_write(_opb,_state->info.frame_height>>4,16);
      oggpackB_write(_opb,_state->info.pic_width,24);
      oggpackB_write(_opb,_state->info.pic_height,24);
      oggpackB_write(_opb,_state->info.pic_x,8);
      oggpackB_write(_opb,_state->info.pic_y,8);
      oggpackB_write(_opb,_state->info.fps_numerator,32);
      oggpackB_write(_opb,_state->info.fps_denominator,32);
      oggpackB_write(_opb,_state->info.aspect_numerator,24);
      oggpackB_write(_opb,_state->info.aspect_denominator,24);
      oggpackB_write(_opb,_state->info.colorspace,8);
      oggpackB_write(_opb,_state->info.target_bitrate,24);
      oggpackB_write(_opb,_state->info.quality,6);
      oggpackB_write(_opb,_state->info.keyframe_granule_shift,5);
      oggpackB_write(_opb,_state->info.pixel_fmt,2);
      /*Spare configuration bits.*/
      oggpackB_write(_opb,0,3);
      b_o_s=1;
    }break;
    /*Comment header.*/
    case OC_PACKET_COMMENT_HDR:{
      int vendor_len;
      int i;
      if(_tc==NULL)return TH_EFAULT;
      vendor_len=strlen(_vendor);
      oggpackB_reset(_opb);
      /*Mark this packet as the comment header.*/
      oggpackB_write(_opb,0x81,8);
      /*Write the codec string.*/
      oc_pack_octets(_opb,"theora",6);
      /*Write the vendor string.*/
      oggpack_write(_opb,vendor_len,32);
      oc_pack_octets(_opb,_vendor,vendor_len);
      oggpack_write(_opb,_tc->comments,32);
      for(i=0;i<_tc->comments;i++){
        if(_tc->user_comments[i]!=NULL){
          oggpack_write(_opb,_tc->comment_lengths[i],32);
          oc_pack_octets(_opb,_tc->user_comments[i],_tc->comment_lengths[i]);
        }
        else oggpack_write(_opb,0,32);
      }
      b_o_s=0;
    }break;
    /*Codec setup header.*/
    case OC_PACKET_SETUP_HDR:{
      int ret;
      oggpackB_reset(_opb);
      /*Mark this packet as the setup header.*/
      oggpackB_write(_opb,0x82,8);
      /*Write the codec string.*/
      oc_pack_octets(_opb,"theora",6);
      /*Write the quantizer tables.*/
      oc_quant_params_pack(_opb,_qinfo);
      /*Write the huffman codes.*/
      ret=oc_huff_codes_pack(_opb,_codes);
      /*This should never happen, because we validate the tables when they
         are set.
        If you see, it's a good chance memory is being corrupted.*/
      if(ret<0)return ret;
      b_o_s=0;
    }break;
    /*No more headers to emit.*/
    default:return 0;
  }
  /*This is kind of fugly: we hand the user a buffer which they do not own.
    We will overwrite it when the next packet is output, so the user better be
     done with it by then.
    Vorbis is little better: it hands back buffers that it will free the next
     time the headers are requested, or when the encoder is cleared.
    Hopefully libogg2 will make this much cleaner.*/
  packet=oggpackB_get_buffer(_opb);
  /*If there's no packet, malloc failed while writing.*/
  if(packet==NULL)return TH_EFAULT;
  _op->packet=packet;
  _op->bytes=oggpackB_bytes(_opb);
  _op->b_o_s=b_o_s;
  _op->e_o_s=0;
  _op->granulepos=0;
  _op->packetno=*_packet_state+3;
  return ++(*_packet_state)+3;
}
