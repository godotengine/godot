/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE libopusfile SOFTWARE CODEC SOURCE CODE. *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE libopusfile SOURCE CODE IS (C) COPYRIGHT 2012                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************/
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "internal.h"
#include <limits.h>
#include <string.h>

static unsigned op_parse_uint16le(const unsigned char *_data){
  return _data[0]|_data[1]<<8;
}

static int op_parse_int16le(const unsigned char *_data){
  int ret;
  ret=_data[0]|_data[1]<<8;
  return (ret^0x8000)-0x8000;
}

static opus_uint32 op_parse_uint32le(const unsigned char *_data){
  return _data[0]|(opus_uint32)_data[1]<<8|
   (opus_uint32)_data[2]<<16|(opus_uint32)_data[3]<<24;
}

static opus_uint32 op_parse_uint32be(const unsigned char *_data){
  return _data[3]|(opus_uint32)_data[2]<<8|
   (opus_uint32)_data[1]<<16|(opus_uint32)_data[0]<<24;
}

int opus_head_parse(OpusHead *_head,const unsigned char *_data,size_t _len){
  OpusHead head;
  if(_len<8)return OP_ENOTFORMAT;
  if(memcmp(_data,"OpusHead",8)!=0)return OP_ENOTFORMAT;
  if(_len<9)return OP_EBADHEADER;
  head.version=_data[8];
  if(head.version>15)return OP_EVERSION;
  if(_len<19)return OP_EBADHEADER;
  head.channel_count=_data[9];
  head.pre_skip=op_parse_uint16le(_data+10);
  head.input_sample_rate=op_parse_uint32le(_data+12);
  head.output_gain=op_parse_int16le(_data+16);
  head.mapping_family=_data[18];
  if(head.mapping_family==0){
    if(head.channel_count<1||head.channel_count>2)return OP_EBADHEADER;
    if(head.version<=1&&_len>19)return OP_EBADHEADER;
    head.stream_count=1;
    head.coupled_count=head.channel_count-1;
    if(_head!=NULL){
      _head->mapping[0]=0;
      _head->mapping[1]=1;
    }
  }
  else if(head.mapping_family==1){
    size_t size;
    int    ci;
    if(head.channel_count<1||head.channel_count>8)return OP_EBADHEADER;
    size=21+head.channel_count;
    if(_len<size||head.version<=1&&_len>size)return OP_EBADHEADER;
    head.stream_count=_data[19];
    if(head.stream_count<1)return OP_EBADHEADER;
    head.coupled_count=_data[20];
    if(head.coupled_count>head.stream_count)return OP_EBADHEADER;
    for(ci=0;ci<head.channel_count;ci++){
      if(_data[21+ci]>=head.stream_count+head.coupled_count
       &&_data[21+ci]!=255){
        return OP_EBADHEADER;
      }
    }
    if(_head!=NULL)memcpy(_head->mapping,_data+21,head.channel_count);
  }
  /*General purpose players should not attempt to play back content with
     channel mapping family 255.*/
  else if(head.mapping_family==255)return OP_EIMPL;
  /*No other channel mapping families are currently defined.*/
  else return OP_EBADHEADER;
  if(_head!=NULL)memcpy(_head,&head,head.mapping-(unsigned char *)&head);
  return 0;
}

void opus_tags_init(OpusTags *_tags){
  memset(_tags,0,sizeof(*_tags));
}

void opus_tags_clear(OpusTags *_tags){
  int ncomments;
  int ci;
  ncomments=_tags->comments;
  if(_tags->user_comments!=NULL)ncomments++;
  for(ci=ncomments;ci-->0;)_ogg_free(_tags->user_comments[ci]);
  _ogg_free(_tags->user_comments);
  _ogg_free(_tags->comment_lengths);
  _ogg_free(_tags->vendor);
}

/*Ensure there's room for up to _ncomments comments.*/
static int op_tags_ensure_capacity(OpusTags *_tags,size_t _ncomments){
  char   **user_comments;
  int     *comment_lengths;
  int      cur_ncomments;
  char    *binary_suffix_data;
  int      binary_suffix_len;
  size_t   size;
  if(OP_UNLIKELY(_ncomments>=(size_t)INT_MAX))return OP_EFAULT;
  size=sizeof(*_tags->comment_lengths)*(_ncomments+1);
  if(size/sizeof(*_tags->comment_lengths)!=_ncomments+1)return OP_EFAULT;
  cur_ncomments=_tags->comments;
  comment_lengths=_tags->comment_lengths;
  binary_suffix_len=comment_lengths==NULL?0:comment_lengths[cur_ncomments];
  comment_lengths=(int *)_ogg_realloc(_tags->comment_lengths,size);
  if(OP_UNLIKELY(comment_lengths==NULL))return OP_EFAULT;
  comment_lengths[_ncomments]=binary_suffix_len;
  _tags->comment_lengths=comment_lengths;
  size=sizeof(*_tags->user_comments)*(_ncomments+1);
  if(size/sizeof(*_tags->user_comments)!=_ncomments+1)return OP_EFAULT;
  user_comments=_tags->user_comments;
  binary_suffix_data=user_comments==NULL?NULL:user_comments[cur_ncomments];
  user_comments=(char **)_ogg_realloc(_tags->user_comments,size);
  if(OP_UNLIKELY(user_comments==NULL))return OP_EFAULT;
  user_comments[_ncomments]=binary_suffix_data;
  _tags->user_comments=user_comments;
  return 0;
}

/*Duplicate a (possibly non-NUL terminated) string with a known length.*/
static char *op_strdup_with_len(const char *_s,size_t _len){
  size_t  size;
  char   *ret;
  size=sizeof(*ret)*(_len+1);
  if(OP_UNLIKELY(size<_len))return NULL;
  ret=(char *)_ogg_malloc(size);
  if(OP_LIKELY(ret!=NULL)){
    ret=(char *)memcpy(ret,_s,sizeof(*ret)*_len);
    ret[_len]='\0';
  }
  return ret;
}

/*The actual implementation of opus_tags_parse().
  Unlike the public API, this function requires _tags to already be
   initialized, modifies its contents before success is guaranteed, and assumes
   the caller will clear it on error.*/
static int opus_tags_parse_impl(OpusTags *_tags,
 const unsigned char *_data,size_t _len){
  opus_uint32 count;
  size_t      len;
  int         ncomments;
  int         ci;
  len=_len;
  if(len<8)return OP_ENOTFORMAT;
  if(memcmp(_data,"OpusTags",8)!=0)return OP_ENOTFORMAT;
  if(len<16)return OP_EBADHEADER;
  _data+=8;
  len-=8;
  count=op_parse_uint32le(_data);
  _data+=4;
  len-=4;
  if(count>len)return OP_EBADHEADER;
  if(_tags!=NULL){
    _tags->vendor=op_strdup_with_len((char *)_data,count);
    if(_tags->vendor==NULL)return OP_EFAULT;
  }
  _data+=count;
  len-=count;
  if(len<4)return OP_EBADHEADER;
  count=op_parse_uint32le(_data);
  _data+=4;
  len-=4;
  /*Check to make sure there's minimally sufficient data left in the packet.*/
  if(count>len>>2)return OP_EBADHEADER;
  /*Check for overflow (the API limits this to an int).*/
  if(count>(opus_uint32)INT_MAX-1)return OP_EFAULT;
  if(_tags!=NULL){
    int ret;
    ret=op_tags_ensure_capacity(_tags,count);
    if(ret<0)return ret;
  }
  ncomments=(int)count;
  for(ci=0;ci<ncomments;ci++){
    /*Check to make sure there's minimally sufficient data left in the packet.*/
    if((size_t)(ncomments-ci)>len>>2)return OP_EBADHEADER;
    count=op_parse_uint32le(_data);
    _data+=4;
    len-=4;
    if(count>len)return OP_EBADHEADER;
    /*Check for overflow (the API limits this to an int).*/
    if(count>(opus_uint32)INT_MAX)return OP_EFAULT;
    if(_tags!=NULL){
      _tags->user_comments[ci]=op_strdup_with_len((char *)_data,count);
      if(_tags->user_comments[ci]==NULL)return OP_EFAULT;
      _tags->comment_lengths[ci]=(int)count;
      _tags->comments=ci+1;
      /*Needed by opus_tags_clear() if we fail before parsing the (optional)
         binary metadata.*/
      _tags->user_comments[ci+1]=NULL;
    }
    _data+=count;
    len-=count;
  }
  if(len>0&&(_data[0]&1)){
    if(len>(opus_uint32)INT_MAX)return OP_EFAULT;
    if(_tags!=NULL){
      _tags->user_comments[ncomments]=(char *)_ogg_malloc(len);
      if(OP_UNLIKELY(_tags->user_comments[ncomments]==NULL))return OP_EFAULT;
      memcpy(_tags->user_comments[ncomments],_data,len);
      _tags->comment_lengths[ncomments]=(int)len;
    }
  }
  return 0;
}

int opus_tags_parse(OpusTags *_tags,const unsigned char *_data,size_t _len){
  if(_tags!=NULL){
    OpusTags tags;
    int      ret;
    opus_tags_init(&tags);
    ret=opus_tags_parse_impl(&tags,_data,_len);
    if(ret<0)opus_tags_clear(&tags);
    else *_tags=*&tags;
    return ret;
  }
  else return opus_tags_parse_impl(NULL,_data,_len);
}

/*The actual implementation of opus_tags_copy().
  Unlike the public API, this function requires _dst to already be
   initialized, modifies its contents before success is guaranteed, and assumes
   the caller will clear it on error.*/
static int opus_tags_copy_impl(OpusTags *_dst,const OpusTags *_src){
  char *vendor;
  int   ncomments;
  int   ret;
  int   ci;
  vendor=_src->vendor;
  _dst->vendor=op_strdup_with_len(vendor,strlen(vendor));
  if(OP_UNLIKELY(_dst->vendor==NULL))return OP_EFAULT;
  ncomments=_src->comments;
  ret=op_tags_ensure_capacity(_dst,ncomments);
  if(OP_UNLIKELY(ret<0))return ret;
  for(ci=0;ci<ncomments;ci++){
    int len;
    len=_src->comment_lengths[ci];
    OP_ASSERT(len>=0);
    _dst->user_comments[ci]=op_strdup_with_len(_src->user_comments[ci],len);
    if(OP_UNLIKELY(_dst->user_comments[ci]==NULL))return OP_EFAULT;
    _dst->comment_lengths[ci]=len;
    _dst->comments=ci+1;
  }
  if(_src->comment_lengths!=NULL){
    int len;
    len=_src->comment_lengths[ncomments];
    if(len>0){
      _dst->user_comments[ncomments]=(char *)_ogg_malloc(len);
      if(OP_UNLIKELY(_dst->user_comments[ncomments]==NULL))return OP_EFAULT;
      memcpy(_dst->user_comments[ncomments],_src->user_comments[ncomments],len);
      _dst->comment_lengths[ncomments]=len;
    }
  }
  return 0;
}

int opus_tags_copy(OpusTags *_dst,const OpusTags *_src){
  OpusTags dst;
  int      ret;
  opus_tags_init(&dst);
  ret=opus_tags_copy_impl(&dst,_src);
  if(OP_UNLIKELY(ret<0))opus_tags_clear(&dst);
  else *_dst=*&dst;
  return 0;
}

int opus_tags_add(OpusTags *_tags,const char *_tag,const char *_value){
  char *comment;
  int   tag_len;
  int   value_len;
  int   ncomments;
  int   ret;
  ncomments=_tags->comments;
  ret=op_tags_ensure_capacity(_tags,ncomments+1);
  if(OP_UNLIKELY(ret<0))return ret;
  tag_len=strlen(_tag);
  value_len=strlen(_value);
  /*+2 for '=' and '\0'.*/
  comment=(char *)_ogg_malloc(sizeof(*comment)*(tag_len+value_len+2));
  if(OP_UNLIKELY(comment==NULL))return OP_EFAULT;
  memcpy(comment,_tag,sizeof(*comment)*tag_len);
  comment[tag_len]='=';
  memcpy(comment+tag_len+1,_value,sizeof(*comment)*(value_len+1));
  _tags->user_comments[ncomments]=comment;
  _tags->comment_lengths[ncomments]=tag_len+value_len+1;
  _tags->comments=ncomments+1;
  return 0;
}

int opus_tags_add_comment(OpusTags *_tags,const char *_comment){
  char *comment;
  int   comment_len;
  int   ncomments;
  int   ret;
  ncomments=_tags->comments;
  ret=op_tags_ensure_capacity(_tags,ncomments+1);
  if(OP_UNLIKELY(ret<0))return ret;
  comment_len=(int)strlen(_comment);
  comment=op_strdup_with_len(_comment,comment_len);
  if(OP_UNLIKELY(comment==NULL))return OP_EFAULT;
  _tags->user_comments[ncomments]=comment;
  _tags->comment_lengths[ncomments]=comment_len;
  _tags->comments=ncomments+1;
  return 0;
}

int opus_tags_set_binary_suffix(OpusTags *_tags,
 const unsigned char *_data,int _len){
  unsigned char *binary_suffix_data;
  int            ncomments;
  int            ret;
  if(_len<0||_len>0&&(_data==NULL||!(_data[0]&1)))return OP_EINVAL;
  ncomments=_tags->comments;
  ret=op_tags_ensure_capacity(_tags,ncomments);
  if(OP_UNLIKELY(ret<0))return ret;
  binary_suffix_data=
   (unsigned char *)_ogg_realloc(_tags->user_comments[ncomments],_len);
  if(OP_UNLIKELY(binary_suffix_data==NULL))return OP_EFAULT;
  memcpy(binary_suffix_data,_data,_len);
  _tags->user_comments[ncomments]=(char *)binary_suffix_data;
  _tags->comment_lengths[ncomments]=_len;
  return 0;
}

int opus_tagcompare(const char *_tag_name,const char *_comment){
  return opus_tagncompare(_tag_name,strlen(_tag_name),_comment);
}

int opus_tagncompare(const char *_tag_name,int _tag_len,const char *_comment){
  int ret;
  OP_ASSERT(_tag_len>=0);
  ret=op_strncasecmp(_tag_name,_comment,_tag_len);
  return ret?ret:'='-_comment[_tag_len];
}

const char *opus_tags_query(const OpusTags *_tags,const char *_tag,int _count){
  char **user_comments;
  int    tag_len;
  int    found;
  int    ncomments;
  int    ci;
  tag_len=strlen(_tag);
  ncomments=_tags->comments;
  user_comments=_tags->user_comments;
  found=0;
  for(ci=0;ci<ncomments;ci++){
    if(!opus_tagncompare(_tag,tag_len,user_comments[ci])){
      /*We return a pointer to the data, not a copy.*/
      if(_count==found++)return user_comments[ci]+tag_len+1;
    }
  }
  /*Didn't find anything.*/
  return NULL;
}

int opus_tags_query_count(const OpusTags *_tags,const char *_tag){
  char **user_comments;
  int    tag_len;
  int    found;
  int    ncomments;
  int    ci;
  tag_len=strlen(_tag);
  ncomments=_tags->comments;
  user_comments=_tags->user_comments;
  found=0;
  for(ci=0;ci<ncomments;ci++){
    if(!opus_tagncompare(_tag,tag_len,user_comments[ci]))found++;
  }
  return found;
}

const unsigned char *opus_tags_get_binary_suffix(const OpusTags *_tags,
 int *_len){
  int ncomments;
  int len;
  ncomments=_tags->comments;
  len=_tags->comment_lengths==NULL?0:_tags->comment_lengths[ncomments];
  *_len=len;
  OP_ASSERT(len==0||_tags->user_comments!=NULL);
  return len>0?(const unsigned char *)_tags->user_comments[ncomments]:NULL;
}

static int opus_tags_get_gain(const OpusTags *_tags,int *_gain_q8,
 const char *_tag_name,size_t _tag_len){
  char **comments;
  int    ncomments;
  int    ci;
  comments=_tags->user_comments;
  ncomments=_tags->comments;
  /*Look for the first valid tag with the name _tag_name and use that.*/
  for(ci=0;ci<ncomments;ci++){
    if(opus_tagncompare(_tag_name,_tag_len,comments[ci])==0){
      char       *p;
      opus_int32  gain_q8;
      int         negative;
      p=comments[ci]+_tag_len+1;
      negative=0;
      if(*p=='-'){
        negative=-1;
        p++;
      }
      else if(*p=='+')p++;
      gain_q8=0;
      while(*p>='0'&&*p<='9'){
        gain_q8=10*gain_q8+*p-'0';
        if(gain_q8>32767-negative)break;
        p++;
      }
      /*This didn't look like a signed 16-bit decimal integer.
        Not a valid gain tag.*/
      if(*p!='\0')continue;
      *_gain_q8=(int)(gain_q8+negative^negative);
      return 0;
    }
  }
  return OP_FALSE;
}

int opus_tags_get_album_gain(const OpusTags *_tags,int *_gain_q8){
  return opus_tags_get_gain(_tags,_gain_q8,"R128_ALBUM_GAIN",15);
}

int opus_tags_get_track_gain(const OpusTags *_tags,int *_gain_q8){
  return opus_tags_get_gain(_tags,_gain_q8,"R128_TRACK_GAIN",15);
}

static int op_is_jpeg(const unsigned char *_buf,size_t _buf_sz){
  return _buf_sz>=11&&memcmp(_buf,"\xFF\xD8\xFF\xE0",4)==0
   &&(_buf[4]<<8|_buf[5])>=16&&memcmp(_buf+6,"JFIF",5)==0;
}

/*Tries to extract the width, height, bits per pixel, and palette size of a
   JPEG.
  On failure, simply leaves its outputs unmodified.*/
static void op_extract_jpeg_params(const unsigned char *_buf,size_t _buf_sz,
 opus_uint32 *_width,opus_uint32 *_height,
 opus_uint32 *_depth,opus_uint32 *_colors,int *_has_palette){
  if(op_is_jpeg(_buf,_buf_sz)){
    size_t offs;
    offs=2;
    for(;;){
      size_t segment_len;
      int    marker;
      while(offs<_buf_sz&&_buf[offs]!=0xFF)offs++;
      while(offs<_buf_sz&&_buf[offs]==0xFF)offs++;
      marker=_buf[offs];
      offs++;
      /*If we hit EOI* (end of image), or another SOI* (start of image),
         or SOS (start of scan), then stop now.*/
      if(offs>=_buf_sz||(marker>=0xD8&&marker<=0xDA))break;
      /*RST* (restart markers): skip (no segment length).*/
      else if(marker>=0xD0&&marker<=0xD7)continue;
      /*Read the length of the marker segment.*/
      if(_buf_sz-offs<2)break;
      segment_len=_buf[offs]<<8|_buf[offs+1];
      if(segment_len<2||_buf_sz-offs<segment_len)break;
      if(marker==0xC0||(marker>0xC0&&marker<0xD0&&(marker&3)!=0)){
        /*Found a SOFn (start of frame) marker segment:*/
        if(segment_len>=8){
          *_height=_buf[offs+3]<<8|_buf[offs+4];
          *_width=_buf[offs+5]<<8|_buf[offs+6];
          *_depth=_buf[offs+2]*_buf[offs+7];
          *_colors=0;
          *_has_palette=0;
        }
        break;
      }
      /*Other markers: skip the whole marker segment.*/
      offs+=segment_len;
    }
  }
}

static int op_is_png(const unsigned char *_buf,size_t _buf_sz){
  return _buf_sz>=8&&memcmp(_buf,"\x89PNG\x0D\x0A\x1A\x0A",8)==0;
}

/*Tries to extract the width, height, bits per pixel, and palette size of a
   PNG.
  On failure, simply leaves its outputs unmodified.*/
static void op_extract_png_params(const unsigned char *_buf,size_t _buf_sz,
 opus_uint32 *_width,opus_uint32 *_height,
 opus_uint32 *_depth,opus_uint32 *_colors,int *_has_palette){
  if(op_is_png(_buf,_buf_sz)){
    size_t offs;
    offs=8;
    while(_buf_sz-offs>=12){
      ogg_uint32_t chunk_len;
      chunk_len=op_parse_uint32be(_buf+offs);
      if(chunk_len>_buf_sz-(offs+12))break;
      else if(chunk_len==13&&memcmp(_buf+offs+4,"IHDR",4)==0){
        int color_type;
        *_width=op_parse_uint32be(_buf+offs+8);
        *_height=op_parse_uint32be(_buf+offs+12);
        color_type=_buf[offs+17];
        if(color_type==3){
          *_depth=24;
          *_has_palette=1;
        }
        else{
          int sample_depth;
          sample_depth=_buf[offs+16];
          if(color_type==0)*_depth=sample_depth;
          else if(color_type==2)*_depth=sample_depth*3;
          else if(color_type==4)*_depth=sample_depth*2;
          else if(color_type==6)*_depth=sample_depth*4;
          *_colors=0;
          *_has_palette=0;
          break;
        }
      }
      else if(*_has_palette>0&&memcmp(_buf+offs+4,"PLTE",4)==0){
        *_colors=chunk_len/3;
        break;
      }
      offs+=12+chunk_len;
    }
  }
}

static int op_is_gif(const unsigned char *_buf,size_t _buf_sz){
  return _buf_sz>=6&&(memcmp(_buf,"GIF87a",6)==0||memcmp(_buf,"GIF89a",6)==0);
}

/*Tries to extract the width, height, bits per pixel, and palette size of a
   GIF.
  On failure, simply leaves its outputs unmodified.*/
static void op_extract_gif_params(const unsigned char *_buf,size_t _buf_sz,
 opus_uint32 *_width,opus_uint32 *_height,
 opus_uint32 *_depth,opus_uint32 *_colors,int *_has_palette){
  if(op_is_gif(_buf,_buf_sz)&&_buf_sz>=14){
    *_width=_buf[6]|_buf[7]<<8;
    *_height=_buf[8]|_buf[9]<<8;
    /*libFLAC hard-codes the depth to 24.*/
    *_depth=24;
    *_colors=1<<((_buf[10]&7)+1);
    *_has_palette=1;
  }
}

/*The actual implementation of opus_picture_tag_parse().
  Unlike the public API, this function requires _pic to already be
   initialized, modifies its contents before success is guaranteed, and assumes
   the caller will clear it on error.*/
static int opus_picture_tag_parse_impl(OpusPictureTag *_pic,const char *_tag,
 unsigned char *_buf,size_t _buf_sz,size_t _base64_sz){
  opus_int32   picture_type;
  opus_uint32  mime_type_length;
  char        *mime_type;
  opus_uint32  description_length;
  char        *description;
  opus_uint32  width;
  opus_uint32  height;
  opus_uint32  depth;
  opus_uint32  colors;
  opus_uint32  data_length;
  opus_uint32  file_width;
  opus_uint32  file_height;
  opus_uint32  file_depth;
  opus_uint32  file_colors;
  int          format;
  int          has_palette;
  int          colors_set;
  size_t       i;
  /*Decode the BASE64 data.*/
  for(i=0;i<_base64_sz;i++){
    opus_uint32 value;
    int         j;
    value=0;
    for(j=0;j<4;j++){
      unsigned c;
      unsigned d;
      c=(unsigned char)_tag[4*i+j];
      if(c=='+')d=62;
      else if(c=='/')d=63;
      else if(c>='0'&&c<='9')d=52+c-'0';
      else if(c>='a'&&c<='z')d=26+c-'a';
      else if(c>='A'&&c<='Z')d=c-'A';
      else if(c=='='&&3*i+j>_buf_sz)d=0;
      else return OP_ENOTFORMAT;
      value=value<<6|d;
    }
    _buf[3*i]=(unsigned char)(value>>16);
    if(3*i+1<_buf_sz){
      _buf[3*i+1]=(unsigned char)(value>>8);
      if(3*i+2<_buf_sz)_buf[3*i+2]=(unsigned char)value;
    }
  }
  i=0;
  picture_type=op_parse_uint32be(_buf+i);
  i+=4;
  /*Extract the MIME type.*/
  mime_type_length=op_parse_uint32be(_buf+i);
  i+=4;
  if(mime_type_length>_buf_sz-32)return OP_ENOTFORMAT;
  mime_type=(char *)_ogg_malloc(sizeof(*_pic->mime_type)*(mime_type_length+1));
  if(mime_type==NULL)return OP_EFAULT;
  memcpy(mime_type,_buf+i,sizeof(*mime_type)*mime_type_length);
  mime_type[mime_type_length]='\0';
  _pic->mime_type=mime_type;
  i+=mime_type_length;
  /*Extract the description string.*/
  description_length=op_parse_uint32be(_buf+i);
  i+=4;
  if(description_length>_buf_sz-mime_type_length-32)return OP_ENOTFORMAT;
  description=
   (char *)_ogg_malloc(sizeof(*_pic->mime_type)*(description_length+1));
  if(description==NULL)return OP_EFAULT;
  memcpy(description,_buf+i,sizeof(*description)*description_length);
  description[description_length]='\0';
  _pic->description=description;
  i+=description_length;
  /*Extract the remaining fields.*/
  width=op_parse_uint32be(_buf+i);
  i+=4;
  height=op_parse_uint32be(_buf+i);
  i+=4;
  depth=op_parse_uint32be(_buf+i);
  i+=4;
  colors=op_parse_uint32be(_buf+i);
  i+=4;
  /*If one of these is set, they all must be, but colors==0 is a valid value.*/
  colors_set=width!=0||height!=0||depth!=0||colors!=0;
  if((width==0||height==0||depth==0)&&colors_set)return OP_ENOTFORMAT;
  data_length=op_parse_uint32be(_buf+i);
  i+=4;
  if(data_length>_buf_sz-i)return OP_ENOTFORMAT;
  /*Trim extraneous data so we don't copy it below.*/
  _buf_sz=i+data_length;
  /*Attempt to determine the image format.*/
  format=OP_PIC_FORMAT_UNKNOWN;
  if(mime_type_length==3&&strcmp(mime_type,"-->")==0){
    format=OP_PIC_FORMAT_URL;
    /*Picture type 1 must be a 32x32 PNG.*/
    if(picture_type==1&&(width!=0||height!=0)&&(width!=32||height!=32)){
      return OP_ENOTFORMAT;
    }
    /*Append a terminating NUL for the convenience of our callers.*/
    _buf[_buf_sz++]='\0';
  }
  else{
    if(mime_type_length==10
     &&op_strncasecmp(mime_type,"image/jpeg",mime_type_length)==0){
      if(op_is_jpeg(_buf+i,data_length))format=OP_PIC_FORMAT_JPEG;
    }
    else if(mime_type_length==9
     &&op_strncasecmp(mime_type,"image/png",mime_type_length)==0){
      if(op_is_png(_buf+i,data_length))format=OP_PIC_FORMAT_PNG;
    }
    else if(mime_type_length==9
     &&op_strncasecmp(mime_type,"image/gif",mime_type_length)==0){
      if(op_is_gif(_buf+i,data_length))format=OP_PIC_FORMAT_GIF;
    }
    else if(mime_type_length==0||(mime_type_length==6
     &&op_strncasecmp(mime_type,"image/",mime_type_length)==0)){
      if(op_is_jpeg(_buf+i,data_length))format=OP_PIC_FORMAT_JPEG;
      else if(op_is_png(_buf+i,data_length))format=OP_PIC_FORMAT_PNG;
      else if(op_is_gif(_buf+i,data_length))format=OP_PIC_FORMAT_GIF;
    }
    file_width=file_height=file_depth=file_colors=0;
    has_palette=-1;
    switch(format){
      case OP_PIC_FORMAT_JPEG:{
        op_extract_jpeg_params(_buf+i,data_length,
         &file_width,&file_height,&file_depth,&file_colors,&has_palette);
      }break;
      case OP_PIC_FORMAT_PNG:{
        op_extract_png_params(_buf+i,data_length,
         &file_width,&file_height,&file_depth,&file_colors,&has_palette);
      }break;
      case OP_PIC_FORMAT_GIF:{
        op_extract_gif_params(_buf+i,data_length,
         &file_width,&file_height,&file_depth,&file_colors,&has_palette);
      }break;
    }
    if(has_palette>=0){
      /*If we successfully extracted these parameters from the image, override
         any declared values.*/
      width=file_width;
      height=file_height;
      depth=file_depth;
      colors=file_colors;
    }
    /*Picture type 1 must be a 32x32 PNG.*/
    if(picture_type==1&&(format!=OP_PIC_FORMAT_PNG||width!=32||height!=32)){
      return OP_ENOTFORMAT;
    }
  }
  /*Adjust _buf_sz instead of using data_length to capture the terminating NUL
     for URLs.*/
  _buf_sz-=i;
  memmove(_buf,_buf+i,sizeof(*_buf)*_buf_sz);
  _buf=(unsigned char *)_ogg_realloc(_buf,_buf_sz);
  if(_buf_sz>0&&_buf==NULL)return OP_EFAULT;
  _pic->type=picture_type;
  _pic->width=width;
  _pic->height=height;
  _pic->depth=depth;
  _pic->colors=colors;
  _pic->data_length=data_length;
  _pic->data=_buf;
  _pic->format=format;
  return 0;
}

int opus_picture_tag_parse(OpusPictureTag *_pic,const char *_tag){
  OpusPictureTag  pic;
  unsigned char  *buf;
  size_t          base64_sz;
  size_t          buf_sz;
  size_t          tag_length;
  int             ret;
  if(opus_tagncompare("METADATA_BLOCK_PICTURE",22,_tag)==0)_tag+=23;
  /*Figure out how much BASE64-encoded data we have.*/
  tag_length=strlen(_tag);
  if(tag_length&3)return OP_ENOTFORMAT;
  base64_sz=tag_length>>2;
  buf_sz=3*base64_sz;
  if(buf_sz<32)return OP_ENOTFORMAT;
  if(_tag[tag_length-1]=='=')buf_sz--;
  if(_tag[tag_length-2]=='=')buf_sz--;
  if(buf_sz<32)return OP_ENOTFORMAT;
  /*Allocate an extra byte to allow appending a terminating NUL to URL data.*/
  buf=(unsigned char *)_ogg_malloc(sizeof(*buf)*(buf_sz+1));
  if(buf==NULL)return OP_EFAULT;
  opus_picture_tag_init(&pic);
  ret=opus_picture_tag_parse_impl(&pic,_tag,buf,buf_sz,base64_sz);
  if(ret<0){
    opus_picture_tag_clear(&pic);
    _ogg_free(buf);
  }
  else *_pic=*&pic;
  return ret;
}

void opus_picture_tag_init(OpusPictureTag *_pic){
  memset(_pic,0,sizeof(*_pic));
}

void opus_picture_tag_clear(OpusPictureTag *_pic){
  _ogg_free(_pic->description);
  _ogg_free(_pic->mime_type);
  _ogg_free(_pic->data);
}
