/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2015             *
 * by the Xiph.Org Foundation https://xiph.org/                     *
 *                                                                  *
 ********************************************************************

 function: stdio-based convenience library for opening/seeking/decoding

 ********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include "vorbis/codec.h"

/* we don't need or want the static callback symbols here */
#define OV_EXCLUDE_STATIC_CALLBACKS
#include "vorbis/vorbisfile.h"

#include "os.h"
#include "misc.h"

/* A 'chained bitstream' is a Vorbis bitstream that contains more than
   one logical bitstream arranged end to end (the only form of Ogg
   multiplexing allowed in a Vorbis bitstream; grouping [parallel
   multiplexing] is not allowed in Vorbis) */

/* A Vorbis file can be played beginning to end (streamed) without
   worrying ahead of time about chaining (see decoder_example.c).  If
   we have the whole file, however, and want random access
   (seeking/scrubbing) or desire to know the total length/time of a
   file, we need to account for the possibility of chaining. */

/* We can handle things a number of ways; we can determine the entire
   bitstream structure right off the bat, or find pieces on demand.
   This example determines and caches structure for the entire
   bitstream, but builds a virtual decoder on the fly when moving
   between links in the chain. */

/* There are also different ways to implement seeking.  Enough
   information exists in an Ogg bitstream to seek to
   sample-granularity positions in the output.  Or, one can seek by
   picking some portion of the stream roughly in the desired area if
   we only want coarse navigation through the stream. */

/*************************************************************************
 * Many, many internal helpers.  The intention is not to be confusing;
 * rampant duplication and monolithic function implementation would be
 * harder to understand anyway.  The high level functions are last.  Begin
 * grokking near the end of the file */

/* read a little more data from the file/pipe into the ogg_sync framer
*/
#define CHUNKSIZE 65536 /* greater-than-page-size granularity seeking */
#define READSIZE 2048 /* a smaller read size is needed for low-rate streaming. */

static long _get_data(OggVorbis_File *vf){
  errno=0;
  if(!(vf->callbacks.read_func))return(-1);
  if(vf->datasource){
    char *buffer=ogg_sync_buffer(&vf->oy,READSIZE);
    long bytes=(vf->callbacks.read_func)(buffer,1,READSIZE,vf->datasource);
    if(bytes>0)ogg_sync_wrote(&vf->oy,bytes);
    if(bytes==0 && errno)return(-1);
    return(bytes);
  }else
    return(0);
}

/* save a tiny smidge of verbosity to make the code more readable */
static int _seek_helper(OggVorbis_File *vf,ogg_int64_t offset){
  if(vf->datasource){
    /* only seek if the file position isn't already there */
    if(vf->offset != offset){
      if(!(vf->callbacks.seek_func)||
         (vf->callbacks.seek_func)(vf->datasource, offset, SEEK_SET) == -1)
        return OV_EREAD;
      vf->offset=offset;
      ogg_sync_reset(&vf->oy);
    }
  }else{
    /* shouldn't happen unless someone writes a broken callback */
    return OV_EFAULT;
  }
  return 0;
}

/* The read/seek functions track absolute position within the stream */

/* from the head of the stream, get the next page.  boundary specifies
   if the function is allowed to fetch more data from the stream (and
   how much) or only use internally buffered data.

   boundary: -1) unbounded search
              0) read no additional data; use cached only
              n) search for a new page beginning for n bytes

   return:   <0) did not find a page (OV_FALSE, OV_EOF, OV_EREAD)
              n) found a page at absolute offset n */

static ogg_int64_t _get_next_page(OggVorbis_File *vf,ogg_page *og,
                                  ogg_int64_t boundary){
  if(boundary>0)boundary+=vf->offset;
  while(1){
    long more;

    if(boundary>0 && vf->offset>=boundary)return(OV_FALSE);
    more=ogg_sync_pageseek(&vf->oy,og);

    if(more<0){
      /* skipped n bytes */
      vf->offset-=more;
    }else{
      if(more==0){
        /* send more paramedics */
        if(!boundary)return(OV_FALSE);
        {
          long ret=_get_data(vf);
          if(ret==0)return(OV_EOF);
          if(ret<0)return(OV_EREAD);
        }
      }else{
        /* got a page.  Return the offset at the page beginning,
           advance the internal offset past the page end */
        ogg_int64_t ret=vf->offset;
        vf->offset+=more;
        return(ret);

      }
    }
  }
}

/* find the latest page beginning before the passed in position. Much
   dirtier than the above as Ogg doesn't have any backward search
   linkage.  no 'readp' as it will certainly have to read. */
/* returns offset or OV_EREAD, OV_FAULT */
static ogg_int64_t _get_prev_page(OggVorbis_File *vf,ogg_int64_t begin,ogg_page *og){
  ogg_int64_t end = begin;
  ogg_int64_t ret;
  ogg_int64_t offset=-1;

  while(offset==-1){
    begin-=CHUNKSIZE;
    if(begin<0)
      begin=0;

    ret=_seek_helper(vf,begin);
    if(ret)return(ret);

    while(vf->offset<end){
      memset(og,0,sizeof(*og));
      ret=_get_next_page(vf,og,end-vf->offset);
      if(ret==OV_EREAD)return(OV_EREAD);
      if(ret<0){
        break;
      }else{
        offset=ret;
      }
    }
  }

  /* In a fully compliant, non-multiplexed stream, we'll still be
     holding the last page.  In multiplexed (or noncompliant streams),
     we will probably have to re-read the last page we saw */
  if(og->header_len==0){
    ret=_seek_helper(vf,offset);
    if(ret)return(ret);

    ret=_get_next_page(vf,og,CHUNKSIZE);
    if(ret<0)
      /* this shouldn't be possible */
      return(OV_EFAULT);
  }

  return(offset);
}

static void _add_serialno(ogg_page *og,long **serialno_list, int *n){
  long s = ogg_page_serialno(og);
  (*n)++;

  if(*serialno_list){
    *serialno_list = _ogg_realloc(*serialno_list, sizeof(**serialno_list)*(*n));
  }else{
    *serialno_list = _ogg_malloc(sizeof(**serialno_list));
  }

  (*serialno_list)[(*n)-1] = s;
}

/* returns nonzero if found */
static int _lookup_serialno(long s, long *serialno_list, int n){
  if(serialno_list){
    while(n--){
      if(*serialno_list == s) return 1;
      serialno_list++;
    }
  }
  return 0;
}

static int _lookup_page_serialno(ogg_page *og, long *serialno_list, int n){
  long s = ogg_page_serialno(og);
  return _lookup_serialno(s,serialno_list,n);
}

/* performs the same search as _get_prev_page, but prefers pages of
   the specified serial number. If a page of the specified serialno is
   spotted during the seek-back-and-read-forward, it will return the
   info of last page of the matching serial number instead of the very
   last page.  If no page of the specified serialno is seen, it will
   return the info of last page and alter *serialno.  */
static ogg_int64_t _get_prev_page_serial(OggVorbis_File *vf, ogg_int64_t begin,
                                         long *serial_list, int serial_n,
                                         int *serialno, ogg_int64_t *granpos){
  ogg_page og;
  ogg_int64_t end=begin;
  ogg_int64_t ret;

  ogg_int64_t prefoffset=-1;
  ogg_int64_t offset=-1;
  ogg_int64_t ret_serialno=-1;
  ogg_int64_t ret_gran=-1;

  while(offset==-1){
    begin-=CHUNKSIZE;
    if(begin<0)
      begin=0;

    ret=_seek_helper(vf,begin);
    if(ret)return(ret);

    while(vf->offset<end){
      ret=_get_next_page(vf,&og,end-vf->offset);
      if(ret==OV_EREAD)return(OV_EREAD);
      if(ret<0){
        break;
      }else{
        ret_serialno=ogg_page_serialno(&og);
        ret_gran=ogg_page_granulepos(&og);
        offset=ret;

        if(ret_serialno == *serialno){
          prefoffset=ret;
          *granpos=ret_gran;
        }

        if(!_lookup_serialno(ret_serialno,serial_list,serial_n)){
          /* we fell off the end of the link, which means we seeked
             back too far and shouldn't have been looking in that link
             to begin with.  If we found the preferred serial number,
             forget that we saw it. */
          prefoffset=-1;
        }
      }
    }
    /*We started from the beginning of the stream and found nothing.
      This should be impossible unless the contents of the stream changed out
      from under us after we read from it.*/
    if(!begin&&vf->offset<0)return OV_EBADLINK;
  }

  /* we're not interested in the page... just the serialno and granpos. */
  if(prefoffset>=0)return(prefoffset);

  *serialno = ret_serialno;
  *granpos = ret_gran;
  return(offset);

}

/* uses the local ogg_stream storage in vf; this is important for
   non-streaming input sources */
static int _fetch_headers(OggVorbis_File *vf,vorbis_info *vi,vorbis_comment *vc,
                          long **serialno_list, int *serialno_n,
                          ogg_page *og_ptr){
  ogg_page og;
  ogg_packet op;
  int i,ret;
  int allbos=0;

  if(!og_ptr){
    ogg_int64_t llret=_get_next_page(vf,&og,CHUNKSIZE);
    if(llret==OV_EREAD)return(OV_EREAD);
    if(llret<0)return(OV_ENOTVORBIS);
    og_ptr=&og;
  }

  vorbis_info_init(vi);
  vorbis_comment_init(vc);
  vf->ready_state=OPENED;

  /* extract the serialnos of all BOS pages + the first set of vorbis
     headers we see in the link */

  while(ogg_page_bos(og_ptr)){
    if(serialno_list){
      if(_lookup_page_serialno(og_ptr,*serialno_list,*serialno_n)){
        /* a dupe serialnumber in an initial header packet set == invalid stream */
        if(*serialno_list)_ogg_free(*serialno_list);
        *serialno_list=0;
        *serialno_n=0;
        ret=OV_EBADHEADER;
        goto bail_header;
      }

      _add_serialno(og_ptr,serialno_list,serialno_n);
    }

    if(vf->ready_state<STREAMSET){
      /* we don't have a vorbis stream in this link yet, so begin
         prospective stream setup. We need a stream to get packets */
      ogg_stream_reset_serialno(&vf->os,ogg_page_serialno(og_ptr));
      ogg_stream_pagein(&vf->os,og_ptr);

      if(ogg_stream_packetout(&vf->os,&op) > 0 &&
         vorbis_synthesis_idheader(&op)){
        /* vorbis header; continue setup */
        vf->ready_state=STREAMSET;
        if((ret=vorbis_synthesis_headerin(vi,vc,&op))){
          ret=OV_EBADHEADER;
          goto bail_header;
        }
      }
    }

    /* get next page */
    {
      ogg_int64_t llret=_get_next_page(vf,og_ptr,CHUNKSIZE);
      if(llret==OV_EREAD){
        ret=OV_EREAD;
        goto bail_header;
      }
      if(llret<0){
        ret=OV_ENOTVORBIS;
        goto bail_header;
      }

      /* if this page also belongs to our vorbis stream, submit it and break */
      if(vf->ready_state==STREAMSET &&
         vf->os.serialno == ogg_page_serialno(og_ptr)){
        ogg_stream_pagein(&vf->os,og_ptr);
        break;
      }
    }
  }

  if(vf->ready_state!=STREAMSET){
    ret = OV_ENOTVORBIS;
    goto bail_header;
  }

  while(1){

    i=0;
    while(i<2){ /* get a page loop */

      while(i<2){ /* get a packet loop */

        int result=ogg_stream_packetout(&vf->os,&op);
        if(result==0)break;
        if(result==-1){
          ret=OV_EBADHEADER;
          goto bail_header;
        }

        if((ret=vorbis_synthesis_headerin(vi,vc,&op)))
          goto bail_header;

        i++;
      }

      while(i<2){
        if(_get_next_page(vf,og_ptr,CHUNKSIZE)<0){
          ret=OV_EBADHEADER;
          goto bail_header;
        }

        /* if this page belongs to the correct stream, go parse it */
        if(vf->os.serialno == ogg_page_serialno(og_ptr)){
          ogg_stream_pagein(&vf->os,og_ptr);
          break;
        }

        /* if we never see the final vorbis headers before the link
           ends, abort */
        if(ogg_page_bos(og_ptr)){
          if(allbos){
            ret = OV_EBADHEADER;
            goto bail_header;
          }else
            allbos=1;
        }

        /* otherwise, keep looking */
      }
    }

    return 0;
  }

 bail_header:
  vorbis_info_clear(vi);
  vorbis_comment_clear(vc);
  vf->ready_state=OPENED;

  return ret;
}

/* Starting from current cursor position, get initial PCM offset of
   next page.  Consumes the page in the process without decoding
   audio, however this is only called during stream parsing upon
   seekable open. */
static ogg_int64_t _initial_pcmoffset(OggVorbis_File *vf, vorbis_info *vi){
  ogg_page    og;
  ogg_int64_t accumulated=0;
  long        lastblock=-1;
  int         result;
  int         serialno = vf->os.serialno;

  while(1){
    ogg_packet op;
    if(_get_next_page(vf,&og,-1)<0)
      break; /* should not be possible unless the file is truncated/mangled */

    if(ogg_page_bos(&og)) break;
    if(ogg_page_serialno(&og)!=serialno) continue;

    /* count blocksizes of all frames in the page */
    ogg_stream_pagein(&vf->os,&og);
    while((result=ogg_stream_packetout(&vf->os,&op))){
      if(result>0){ /* ignore holes */
        long thisblock=vorbis_packet_blocksize(vi,&op);
        if(thisblock>=0){
          if(lastblock!=-1)
            accumulated+=(lastblock+thisblock)>>2;
          lastblock=thisblock;
        }
      }
    }

    if(ogg_page_granulepos(&og)!=-1){
      /* pcm offset of last packet on the first audio page */
      accumulated= ogg_page_granulepos(&og)-accumulated;
      break;
    }
  }

  /* less than zero?  Either a corrupt file or a stream with samples
     trimmed off the beginning, a normal occurrence; in both cases set
     the offset to zero */
  if(accumulated<0)accumulated=0;

  return accumulated;
}

/* finds each bitstream link one at a time using a bisection search
   (has to begin by knowing the offset of the lb's initial page).
   Recurses for each link so it can alloc the link storage after
   finding them all, then unroll and fill the cache at the same time */
static int _bisect_forward_serialno(OggVorbis_File *vf,
                                    ogg_int64_t begin,
                                    ogg_int64_t searched,
                                    ogg_int64_t end,
                                    ogg_int64_t endgran,
                                    int endserial,
                                    long *currentno_list,
                                    int  currentnos,
                                    long m){
  ogg_int64_t pcmoffset;
  ogg_int64_t dataoffset=searched;
  ogg_int64_t endsearched=end;
  ogg_int64_t next=end;
  ogg_int64_t searchgran=-1;
  ogg_page og;
  ogg_int64_t ret,last;
  int serialno = vf->os.serialno;

  /* invariants:
     we have the headers and serialnos for the link beginning at 'begin'
     we have the offset and granpos of the last page in the file (potentially
       not a page we care about)
  */

  /* Is the last page in our list of current serialnumbers? */
  if(_lookup_serialno(endserial,currentno_list,currentnos)){

    /* last page is in the starting serialno list, so we've bisected
       down to (or just started with) a single link.  Now we need to
       find the last vorbis page belonging to the first vorbis stream
       for this link. */
    searched = end;
    while(endserial != serialno){
      endserial = serialno;
      searched=_get_prev_page_serial(vf,searched,currentno_list,currentnos,&endserial,&endgran);
    }

    vf->links=m+1;
    if(vf->offsets)_ogg_free(vf->offsets);
    if(vf->serialnos)_ogg_free(vf->serialnos);
    if(vf->dataoffsets)_ogg_free(vf->dataoffsets);

    vf->offsets=_ogg_malloc((vf->links+1)*sizeof(*vf->offsets));
    vf->vi=_ogg_realloc(vf->vi,vf->links*sizeof(*vf->vi));
    vf->vc=_ogg_realloc(vf->vc,vf->links*sizeof(*vf->vc));
    vf->serialnos=_ogg_malloc(vf->links*sizeof(*vf->serialnos));
    vf->dataoffsets=_ogg_malloc(vf->links*sizeof(*vf->dataoffsets));
    vf->pcmlengths=_ogg_malloc(vf->links*2*sizeof(*vf->pcmlengths));

    vf->offsets[m+1]=end;
    vf->offsets[m]=begin;
    vf->pcmlengths[m*2+1]=(endgran<0?0:endgran);

  }else{

    /* last page is not in the starting stream's serial number list,
       so we have multiple links.  Find where the stream that begins
       our bisection ends. */

    long *next_serialno_list=NULL;
    int next_serialnos=0;
    vorbis_info vi;
    vorbis_comment vc;
    int testserial = serialno+1;

    /* the below guards against garbage seperating the last and
       first pages of two links. */
    while(searched<endsearched){
      ogg_int64_t bisect;

      if(endsearched-searched<CHUNKSIZE){
        bisect=searched;
      }else{
        bisect=(searched+endsearched)/2;
      }

      ret=_seek_helper(vf,bisect);
      if(ret)return(ret);

      last=_get_next_page(vf,&og,-1);
      if(last==OV_EREAD)return(OV_EREAD);
      if(last<0 || !_lookup_page_serialno(&og,currentno_list,currentnos)){
        endsearched=bisect;
        if(last>=0)next=last;
      }else{
        searched=vf->offset;
      }
    }

    /* Bisection point found */
    /* for the time being, fetch end PCM offset the simple way */
    searched = next;
    while(testserial != serialno){
      testserial = serialno;
      searched = _get_prev_page_serial(vf,searched,currentno_list,currentnos,&testserial,&searchgran);
    }

    ret=_seek_helper(vf,next);
    if(ret)return(ret);

    ret=_fetch_headers(vf,&vi,&vc,&next_serialno_list,&next_serialnos,NULL);
    if(ret)return(ret);
    serialno = vf->os.serialno;
    dataoffset = vf->offset;

    /* this will consume a page, however the next bisection always
       starts with a raw seek */
    pcmoffset = _initial_pcmoffset(vf,&vi);

    ret=_bisect_forward_serialno(vf,next,vf->offset,end,endgran,endserial,
                                 next_serialno_list,next_serialnos,m+1);
    if(ret)return(ret);

    if(next_serialno_list)_ogg_free(next_serialno_list);

    vf->offsets[m+1]=next;
    vf->serialnos[m+1]=serialno;
    vf->dataoffsets[m+1]=dataoffset;

    vf->vi[m+1]=vi;
    vf->vc[m+1]=vc;

    vf->pcmlengths[m*2+1]=searchgran;
    vf->pcmlengths[m*2+2]=pcmoffset;
    vf->pcmlengths[m*2+3]-=pcmoffset;
    if(vf->pcmlengths[m*2+3]<0)vf->pcmlengths[m*2+3]=0;
  }
  return(0);
}

static int _make_decode_ready(OggVorbis_File *vf){
  if(vf->ready_state>STREAMSET)return 0;
  if(vf->ready_state<STREAMSET)return OV_EFAULT;
  if(vf->seekable){
    if(vorbis_synthesis_init(&vf->vd,vf->vi+vf->current_link))
      return OV_EBADLINK;
  }else{
    if(vorbis_synthesis_init(&vf->vd,vf->vi))
      return OV_EBADLINK;
  }
  vorbis_block_init(&vf->vd,&vf->vb);
  vf->ready_state=INITSET;
  vf->bittrack=0.f;
  vf->samptrack=0.f;
  return 0;
}

static int _open_seekable2(OggVorbis_File *vf){
  ogg_int64_t dataoffset=vf->dataoffsets[0],end,endgran=-1;
  int endserial=vf->os.serialno;
  int serialno=vf->os.serialno;

  /* we're partially open and have a first link header state in
     storage in vf */

  /* fetch initial PCM offset */
  ogg_int64_t pcmoffset = _initial_pcmoffset(vf,vf->vi);

  /* we can seek, so set out learning all about this file */
  if(vf->callbacks.seek_func && vf->callbacks.tell_func){
    (vf->callbacks.seek_func)(vf->datasource,0,SEEK_END);
    vf->offset=vf->end=(vf->callbacks.tell_func)(vf->datasource);
  }else{
    vf->offset=vf->end=-1;
  }

  /* If seek_func is implemented, tell_func must also be implemented */
  if(vf->end==-1) return(OV_EINVAL);

  /* Get the offset of the last page of the physical bitstream, or, if
     we're lucky the last vorbis page of this link as most OggVorbis
     files will contain a single logical bitstream */
  end=_get_prev_page_serial(vf,vf->end,vf->serialnos+2,vf->serialnos[1],&endserial,&endgran);
  if(end<0)return(end);

  /* now determine bitstream structure recursively */
  if(_bisect_forward_serialno(vf,0,dataoffset,end,endgran,endserial,
                              vf->serialnos+2,vf->serialnos[1],0)<0)return(OV_EREAD);

  vf->offsets[0]=0;
  vf->serialnos[0]=serialno;
  vf->dataoffsets[0]=dataoffset;
  vf->pcmlengths[0]=pcmoffset;
  vf->pcmlengths[1]-=pcmoffset;
  if(vf->pcmlengths[1]<0)vf->pcmlengths[1]=0;

  return(ov_raw_seek(vf,dataoffset));
}

/* clear out the current logical bitstream decoder */
static void _decode_clear(OggVorbis_File *vf){
  vorbis_dsp_clear(&vf->vd);
  vorbis_block_clear(&vf->vb);
  vf->ready_state=OPENED;
}

/* fetch and process a packet.  Handles the case where we're at a
   bitstream boundary and dumps the decoding machine.  If the decoding
   machine is unloaded, it loads it.  It also keeps pcm_offset up to
   date (seek and read both use this.  seek uses a special hack with
   readp).

   return: <0) error, OV_HOLE (lost packet) or OV_EOF
            0) need more data (only if readp==0)
            1) got a packet
*/

static int _fetch_and_process_packet(OggVorbis_File *vf,
                                     ogg_packet *op_in,
                                     int readp,
                                     int spanp){
  ogg_page og;

  /* handle one packet.  Try to fetch it from current stream state */
  /* extract packets from page */
  while(1){

    if(vf->ready_state==STREAMSET){
      int ret=_make_decode_ready(vf);
      if(ret<0)return ret;
    }

    /* process a packet if we can. */

    if(vf->ready_state==INITSET){
      int hs=vorbis_synthesis_halfrate_p(vf->vi);

      while(1) {
              ogg_packet op;
              ogg_packet *op_ptr=(op_in?op_in:&op);
        int result=ogg_stream_packetout(&vf->os,op_ptr);
        ogg_int64_t granulepos;

        op_in=NULL;
        if(result==-1)return(OV_HOLE); /* hole in the data. */
        if(result>0){
          /* got a packet.  process it */
          granulepos=op_ptr->granulepos;
          if(!vorbis_synthesis(&vf->vb,op_ptr)){ /* lazy check for lazy
                                                    header handling.  The
                                                    header packets aren't
                                                    audio, so if/when we
                                                    submit them,
                                                    vorbis_synthesis will
                                                    reject them */

            /* suck in the synthesis data and track bitrate */
            {
              int oldsamples=vorbis_synthesis_pcmout(&vf->vd,NULL);
              /* for proper use of libvorbis within libvorbisfile,
                 oldsamples will always be zero. */
              if(oldsamples)return(OV_EFAULT);

              vorbis_synthesis_blockin(&vf->vd,&vf->vb);
              vf->samptrack+=(vorbis_synthesis_pcmout(&vf->vd,NULL)<<hs);
              vf->bittrack+=op_ptr->bytes*8;
            }

            /* update the pcm offset. */
            if(granulepos!=-1 && !op_ptr->e_o_s){
              int link=(vf->seekable?vf->current_link:0);
              int i,samples;

              /* this packet has a pcm_offset on it (the last packet
                 completed on a page carries the offset) After processing
                 (above), we know the pcm position of the *last* sample
                 ready to be returned. Find the offset of the *first*

                 As an aside, this trick is inaccurate if we begin
                 reading anew right at the last page; the end-of-stream
                 granulepos declares the last frame in the stream, and the
                 last packet of the last page may be a partial frame.
                 So, we need a previous granulepos from an in-sequence page
                 to have a reference point.  Thus the !op_ptr->e_o_s clause
                 above */

              if(vf->seekable && link>0)
                granulepos-=vf->pcmlengths[link*2];
              if(granulepos<0)granulepos=0; /* actually, this
                                               shouldn't be possible
                                               here unless the stream
                                               is very broken */

              samples=(vorbis_synthesis_pcmout(&vf->vd,NULL)<<hs);

              granulepos-=samples;
              for(i=0;i<link;i++)
                granulepos+=vf->pcmlengths[i*2+1];
              vf->pcm_offset=granulepos;
            }
            return(1);
          }
        }
        else
          break;
      }
    }

    if(vf->ready_state>=OPENED){
      ogg_int64_t ret;

      while(1){
        /* the loop is not strictly necessary, but there's no sense in
           doing the extra checks of the larger loop for the common
           case in a multiplexed bistream where the page is simply
           part of a different logical bitstream; keep reading until
           we get one with the correct serialno */

        if(!readp)return(0);
        if((ret=_get_next_page(vf,&og,-1))<0){
          return(OV_EOF); /* eof. leave unitialized */
        }

        /* bitrate tracking; add the header's bytes here, the body bytes
           are done by packet above */
        vf->bittrack+=og.header_len*8;

        if(vf->ready_state==INITSET){
          if(vf->current_serialno!=ogg_page_serialno(&og)){

            /* two possibilities:
               1) our decoding just traversed a bitstream boundary
               2) another stream is multiplexed into this logical section */

            if(ogg_page_bos(&og)){
              /* boundary case */
              if(!spanp)
                return(OV_EOF);

              _decode_clear(vf);

              if(!vf->seekable){
                vorbis_info_clear(vf->vi);
                vorbis_comment_clear(vf->vc);
              }
              break;

            }else
              continue; /* possibility #2 */
          }
        }

        break;
      }
    }

    /* Do we need to load a new machine before submitting the page? */
    /* This is different in the seekable and non-seekable cases.

       In the seekable case, we already have all the header
       information loaded and cached; we just initialize the machine
       with it and continue on our merry way.

       In the non-seekable (streaming) case, we'll only be at a
       boundary if we just left the previous logical bitstream and
       we're now nominally at the header of the next bitstream
    */

    if(vf->ready_state!=INITSET){
      int link;

      if(vf->ready_state<STREAMSET){
        if(vf->seekable){
          long serialno = ogg_page_serialno(&og);

          /* match the serialno to bitstream section.  We use this rather than
             offset positions to avoid problems near logical bitstream
             boundaries */

          for(link=0;link<vf->links;link++)
            if(vf->serialnos[link]==serialno)break;

          if(link==vf->links) continue; /* not the desired Vorbis
                                           bitstream section; keep
                                           trying */

          vf->current_serialno=serialno;
          vf->current_link=link;

          ogg_stream_reset_serialno(&vf->os,vf->current_serialno);
          vf->ready_state=STREAMSET;

        }else{
          /* we're streaming */
          /* fetch the three header packets, build the info struct */

          int ret=_fetch_headers(vf,vf->vi,vf->vc,NULL,NULL,&og);
          if(ret)return(ret);
          vf->current_serialno=vf->os.serialno;
          vf->current_link++;
          link=0;
        }
      }
    }

    /* the buffered page is the data we want, and we're ready for it;
       add it to the stream state */
    ogg_stream_pagein(&vf->os,&og);

  }
}

/* if, eg, 64 bit stdio is configured by default, this will build with
   fseek64 */
static int _fseek64_wrap(FILE *f,ogg_int64_t off,int whence){
  if(f==NULL)return(-1);
  return fseek(f,off,whence);
}

static int _ov_open1(void *f,OggVorbis_File *vf,const char *initial,
                     long ibytes, ov_callbacks callbacks){
  int offsettest=((f && callbacks.seek_func)?callbacks.seek_func(f,0,SEEK_CUR):-1);
  long *serialno_list=NULL;
  int serialno_list_size=0;
  int ret;

  memset(vf,0,sizeof(*vf));
  vf->datasource=f;
  vf->callbacks = callbacks;

  /* init the framing state */
  ogg_sync_init(&vf->oy);

  /* perhaps some data was previously read into a buffer for testing
     against other stream types.  Allow initialization from this
     previously read data (especially as we may be reading from a
     non-seekable stream) */
  if(initial){
    char *buffer=ogg_sync_buffer(&vf->oy,ibytes);
    memcpy(buffer,initial,ibytes);
    ogg_sync_wrote(&vf->oy,ibytes);
  }

  /* can we seek? Stevens suggests the seek test was portable */
  if(offsettest!=-1)vf->seekable=1;

  /* No seeking yet; Set up a 'single' (current) logical bitstream
     entry for partial open */
  vf->links=1;
  vf->vi=_ogg_calloc(vf->links,sizeof(*vf->vi));
  vf->vc=_ogg_calloc(vf->links,sizeof(*vf->vc));
  ogg_stream_init(&vf->os,-1); /* fill in the serialno later */

  /* Fetch all BOS pages, store the vorbis header and all seen serial
     numbers, load subsequent vorbis setup headers */
  if((ret=_fetch_headers(vf,vf->vi,vf->vc,&serialno_list,&serialno_list_size,NULL))<0){
    vf->datasource=NULL;
    ov_clear(vf);
  }else{
    /* serial number list for first link needs to be held somewhere
       for second stage of seekable stream open; this saves having to
       seek/reread first link's serialnumber data then. */
    vf->serialnos=_ogg_calloc(serialno_list_size+2,sizeof(*vf->serialnos));
    vf->serialnos[0]=vf->current_serialno=vf->os.serialno;
    vf->serialnos[1]=serialno_list_size;
    memcpy(vf->serialnos+2,serialno_list,serialno_list_size*sizeof(*vf->serialnos));

    vf->offsets=_ogg_calloc(1,sizeof(*vf->offsets));
    vf->dataoffsets=_ogg_calloc(1,sizeof(*vf->dataoffsets));
    vf->offsets[0]=0;
    vf->dataoffsets[0]=vf->offset;

    vf->ready_state=PARTOPEN;
  }
  if(serialno_list)_ogg_free(serialno_list);
  return(ret);
}

static int _ov_open2(OggVorbis_File *vf){
  if(vf->ready_state != PARTOPEN) return OV_EINVAL;
  vf->ready_state=OPENED;
  if(vf->seekable){
    int ret=_open_seekable2(vf);
    if(ret){
      vf->datasource=NULL;
      ov_clear(vf);
    }
    return(ret);
  }else
    vf->ready_state=STREAMSET;

  return 0;
}


/* clear out the OggVorbis_File struct */
int ov_clear(OggVorbis_File *vf){
  if(vf){
    vorbis_block_clear(&vf->vb);
    vorbis_dsp_clear(&vf->vd);
    ogg_stream_clear(&vf->os);

    if(vf->vi && vf->links){
      int i;
      for(i=0;i<vf->links;i++){
        vorbis_info_clear(vf->vi+i);
        vorbis_comment_clear(vf->vc+i);
      }
      _ogg_free(vf->vi);
      _ogg_free(vf->vc);
    }
    if(vf->dataoffsets)_ogg_free(vf->dataoffsets);
    if(vf->pcmlengths)_ogg_free(vf->pcmlengths);
    if(vf->serialnos)_ogg_free(vf->serialnos);
    if(vf->offsets)_ogg_free(vf->offsets);
    ogg_sync_clear(&vf->oy);
    if(vf->datasource && vf->callbacks.close_func)
      (vf->callbacks.close_func)(vf->datasource);
    memset(vf,0,sizeof(*vf));
  }
#ifdef DEBUG_LEAKS
  _VDBG_dump();
#endif
  return(0);
}

/* inspects the OggVorbis file and finds/documents all the logical
   bitstreams contained in it.  Tries to be tolerant of logical
   bitstream sections that are truncated/woogie.

   return: -1) error
            0) OK
*/

int ov_open_callbacks(void *f,OggVorbis_File *vf,
    const char *initial,long ibytes,ov_callbacks callbacks){
  int ret=_ov_open1(f,vf,initial,ibytes,callbacks);
  if(ret)return ret;
  return _ov_open2(vf);
}

int ov_open(FILE *f,OggVorbis_File *vf,const char *initial,long ibytes){
  ov_callbacks callbacks = {
    (size_t (*)(void *, size_t, size_t, void *))  fread,
    (int (*)(void *, ogg_int64_t, int))              _fseek64_wrap,
    (int (*)(void *))                             fclose,
    (long (*)(void *))                            ftell
  };

  return ov_open_callbacks((void *)f, vf, initial, ibytes, callbacks);
}

int ov_fopen(const char *path,OggVorbis_File *vf){
  int ret;
  FILE *f = fopen(path,"rb");
  if(!f) return -1;

  ret = ov_open(f,vf,NULL,0);
  if(ret) fclose(f);
  return ret;
}


/* cheap hack for game usage where downsampling is desirable; there's
   no need for SRC as we can just do it cheaply in libvorbis. */

int ov_halfrate(OggVorbis_File *vf,int flag){
  int i;
  if(vf->vi==NULL)return OV_EINVAL;
  if(vf->ready_state>STREAMSET){
    /* clear out stream state; dumping the decode machine is needed to
       reinit the MDCT lookups. */
    vorbis_dsp_clear(&vf->vd);
    vorbis_block_clear(&vf->vb);
    vf->ready_state=STREAMSET;
    if(vf->pcm_offset>=0){
      ogg_int64_t pos=vf->pcm_offset;
      vf->pcm_offset=-1; /* make sure the pos is dumped if unseekable */
      ov_pcm_seek(vf,pos);
    }
  }

  for(i=0;i<vf->links;i++){
    if(vorbis_synthesis_halfrate(vf->vi+i,flag)){
      if(flag) ov_halfrate(vf,0);
      return OV_EINVAL;
    }
  }
  return 0;
}

int ov_halfrate_p(OggVorbis_File *vf){
  if(vf->vi==NULL)return OV_EINVAL;
  return vorbis_synthesis_halfrate_p(vf->vi);
}

/* Only partially open the vorbis file; test for Vorbisness, and load
   the headers for the first chain.  Do not seek (although test for
   seekability).  Use ov_test_open to finish opening the file, else
   ov_clear to close/free it. Same return codes as open.

   Note that vorbisfile does _not_ take ownership of the file if the
   call fails; the calling applicaiton is responsible for closing the file
   if this call returns an error. */

int ov_test_callbacks(void *f,OggVorbis_File *vf,
    const char *initial,long ibytes,ov_callbacks callbacks)
{
  return _ov_open1(f,vf,initial,ibytes,callbacks);
}

int ov_test(FILE *f,OggVorbis_File *vf,const char *initial,long ibytes){
  ov_callbacks callbacks = {
    (size_t (*)(void *, size_t, size_t, void *))  fread,
    (int (*)(void *, ogg_int64_t, int))              _fseek64_wrap,
    (int (*)(void *))                             fclose,
    (long (*)(void *))                            ftell
  };

  return ov_test_callbacks((void *)f, vf, initial, ibytes, callbacks);
}

int ov_test_open(OggVorbis_File *vf){
  if(vf->ready_state!=PARTOPEN)return(OV_EINVAL);
  return _ov_open2(vf);
}

/* How many logical bitstreams in this physical bitstream? */
long ov_streams(OggVorbis_File *vf){
  return vf->links;
}

/* Is the FILE * associated with vf seekable? */
long ov_seekable(OggVorbis_File *vf){
  return vf->seekable;
}

/* returns the bitrate for a given logical bitstream or the entire
   physical bitstream.  If the file is open for random access, it will
   find the *actual* average bitrate.  If the file is streaming, it
   returns the nominal bitrate (if set) else the average of the
   upper/lower bounds (if set) else -1 (unset).

   If you want the actual bitrate field settings, get them from the
   vorbis_info structs */

long ov_bitrate(OggVorbis_File *vf,int i){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(i>=vf->links)return(OV_EINVAL);
  if(!vf->seekable && i!=0)return(ov_bitrate(vf,0));
  if(i<0){
    ogg_int64_t bits=0;
    int i;
    float br;
    for(i=0;i<vf->links;i++)
      bits+=(vf->offsets[i+1]-vf->dataoffsets[i])*8;
    /* This once read: return(rint(bits/ov_time_total(vf,-1)));
     * gcc 3.x on x86 miscompiled this at optimisation level 2 and above,
     * so this is slightly transformed to make it work.
     */
    br = bits/ov_time_total(vf,-1);
    return(rint(br));
  }else{
    if(vf->seekable){
      /* return the actual bitrate */
      return(rint((vf->offsets[i+1]-vf->dataoffsets[i])*8/ov_time_total(vf,i)));
    }else{
      /* return nominal if set */
      if(vf->vi[i].bitrate_nominal>0){
        return vf->vi[i].bitrate_nominal;
      }else{
        if(vf->vi[i].bitrate_upper>0){
          if(vf->vi[i].bitrate_lower>0){
            return (vf->vi[i].bitrate_upper+vf->vi[i].bitrate_lower)/2;
          }else{
            return vf->vi[i].bitrate_upper;
          }
        }
        return(OV_FALSE);
      }
    }
  }
}

/* returns the actual bitrate since last call.  returns -1 if no
   additional data to offer since last call (or at beginning of stream),
   EINVAL if stream is only partially open
*/
long ov_bitrate_instant(OggVorbis_File *vf){
  int link=(vf->seekable?vf->current_link:0);
  long ret;
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(vf->samptrack==0)return(OV_FALSE);
  ret=vf->bittrack/vf->samptrack*vf->vi[link].rate+.5;
  vf->bittrack=0.f;
  vf->samptrack=0.f;
  return(ret);
}

/* Guess */
long ov_serialnumber(OggVorbis_File *vf,int i){
  if(i>=vf->links)return(ov_serialnumber(vf,vf->links-1));
  if(!vf->seekable && i>=0)return(ov_serialnumber(vf,-1));
  if(i<0){
    return(vf->current_serialno);
  }else{
    return(vf->serialnos[i]);
  }
}

/* returns: total raw (compressed) length of content if i==-1
            raw (compressed) length of that logical bitstream for i==0 to n
            OV_EINVAL if the stream is not seekable (we can't know the length)
            or if stream is only partially open
*/
ogg_int64_t ov_raw_total(OggVorbis_File *vf,int i){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable || i>=vf->links)return(OV_EINVAL);
  if(i<0){
    ogg_int64_t acc=0;
    int i;
    for(i=0;i<vf->links;i++)
      acc+=ov_raw_total(vf,i);
    return(acc);
  }else{
    return(vf->offsets[i+1]-vf->offsets[i]);
  }
}

/* returns: total PCM length (samples) of content if i==-1 PCM length
            (samples) of that logical bitstream for i==0 to n
            OV_EINVAL if the stream is not seekable (we can't know the
            length) or only partially open
*/
ogg_int64_t ov_pcm_total(OggVorbis_File *vf,int i){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable || i>=vf->links)return(OV_EINVAL);
  if(i<0){
    ogg_int64_t acc=0;
    int i;
    for(i=0;i<vf->links;i++)
      acc+=ov_pcm_total(vf,i);
    return(acc);
  }else{
    return(vf->pcmlengths[i*2+1]);
  }
}

/* returns: total seconds of content if i==-1
            seconds in that logical bitstream for i==0 to n
            OV_EINVAL if the stream is not seekable (we can't know the
            length) or only partially open
*/
double ov_time_total(OggVorbis_File *vf,int i){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable || i>=vf->links)return(OV_EINVAL);
  if(i<0){
    double acc=0;
    int i;
    for(i=0;i<vf->links;i++)
      acc+=ov_time_total(vf,i);
    return(acc);
  }else{
    return((double)(vf->pcmlengths[i*2+1])/vf->vi[i].rate);
  }
}

/* seek to an offset relative to the *compressed* data. This also
   scans packets to update the PCM cursor. It will cross a logical
   bitstream boundary, but only if it can't get any packets out of the
   tail of the bitstream we seek to (so no surprises).

   returns zero on success, nonzero on failure */

int ov_raw_seek(OggVorbis_File *vf,ogg_int64_t pos){
  ogg_stream_state work_os;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable)
    return(OV_ENOSEEK); /* don't dump machine if we can't seek */

  if(pos<0 || pos>vf->end)return(OV_EINVAL);

  /* is the seek position outside our current link [if any]? */
  if(vf->ready_state>=STREAMSET){
    if(pos<vf->offsets[vf->current_link] || pos>=vf->offsets[vf->current_link+1])
      _decode_clear(vf); /* clear out stream state */
  }

  /* don't yet clear out decoding machine (if it's initialized), in
     the case we're in the same link.  Restart the decode lapping, and
     let _fetch_and_process_packet deal with a potential bitstream
     boundary */
  vf->pcm_offset=-1;
  ogg_stream_reset_serialno(&vf->os,
                            vf->current_serialno); /* must set serialno */
  vorbis_synthesis_restart(&vf->vd);

  if(_seek_helper(vf,pos)) {
    /* dump the machine so we're in a known state */
    vf->pcm_offset=-1;
    _decode_clear(vf);
    return OV_EBADLINK;
  }

  /* we need to make sure the pcm_offset is set, but we don't want to
     advance the raw cursor past good packets just to get to the first
     with a granulepos.  That's not equivalent behavior to beginning
     decoding as immediately after the seek position as possible.

     So, a hack.  We use two stream states; a local scratch state and
     the shared vf->os stream state.  We use the local state to
     scan, and the shared state as a buffer for later decode.

     Unfortuantely, on the last page we still advance to last packet
     because the granulepos on the last page is not necessarily on a
     packet boundary, and we need to make sure the granpos is
     correct.
  */

  {
    ogg_page og;
    ogg_packet op;
    int lastblock=0;
    int accblock=0;
    int thisblock=0;
    int lastflag=0;
    int firstflag=0;
    ogg_int64_t pagepos=-1;

    ogg_stream_init(&work_os,vf->current_serialno); /* get the memory ready */
    ogg_stream_reset(&work_os); /* eliminate the spurious OV_HOLE
                                   return from not necessarily
                                   starting from the beginning */

    while(1){
      if(vf->ready_state>=STREAMSET){
        /* snarf/scan a packet if we can */
        int result=ogg_stream_packetout(&work_os,&op);

        if(result>0){

          if(vf->vi[vf->current_link].codec_setup){
            thisblock=vorbis_packet_blocksize(vf->vi+vf->current_link,&op);
            if(thisblock<0){
              ogg_stream_packetout(&vf->os,NULL);
              thisblock=0;
            }else{

              /* We can't get a guaranteed correct pcm position out of the
                 last page in a stream because it might have a 'short'
                 granpos, which can only be detected in the presence of a
                 preceding page.  However, if the last page is also the first
                 page, the granpos rules of a first page take precedence.  Not
                 only that, but for first==last, the EOS page must be treated
                 as if its a normal first page for the stream to open/play. */
              if(lastflag && !firstflag)
                ogg_stream_packetout(&vf->os,NULL);
              else
                if(lastblock)accblock+=(lastblock+thisblock)>>2;
            }

            if(op.granulepos!=-1){
              int i,link=vf->current_link;
              ogg_int64_t granulepos=op.granulepos-vf->pcmlengths[link*2];
              if(granulepos<0)granulepos=0;

              for(i=0;i<link;i++)
                granulepos+=vf->pcmlengths[i*2+1];
              vf->pcm_offset=granulepos-accblock;
              if(vf->pcm_offset<0)vf->pcm_offset=0;
              break;
            }
            lastblock=thisblock;
            continue;
          }else
            ogg_stream_packetout(&vf->os,NULL);
        }
      }

      if(!lastblock){
        pagepos=_get_next_page(vf,&og,-1);
        if(pagepos<0){
          vf->pcm_offset=ov_pcm_total(vf,-1);
          break;
        }
      }else{
        /* huh?  Bogus stream with packets but no granulepos */
        vf->pcm_offset=-1;
        break;
      }

      /* has our decoding just traversed a bitstream boundary? */
      if(vf->ready_state>=STREAMSET){
        if(vf->current_serialno!=ogg_page_serialno(&og)){

          /* two possibilities:
             1) our decoding just traversed a bitstream boundary
             2) another stream is multiplexed into this logical section? */

          if(ogg_page_bos(&og)){
            /* we traversed */
            _decode_clear(vf); /* clear out stream state */
            ogg_stream_clear(&work_os);
          } /* else, do nothing; next loop will scoop another page */
        }
      }

      if(vf->ready_state<STREAMSET){
        int link;
        long serialno = ogg_page_serialno(&og);

        for(link=0;link<vf->links;link++)
          if(vf->serialnos[link]==serialno)break;

        if(link==vf->links) continue; /* not the desired Vorbis
                                         bitstream section; keep
                                         trying */
        vf->current_link=link;
        vf->current_serialno=serialno;
        ogg_stream_reset_serialno(&vf->os,serialno);
        ogg_stream_reset_serialno(&work_os,serialno);
        vf->ready_state=STREAMSET;
        firstflag=(pagepos<=vf->dataoffsets[link]);
      }

      ogg_stream_pagein(&vf->os,&og);
      ogg_stream_pagein(&work_os,&og);
      lastflag=ogg_page_eos(&og);

    }
  }

  ogg_stream_clear(&work_os);
  vf->bittrack=0.f;
  vf->samptrack=0.f;
  return(0);
}

/* Page granularity seek (faster than sample granularity because we
   don't do the last bit of decode to find a specific sample).

   Seek to the last [granule marked] page preceding the specified pos
   location, such that decoding past the returned point will quickly
   arrive at the requested position. */
int ov_pcm_seek_page(OggVorbis_File *vf,ogg_int64_t pos){
  int link=-1;
  ogg_int64_t result=0;
  ogg_int64_t total=ov_pcm_total(vf,-1);

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable)return(OV_ENOSEEK);

  if(pos<0 || pos>total)return(OV_EINVAL);

  /* which bitstream section does this pcm offset occur in? */
  for(link=vf->links-1;link>=0;link--){
    total-=vf->pcmlengths[link*2+1];
    if(pos>=total)break;
  }

  /* Search within the logical bitstream for the page with the highest
     pcm_pos preceding pos.  If we're looking for a position on the
     first page, bisection will halt without finding our position as
     it's before the first explicit granulepos fencepost. That case is
     handled separately below.

     There is a danger here; missing pages or incorrect frame number
     information in the bitstream could make our task impossible.
     Account for that (it would be an error condition) */

  /* new search algorithm originally by HB (Nicholas Vinen) */

  {
    ogg_int64_t end=vf->offsets[link+1];
    ogg_int64_t begin=vf->dataoffsets[link];
    ogg_int64_t begintime = vf->pcmlengths[link*2];
    ogg_int64_t endtime = vf->pcmlengths[link*2+1]+begintime;
    ogg_int64_t target=pos-total+begintime;
    ogg_int64_t best=-1;
    int         got_page=0;

    ogg_page og;

    /* if we have only one page, there will be no bisection.  Grab the page here */
    if(begin==end){
      result=_seek_helper(vf,begin);
      if(result) goto seek_error;

      result=_get_next_page(vf,&og,1);
      if(result<0) goto seek_error;

      got_page=1;
    }

    /* bisection loop */
    while(begin<end){
      ogg_int64_t bisect;

      if(end-begin<CHUNKSIZE){
        bisect=begin;
      }else{
        /* take a (pretty decent) guess. */
        bisect=begin +
          (ogg_int64_t)((double)(target-begintime)*(end-begin)/(endtime-begintime))
          - CHUNKSIZE;
        if(bisect<begin+CHUNKSIZE)
          bisect=begin;
      }

      result=_seek_helper(vf,bisect);
      if(result) goto seek_error;

      /* read loop within the bisection loop */
      while(begin<end){
        result=_get_next_page(vf,&og,end-vf->offset);
        if(result==OV_EREAD) goto seek_error;
        if(result<0){
          /* there is no next page! */
          if(bisect<=begin+1)
              /* No bisection left to perform.  We've either found the
                 best candidate already or failed. Exit loop. */
            end=begin;
          else{
            /* We tried to load a fraction of the last page; back up a
               bit and try to get the whole last page */
            if(bisect==0) goto seek_error;
            bisect-=CHUNKSIZE;

            /* don't repeat/loop on a read we've already performed */
            if(bisect<=begin)bisect=begin+1;

            /* seek and cntinue bisection */
            result=_seek_helper(vf,bisect);
            if(result) goto seek_error;
          }
        }else{
          ogg_int64_t granulepos;
          got_page=1;

          /* got a page. analyze it */
          /* only consider pages from primary vorbis stream */
          if(ogg_page_serialno(&og)!=vf->serialnos[link])
            continue;

          /* only consider pages with the granulepos set */
          granulepos=ogg_page_granulepos(&og);
          if(granulepos==-1)continue;

          if(granulepos<target){
            /* this page is a successful candidate! Set state */

            best=result;  /* raw offset of packet with granulepos */
            begin=vf->offset; /* raw offset of next page */
            begintime=granulepos;

            /* if we're before our target but within a short distance,
               don't bisect; read forward */
            if(target-begintime>44100)break;

            bisect=begin; /* *not* begin + 1 as above */
          }else{

            /* This is one of our pages, but the granpos is
               post-target; it is not a bisection return
               candidate. (The only way we'd use it is if it's the
               first page in the stream; we handle that case later
               outside the bisection) */
            if(bisect<=begin+1){
              /* No bisection left to perform.  We've either found the
                 best candidate already or failed. Exit loop. */
              end=begin;
            }else{
              if(end==vf->offset){
                /* bisection read to the end; use the known page
                   boundary (result) to update bisection, back up a
                   little bit, and try again */
                end=result;
                bisect-=CHUNKSIZE;
                if(bisect<=begin)bisect=begin+1;
                result=_seek_helper(vf,bisect);
                if(result) goto seek_error;
              }else{
                /* Normal bisection */
                end=bisect;
                endtime=granulepos;
                break;
              }
            }
          }
        }
      }
    }

    /* Out of bisection: did it 'fail?' */
    if(best == -1){

      /* Check the 'looking for data in first page' special case;
         bisection would 'fail' because our search target was before the
         first PCM granule position fencepost. */

      if(got_page &&
         begin == vf->dataoffsets[link] &&
         ogg_page_serialno(&og)==vf->serialnos[link]){

        /* Yes, this is the beginning-of-stream case. We already have
           our page, right at the beginning of PCM data.  Set state
           and return. */

        vf->pcm_offset=total;

        if(link!=vf->current_link){
          /* Different link; dump entire decode machine */
          _decode_clear(vf);

          vf->current_link=link;
          vf->current_serialno=vf->serialnos[link];
          vf->ready_state=STREAMSET;

        }else{
          vorbis_synthesis_restart(&vf->vd);
        }

        ogg_stream_reset_serialno(&vf->os,vf->current_serialno);
        ogg_stream_pagein(&vf->os,&og);

      }else
        goto seek_error;

    }else{

      /* Bisection found our page. seek to it, update pcm offset. Easier case than
         raw_seek, don't keep packets preceding granulepos. */

      ogg_page og;
      ogg_packet op;

      /* seek */
      result=_seek_helper(vf,best);
      vf->pcm_offset=-1;
      if(result) goto seek_error;
      result=_get_next_page(vf,&og,-1);
      if(result<0) goto seek_error;

      if(link!=vf->current_link){
        /* Different link; dump entire decode machine */
        _decode_clear(vf);

        vf->current_link=link;
        vf->current_serialno=vf->serialnos[link];
        vf->ready_state=STREAMSET;

      }else{
        vorbis_synthesis_restart(&vf->vd);
      }

      ogg_stream_reset_serialno(&vf->os,vf->current_serialno);
      ogg_stream_pagein(&vf->os,&og);

      /* pull out all but last packet; the one with granulepos */
      while(1){
        result=ogg_stream_packetpeek(&vf->os,&op);
        if(result==0){
          /* No packet returned; we exited the bisection with 'best'
             pointing to a page with a granule position, so the packet
             finishing this page ('best') originated on a preceding
             page. Keep fetching previous pages until we get one with
             a granulepos or without the 'continued' flag set.  Then
             just use raw_seek for simplicity. */
          /* Do not rewind past the beginning of link data; if we do,
             it's either a bug or a broken stream */
          result=best;
          while(result>vf->dataoffsets[link]){
            result=_get_prev_page(vf,result,&og);
            if(result<0) goto seek_error;
            if(ogg_page_serialno(&og)==vf->current_serialno &&
               (ogg_page_granulepos(&og)>-1 ||
                !ogg_page_continued(&og))){
              return ov_raw_seek(vf,result);
            }
          }
        }
        if(result<0){
          result = OV_EBADPACKET;
          goto seek_error;
        }
        if(op.granulepos!=-1){
          vf->pcm_offset=op.granulepos-vf->pcmlengths[vf->current_link*2];
          if(vf->pcm_offset<0)vf->pcm_offset=0;
          vf->pcm_offset+=total;
          break;
        }else
          result=ogg_stream_packetout(&vf->os,NULL);
      }
    }
  }

  /* verify result */
  if(vf->pcm_offset>pos || pos>ov_pcm_total(vf,-1)){
    result=OV_EFAULT;
    goto seek_error;
  }
  vf->bittrack=0.f;
  vf->samptrack=0.f;
  return(0);

 seek_error:
  /* dump machine so we're in a known state */
  vf->pcm_offset=-1;
  _decode_clear(vf);
  return (int)result;
}

/* seek to a sample offset relative to the decompressed pcm stream
   returns zero on success, nonzero on failure */

int ov_pcm_seek(OggVorbis_File *vf,ogg_int64_t pos){
  int thisblock,lastblock=0;
  int ret=ov_pcm_seek_page(vf,pos);
  if(ret<0)return(ret);
  if((ret=_make_decode_ready(vf)))return ret;

  /* discard leading packets we don't need for the lapping of the
     position we want; don't decode them */

  while(1){
    ogg_packet op;
    ogg_page og;

    int ret=ogg_stream_packetpeek(&vf->os,&op);
    if(ret>0){
      thisblock=vorbis_packet_blocksize(vf->vi+vf->current_link,&op);
      if(thisblock<0){
        ogg_stream_packetout(&vf->os,NULL);
        continue; /* non audio packet */
      }
      if(lastblock)vf->pcm_offset+=(lastblock+thisblock)>>2;

      if(vf->pcm_offset+((thisblock+
                          vorbis_info_blocksize(vf->vi,1))>>2)>=pos)break;

      /* remove the packet from packet queue and track its granulepos */
      ogg_stream_packetout(&vf->os,NULL);
      vorbis_synthesis_trackonly(&vf->vb,&op);  /* set up a vb with
                                                   only tracking, no
                                                   pcm_decode */
      vorbis_synthesis_blockin(&vf->vd,&vf->vb);

      /* end of logical stream case is hard, especially with exact
         length positioning. */

      if(op.granulepos>-1){
        int i;
        /* always believe the stream markers */
        vf->pcm_offset=op.granulepos-vf->pcmlengths[vf->current_link*2];
        if(vf->pcm_offset<0)vf->pcm_offset=0;
        for(i=0;i<vf->current_link;i++)
          vf->pcm_offset+=vf->pcmlengths[i*2+1];
      }

      lastblock=thisblock;

    }else{
      if(ret<0 && ret!=OV_HOLE)break;

      /* suck in a new page */
      if(_get_next_page(vf,&og,-1)<0)break;
      if(ogg_page_bos(&og))_decode_clear(vf);

      if(vf->ready_state<STREAMSET){
        long serialno=ogg_page_serialno(&og);
        int link;

        for(link=0;link<vf->links;link++)
          if(vf->serialnos[link]==serialno)break;
        if(link==vf->links) continue;
        vf->current_link=link;

        vf->ready_state=STREAMSET;
        vf->current_serialno=ogg_page_serialno(&og);
        ogg_stream_reset_serialno(&vf->os,serialno);
        ret=_make_decode_ready(vf);
        if(ret)return ret;
        lastblock=0;
      }

      ogg_stream_pagein(&vf->os,&og);
    }
  }

  vf->bittrack=0.f;
  vf->samptrack=0.f;
  /* discard samples until we reach the desired position. Crossing a
     logical bitstream boundary with abandon is OK. */
  {
    /* note that halfrate could be set differently in each link, but
       vorbisfile encoforces all links are set or unset */
    int hs=vorbis_synthesis_halfrate_p(vf->vi);
    while(vf->pcm_offset<((pos>>hs)<<hs)){
      ogg_int64_t target=(pos-vf->pcm_offset)>>hs;
      long samples=vorbis_synthesis_pcmout(&vf->vd,NULL);

      if(samples>target)samples=target;
      vorbis_synthesis_read(&vf->vd,samples);
      vf->pcm_offset+=samples<<hs;

      if(samples<target)
        if(_fetch_and_process_packet(vf,NULL,1,1)<=0)
          vf->pcm_offset=ov_pcm_total(vf,-1); /* eof */
    }
  }
  return 0;
}

/* seek to a playback time relative to the decompressed pcm stream
   returns zero on success, nonzero on failure */
int ov_time_seek(OggVorbis_File *vf,double seconds){
  /* translate time to PCM position and call ov_pcm_seek */

  int link=-1;
  ogg_int64_t pcm_total=0;
  double time_total=0.;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable)return(OV_ENOSEEK);
  if(seconds<0)return(OV_EINVAL);

  /* which bitstream section does this time offset occur in? */
  for(link=0;link<vf->links;link++){
    double addsec = ov_time_total(vf,link);
    if(seconds<time_total+addsec)break;
    time_total+=addsec;
    pcm_total+=vf->pcmlengths[link*2+1];
  }

  if(link==vf->links)return(OV_EINVAL);

  /* enough information to convert time offset to pcm offset */
  {
    ogg_int64_t target=pcm_total+(seconds-time_total)*vf->vi[link].rate;
    return(ov_pcm_seek(vf,target));
  }
}

/* page-granularity version of ov_time_seek
   returns zero on success, nonzero on failure */
int ov_time_seek_page(OggVorbis_File *vf,double seconds){
  /* translate time to PCM position and call ov_pcm_seek */

  int link=-1;
  ogg_int64_t pcm_total=0;
  double time_total=0.;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(!vf->seekable)return(OV_ENOSEEK);
  if(seconds<0)return(OV_EINVAL);

  /* which bitstream section does this time offset occur in? */
  for(link=0;link<vf->links;link++){
    double addsec = ov_time_total(vf,link);
    if(seconds<time_total+addsec)break;
    time_total+=addsec;
    pcm_total+=vf->pcmlengths[link*2+1];
  }

  if(link==vf->links)return(OV_EINVAL);

  /* enough information to convert time offset to pcm offset */
  {
    ogg_int64_t target=pcm_total+(seconds-time_total)*vf->vi[link].rate;
    return(ov_pcm_seek_page(vf,target));
  }
}

/* tell the current stream offset cursor.  Note that seek followed by
   tell will likely not give the set offset due to caching */
ogg_int64_t ov_raw_tell(OggVorbis_File *vf){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  return(vf->offset);
}

/* return PCM offset (sample) of next PCM sample to be read */
ogg_int64_t ov_pcm_tell(OggVorbis_File *vf){
  if(vf->ready_state<OPENED)return(OV_EINVAL);
  return(vf->pcm_offset);
}

/* return time offset (seconds) of next PCM sample to be read */
double ov_time_tell(OggVorbis_File *vf){
  int link=0;
  ogg_int64_t pcm_total=0;
  double time_total=0.f;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(vf->seekable){
    pcm_total=ov_pcm_total(vf,-1);
    time_total=ov_time_total(vf,-1);

    /* which bitstream section does this time offset occur in? */
    for(link=vf->links-1;link>=0;link--){
      pcm_total-=vf->pcmlengths[link*2+1];
      time_total-=ov_time_total(vf,link);
      if(vf->pcm_offset>=pcm_total)break;
    }
  }

  return((double)time_total+(double)(vf->pcm_offset-pcm_total)/vf->vi[link].rate);
}

/*  link:   -1) return the vorbis_info struct for the bitstream section
                currently being decoded
           0-n) to request information for a specific bitstream section

    In the case of a non-seekable bitstream, any call returns the
    current bitstream.  NULL in the case that the machine is not
    initialized */

vorbis_info *ov_info(OggVorbis_File *vf,int link){
  if(vf->seekable){
    if(link<0)
      if(vf->ready_state>=STREAMSET)
        return vf->vi+vf->current_link;
      else
      return vf->vi;
    else
      if(link>=vf->links)
        return NULL;
      else
        return vf->vi+link;
  }else{
    return vf->vi;
  }
}

/* grr, strong typing, grr, no templates/inheritence, grr */
vorbis_comment *ov_comment(OggVorbis_File *vf,int link){
  if(vf->seekable){
    if(link<0)
      if(vf->ready_state>=STREAMSET)
        return vf->vc+vf->current_link;
      else
        return vf->vc;
    else
      if(link>=vf->links)
        return NULL;
      else
        return vf->vc+link;
  }else{
    return vf->vc;
  }
}

static int host_is_big_endian() {
  ogg_int32_t pattern = 0xfeedface; /* deadbeef */
  unsigned char *bytewise = (unsigned char *)&pattern;
  if (bytewise[0] == 0xfe) return 1;
  return 0;
}

/* up to this point, everything could more or less hide the multiple
   logical bitstream nature of chaining from the toplevel application
   if the toplevel application didn't particularly care.  However, at
   the point that we actually read audio back, the multiple-section
   nature must surface: Multiple bitstream sections do not necessarily
   have to have the same number of channels or sampling rate.

   ov_read returns the sequential logical bitstream number currently
   being decoded along with the PCM data in order that the toplevel
   application can take action on channel/sample rate changes.  This
   number will be incremented even for streamed (non-seekable) streams
   (for seekable streams, it represents the actual logical bitstream
   index within the physical bitstream.  Note that the accessor
   functions above are aware of this dichotomy).

   ov_read_filter is exactly the same as ov_read except that it processes
   the decoded audio data through a filter before packing it into the
   requested format. This gives greater accuracy than applying a filter
   after the audio has been converted into integral PCM.

   input values: buffer) a buffer to hold packed PCM data for return
                 length) the byte length requested to be placed into buffer
                 bigendianp) should the data be packed LSB first (0) or
                             MSB first (1)
                 word) word size for output.  currently 1 (byte) or
                       2 (16 bit short)

   return values: <0) error/hole in data (OV_HOLE), partial open (OV_EINVAL)
                   0) EOF
                   n) number of bytes of PCM actually returned.  The
                   below works on a packet-by-packet basis, so the
                   return length is not related to the 'length' passed
                   in, just guaranteed to fit.

            *section) set to the logical bitstream number */

long ov_read_filter(OggVorbis_File *vf,char *buffer,int length,
                    int bigendianp,int word,int sgned,int *bitstream,
                    void (*filter)(float **pcm,long channels,long samples,void *filter_param),void *filter_param){
  int i,j;
  int host_endian = host_is_big_endian();
  int hs;

  float **pcm;
  long samples;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  if(word<=0)return(OV_EINVAL);

  while(1){
    if(vf->ready_state==INITSET){
      samples=vorbis_synthesis_pcmout(&vf->vd,&pcm);
      if(samples)break;
    }

    /* suck in another packet */
    {
      int ret=_fetch_and_process_packet(vf,NULL,1,1);
      if(ret==OV_EOF)
        return(0);
      if(ret<=0)
        return(ret);
    }

  }

  if(samples>0){

    /* yay! proceed to pack data into the byte buffer */

    long channels=ov_info(vf,-1)->channels;
    long bytespersample=word * channels;
    vorbis_fpu_control fpu;

    if(channels<1||channels>255)return(OV_EINVAL);
    if(samples>length/bytespersample)samples=length/bytespersample;

    if(samples <= 0)
      return OV_EINVAL;

    /* Here. */
    if(filter)
      filter(pcm,channels,samples,filter_param);

    /* a tight loop to pack each size */
    {
      int val;
      if(word==1){
        int off=(sgned?0:128);
        vorbis_fpu_setround(&fpu);
        for(j=0;j<samples;j++)
          for(i=0;i<channels;i++){
            val=vorbis_ftoi(pcm[i][j]*128.f);
            if(val>127)val=127;
            else if(val<-128)val=-128;
            *buffer++=val+off;
          }
        vorbis_fpu_restore(fpu);
      }else{
        int off=(sgned?0:32768);

        if(host_endian==bigendianp){
          if(sgned){

            vorbis_fpu_setround(&fpu);
            for(i=0;i<channels;i++) { /* It's faster in this order */
              float *src=pcm[i];
              short *dest=((short *)buffer)+i;
              for(j=0;j<samples;j++) {
                val=vorbis_ftoi(src[j]*32768.f);
                if(val>32767)val=32767;
                else if(val<-32768)val=-32768;
                *dest=val;
                dest+=channels;
              }
            }
            vorbis_fpu_restore(fpu);

          }else{

            vorbis_fpu_setround(&fpu);
            for(i=0;i<channels;i++) {
              float *src=pcm[i];
              short *dest=((short *)buffer)+i;
              for(j=0;j<samples;j++) {
                val=vorbis_ftoi(src[j]*32768.f);
                if(val>32767)val=32767;
                else if(val<-32768)val=-32768;
                *dest=val+off;
                dest+=channels;
              }
            }
            vorbis_fpu_restore(fpu);

          }
        }else if(bigendianp){

          vorbis_fpu_setround(&fpu);
          for(j=0;j<samples;j++)
            for(i=0;i<channels;i++){
              val=vorbis_ftoi(pcm[i][j]*32768.f);
              if(val>32767)val=32767;
              else if(val<-32768)val=-32768;
              val+=off;
              *buffer++=(val>>8);
              *buffer++=(val&0xff);
            }
          vorbis_fpu_restore(fpu);

        }else{
          int val;
          vorbis_fpu_setround(&fpu);
          for(j=0;j<samples;j++)
            for(i=0;i<channels;i++){
              val=vorbis_ftoi(pcm[i][j]*32768.f);
              if(val>32767)val=32767;
              else if(val<-32768)val=-32768;
              val+=off;
              *buffer++=(val&0xff);
              *buffer++=(val>>8);
                  }
          vorbis_fpu_restore(fpu);

        }
      }
    }

    vorbis_synthesis_read(&vf->vd,samples);
    hs=vorbis_synthesis_halfrate_p(vf->vi);
    vf->pcm_offset+=(samples<<hs);
    if(bitstream)*bitstream=vf->current_link;
    return(samples*bytespersample);
  }else{
    return(samples);
  }
}

long ov_read(OggVorbis_File *vf,char *buffer,int length,
             int bigendianp,int word,int sgned,int *bitstream){
  return ov_read_filter(vf, buffer, length, bigendianp, word, sgned, bitstream, NULL, NULL);
}

/* input values: pcm_channels) a float vector per channel of output
                 length) the sample length being read by the app

   return values: <0) error/hole in data (OV_HOLE), partial open (OV_EINVAL)
                   0) EOF
                   n) number of samples of PCM actually returned.  The
                   below works on a packet-by-packet basis, so the
                   return length is not related to the 'length' passed
                   in, just guaranteed to fit.

            *section) set to the logical bitstream number */



long ov_read_float(OggVorbis_File *vf,float ***pcm_channels,int length,
                   int *bitstream){

  if(vf->ready_state<OPENED)return(OV_EINVAL);

  while(1){
    if(vf->ready_state==INITSET){
      float **pcm;
      long samples=vorbis_synthesis_pcmout(&vf->vd,&pcm);
      if(samples){
        int hs=vorbis_synthesis_halfrate_p(vf->vi);
        if(pcm_channels)*pcm_channels=pcm;
        if(samples>length)samples=length;
        vorbis_synthesis_read(&vf->vd,samples);
        vf->pcm_offset+=samples<<hs;
        if(bitstream)*bitstream=vf->current_link;
        return samples;

      }
    }

    /* suck in another packet */
    {
      int ret=_fetch_and_process_packet(vf,NULL,1,1);
      if(ret==OV_EOF)return(0);
      if(ret<=0)return(ret);
    }

  }
}

extern const float *vorbis_window(vorbis_dsp_state *v,int W);

static void _ov_splice(float **pcm,float **lappcm,
                       int n1, int n2,
                       int ch1, int ch2,
                       const float *w1, const float *w2){
  int i,j;
  const float *w=w1;
  int n=n1;

  if(n1>n2){
    n=n2;
    w=w2;
  }

  /* splice */
  for(j=0;j<ch1 && j<ch2;j++){
    float *s=lappcm[j];
    float *d=pcm[j];

    for(i=0;i<n;i++){
      float wd=w[i]*w[i];
      float ws=1.-wd;
      d[i]=d[i]*wd + s[i]*ws;
    }
  }
  /* window from zero */
  for(;j<ch2;j++){
    float *d=pcm[j];
    for(i=0;i<n;i++){
      float wd=w[i]*w[i];
      d[i]=d[i]*wd;
    }
  }

}

/* make sure vf is INITSET */
static int _ov_initset(OggVorbis_File *vf){
  while(1){
    if(vf->ready_state==INITSET)break;
    /* suck in another packet */
    {
      int ret=_fetch_and_process_packet(vf,NULL,1,0);
      if(ret<0 && ret!=OV_HOLE)return(ret);
    }
  }
  return 0;
}

/* make sure vf is INITSET and that we have a primed buffer; if
   we're crosslapping at a stream section boundary, this also makes
   sure we're sanity checking against the right stream information */
static int _ov_initprime(OggVorbis_File *vf){
  vorbis_dsp_state *vd=&vf->vd;
  while(1){
    if(vf->ready_state==INITSET)
      if(vorbis_synthesis_pcmout(vd,NULL))break;

    /* suck in another packet */
    {
      int ret=_fetch_and_process_packet(vf,NULL,1,0);
      if(ret<0 && ret!=OV_HOLE)return(ret);
    }
  }
  return 0;
}

/* grab enough data for lapping from vf; this may be in the form of
   unreturned, already-decoded pcm, remaining PCM we will need to
   decode, or synthetic postextrapolation from last packets. */
static void _ov_getlap(OggVorbis_File *vf,vorbis_info *vi,vorbis_dsp_state *vd,
                       float **lappcm,int lapsize){
  int lapcount=0,i;
  float **pcm;

  /* try first to decode the lapping data */
  while(lapcount<lapsize){
    int samples=vorbis_synthesis_pcmout(vd,&pcm);
    if(samples){
      if(samples>lapsize-lapcount)samples=lapsize-lapcount;
      for(i=0;i<vi->channels;i++)
        memcpy(lappcm[i]+lapcount,pcm[i],sizeof(**pcm)*samples);
      lapcount+=samples;
      vorbis_synthesis_read(vd,samples);
    }else{
    /* suck in another packet */
      int ret=_fetch_and_process_packet(vf,NULL,1,0); /* do *not* span */
      if(ret==OV_EOF)break;
    }
  }
  if(lapcount<lapsize){
    /* failed to get lapping data from normal decode; pry it from the
       postextrapolation buffering, or the second half of the MDCT
       from the last packet */
    int samples=vorbis_synthesis_lapout(&vf->vd,&pcm);
    if(samples==0){
      for(i=0;i<vi->channels;i++)
        memset(lappcm[i]+lapcount,0,sizeof(**pcm)*lapsize-lapcount);
      lapcount=lapsize;
    }else{
      if(samples>lapsize-lapcount)samples=lapsize-lapcount;
      for(i=0;i<vi->channels;i++)
        memcpy(lappcm[i]+lapcount,pcm[i],sizeof(**pcm)*samples);
      lapcount+=samples;
    }
  }
}

/* this sets up crosslapping of a sample by using trailing data from
   sample 1 and lapping it into the windowing buffer of sample 2 */
int ov_crosslap(OggVorbis_File *vf1, OggVorbis_File *vf2){
  vorbis_info *vi1,*vi2;
  float **lappcm;
  float **pcm;
  const float *w1,*w2;
  int n1,n2,i,ret,hs1,hs2;

  if(vf1==vf2)return(0); /* degenerate case */
  if(vf1->ready_state<OPENED)return(OV_EINVAL);
  if(vf2->ready_state<OPENED)return(OV_EINVAL);

  /* the relevant overlap buffers must be pre-checked and pre-primed
     before looking at settings in the event that priming would cross
     a bitstream boundary.  So, do it now */

  ret=_ov_initset(vf1);
  if(ret)return(ret);
  ret=_ov_initprime(vf2);
  if(ret)return(ret);

  vi1=ov_info(vf1,-1);
  vi2=ov_info(vf2,-1);
  hs1=ov_halfrate_p(vf1);
  hs2=ov_halfrate_p(vf2);

  lappcm=alloca(sizeof(*lappcm)*vi1->channels);
  n1=vorbis_info_blocksize(vi1,0)>>(1+hs1);
  n2=vorbis_info_blocksize(vi2,0)>>(1+hs2);
  w1=vorbis_window(&vf1->vd,0);
  w2=vorbis_window(&vf2->vd,0);

  for(i=0;i<vi1->channels;i++)
    lappcm[i]=alloca(sizeof(**lappcm)*n1);

  _ov_getlap(vf1,vi1,&vf1->vd,lappcm,n1);

  /* have a lapping buffer from vf1; now to splice it into the lapping
     buffer of vf2 */
  /* consolidate and expose the buffer. */
  vorbis_synthesis_lapout(&vf2->vd,&pcm);

#if 0
  _analysis_output_always("pcmL",0,pcm[0],n1*2,0,0,0);
  _analysis_output_always("pcmR",0,pcm[1],n1*2,0,0,0);
#endif

  /* splice */
  _ov_splice(pcm,lappcm,n1,n2,vi1->channels,vi2->channels,w1,w2);

  /* done */
  return(0);
}

static int _ov_64_seek_lap(OggVorbis_File *vf,ogg_int64_t pos,
                           int (*localseek)(OggVorbis_File *,ogg_int64_t)){
  vorbis_info *vi;
  float **lappcm;
  float **pcm;
  const float *w1,*w2;
  int n1,n2,ch1,ch2,hs;
  int i,ret;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  ret=_ov_initset(vf);
  if(ret)return(ret);
  vi=ov_info(vf,-1);
  hs=ov_halfrate_p(vf);

  ch1=vi->channels;
  n1=vorbis_info_blocksize(vi,0)>>(1+hs);
  w1=vorbis_window(&vf->vd,0);  /* window arrays from libvorbis are
                                   persistent; even if the decode state
                                   from this link gets dumped, this
                                   window array continues to exist */

  lappcm=alloca(sizeof(*lappcm)*ch1);
  for(i=0;i<ch1;i++)
    lappcm[i]=alloca(sizeof(**lappcm)*n1);
  _ov_getlap(vf,vi,&vf->vd,lappcm,n1);

  /* have lapping data; seek and prime the buffer */
  ret=localseek(vf,pos);
  if(ret)return ret;
  ret=_ov_initprime(vf);
  if(ret)return(ret);

 /* Guard against cross-link changes; they're perfectly legal */
  vi=ov_info(vf,-1);
  ch2=vi->channels;
  n2=vorbis_info_blocksize(vi,0)>>(1+hs);
  w2=vorbis_window(&vf->vd,0);

  /* consolidate and expose the buffer. */
  vorbis_synthesis_lapout(&vf->vd,&pcm);

  /* splice */
  _ov_splice(pcm,lappcm,n1,n2,ch1,ch2,w1,w2);

  /* done */
  return(0);
}

int ov_raw_seek_lap(OggVorbis_File *vf,ogg_int64_t pos){
  return _ov_64_seek_lap(vf,pos,ov_raw_seek);
}

int ov_pcm_seek_lap(OggVorbis_File *vf,ogg_int64_t pos){
  return _ov_64_seek_lap(vf,pos,ov_pcm_seek);
}

int ov_pcm_seek_page_lap(OggVorbis_File *vf,ogg_int64_t pos){
  return _ov_64_seek_lap(vf,pos,ov_pcm_seek_page);
}

static int _ov_d_seek_lap(OggVorbis_File *vf,double pos,
                           int (*localseek)(OggVorbis_File *,double)){
  vorbis_info *vi;
  float **lappcm;
  float **pcm;
  const float *w1,*w2;
  int n1,n2,ch1,ch2,hs;
  int i,ret;

  if(vf->ready_state<OPENED)return(OV_EINVAL);
  ret=_ov_initset(vf);
  if(ret)return(ret);
  vi=ov_info(vf,-1);
  hs=ov_halfrate_p(vf);

  ch1=vi->channels;
  n1=vorbis_info_blocksize(vi,0)>>(1+hs);
  w1=vorbis_window(&vf->vd,0);  /* window arrays from libvorbis are
                                   persistent; even if the decode state
                                   from this link gets dumped, this
                                   window array continues to exist */

  lappcm=alloca(sizeof(*lappcm)*ch1);
  for(i=0;i<ch1;i++)
    lappcm[i]=alloca(sizeof(**lappcm)*n1);
  _ov_getlap(vf,vi,&vf->vd,lappcm,n1);

  /* have lapping data; seek and prime the buffer */
  ret=localseek(vf,pos);
  if(ret)return ret;
  ret=_ov_initprime(vf);
  if(ret)return(ret);

 /* Guard against cross-link changes; they're perfectly legal */
  vi=ov_info(vf,-1);
  ch2=vi->channels;
  n2=vorbis_info_blocksize(vi,0)>>(1+hs);
  w2=vorbis_window(&vf->vd,0);

  /* consolidate and expose the buffer. */
  vorbis_synthesis_lapout(&vf->vd,&pcm);

  /* splice */
  _ov_splice(pcm,lappcm,n1,n2,ch1,ch2,w1,w2);

  /* done */
  return(0);
}

int ov_time_seek_lap(OggVorbis_File *vf,double pos){
  return _ov_d_seek_lap(vf,pos,ov_time_seek);
}

int ov_time_seek_page_lap(OggVorbis_File *vf,double pos){
  return _ov_d_seek_lap(vf,pos,ov_time_seek_page);
}
