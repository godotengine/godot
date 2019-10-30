/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE Ogg CONTAINER SOURCE CODE.              *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2018             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: code raw packets into framed OggSquish stream and
           decode Ogg streams back into raw packets

 note: The CRC code is directly derived from public domain code by
 Ross Williams (ross@guest.adelaide.edu.au).  See docs/framing.html
 for details.

 ********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <ogg/ogg.h>

/* A complete description of Ogg framing exists in docs/framing.html */

int ogg_page_version(const ogg_page *og){
  return((int)(og->header[4]));
}

int ogg_page_continued(const ogg_page *og){
  return((int)(og->header[5]&0x01));
}

int ogg_page_bos(const ogg_page *og){
  return((int)(og->header[5]&0x02));
}

int ogg_page_eos(const ogg_page *og){
  return((int)(og->header[5]&0x04));
}

ogg_int64_t ogg_page_granulepos(const ogg_page *og){
  unsigned char *page=og->header;
  ogg_uint64_t granulepos=page[13]&(0xff);
  granulepos= (granulepos<<8)|(page[12]&0xff);
  granulepos= (granulepos<<8)|(page[11]&0xff);
  granulepos= (granulepos<<8)|(page[10]&0xff);
  granulepos= (granulepos<<8)|(page[9]&0xff);
  granulepos= (granulepos<<8)|(page[8]&0xff);
  granulepos= (granulepos<<8)|(page[7]&0xff);
  granulepos= (granulepos<<8)|(page[6]&0xff);
  return((ogg_int64_t)granulepos);
}

int ogg_page_serialno(const ogg_page *og){
  return((int)((ogg_uint32_t)og->header[14]) |
              ((ogg_uint32_t)og->header[15]<<8) |
              ((ogg_uint32_t)og->header[16]<<16) |
              ((ogg_uint32_t)og->header[17]<<24));
}

long ogg_page_pageno(const ogg_page *og){
  return((long)((ogg_uint32_t)og->header[18]) |
               ((ogg_uint32_t)og->header[19]<<8) |
               ((ogg_uint32_t)og->header[20]<<16) |
               ((ogg_uint32_t)og->header[21]<<24));
}



/* returns the number of packets that are completed on this page (if
   the leading packet is begun on a previous page, but ends on this
   page, it's counted */

/* NOTE:
   If a page consists of a packet begun on a previous page, and a new
   packet begun (but not completed) on this page, the return will be:
     ogg_page_packets(page)   ==1,
     ogg_page_continued(page) !=0

   If a page happens to be a single packet that was begun on a
   previous page, and spans to the next page (in the case of a three or
   more page packet), the return will be:
     ogg_page_packets(page)   ==0,
     ogg_page_continued(page) !=0
*/

int ogg_page_packets(const ogg_page *og){
  int i,n=og->header[26],count=0;
  for(i=0;i<n;i++)
    if(og->header[27+i]<255)count++;
  return(count);
}


#if 0
/* helper to initialize lookup for direct-table CRC (illustrative; we
   use the static init in crctable.h) */

static void _ogg_crc_init(){
  int i, j;
  ogg_uint32_t polynomial, crc;
  polynomial = 0x04c11db7; /* The same as the ethernet generator
                              polynomial, although we use an
                              unreflected alg and an init/final
                              of 0, not 0xffffffff */
  for (i = 0; i <= 0xFF; i++){
    crc = i << 24;

    for (j = 0; j < 8; j++)
      crc = (crc << 1) ^ (crc & (1 << 31) ? polynomial : 0);

    crc_lookup[0][i] = crc;
  }

  for (i = 0; i <= 0xFF; i++)
    for (j = 1; j < 8; j++)
      crc_lookup[j][i] = crc_lookup[0][(crc_lookup[j - 1][i] >> 24) & 0xFF] ^ (crc_lookup[j - 1][i] << 8);
}
#endif

#include "crctable.h"

/* init the encode/decode logical stream state */

int ogg_stream_init(ogg_stream_state *os,int serialno){
  if(os){
    memset(os,0,sizeof(*os));
    os->body_storage=16*1024;
    os->lacing_storage=1024;

    os->body_data=_ogg_malloc(os->body_storage*sizeof(*os->body_data));
    os->lacing_vals=_ogg_malloc(os->lacing_storage*sizeof(*os->lacing_vals));
    os->granule_vals=_ogg_malloc(os->lacing_storage*sizeof(*os->granule_vals));

    if(!os->body_data || !os->lacing_vals || !os->granule_vals){
      ogg_stream_clear(os);
      return -1;
    }

    os->serialno=serialno;

    return(0);
  }
  return(-1);
}

/* async/delayed error detection for the ogg_stream_state */
int ogg_stream_check(ogg_stream_state *os){
  if(!os || !os->body_data) return -1;
  return 0;
}

/* _clear does not free os, only the non-flat storage within */
int ogg_stream_clear(ogg_stream_state *os){
  if(os){
    if(os->body_data)_ogg_free(os->body_data);
    if(os->lacing_vals)_ogg_free(os->lacing_vals);
    if(os->granule_vals)_ogg_free(os->granule_vals);

    memset(os,0,sizeof(*os));
  }
  return(0);
}

int ogg_stream_destroy(ogg_stream_state *os){
  if(os){
    ogg_stream_clear(os);
    _ogg_free(os);
  }
  return(0);
}

/* Helpers for ogg_stream_encode; this keeps the structure and
   what's happening fairly clear */

static int _os_body_expand(ogg_stream_state *os,long needed){
  if(os->body_storage-needed<=os->body_fill){
    long body_storage;
    void *ret;
    if(os->body_storage>LONG_MAX-needed){
      ogg_stream_clear(os);
      return -1;
    }
    body_storage=os->body_storage+needed;
    if(body_storage<LONG_MAX-1024)body_storage+=1024;
    ret=_ogg_realloc(os->body_data,body_storage*sizeof(*os->body_data));
    if(!ret){
      ogg_stream_clear(os);
      return -1;
    }
    os->body_storage=body_storage;
    os->body_data=ret;
  }
  return 0;
}

static int _os_lacing_expand(ogg_stream_state *os,long needed){
  if(os->lacing_storage-needed<=os->lacing_fill){
    long lacing_storage;
    void *ret;
    if(os->lacing_storage>LONG_MAX-needed){
      ogg_stream_clear(os);
      return -1;
    }
    lacing_storage=os->lacing_storage+needed;
    if(lacing_storage<LONG_MAX-32)lacing_storage+=32;
    ret=_ogg_realloc(os->lacing_vals,lacing_storage*sizeof(*os->lacing_vals));
    if(!ret){
      ogg_stream_clear(os);
      return -1;
    }
    os->lacing_vals=ret;
    ret=_ogg_realloc(os->granule_vals,lacing_storage*
                     sizeof(*os->granule_vals));
    if(!ret){
      ogg_stream_clear(os);
      return -1;
    }
    os->granule_vals=ret;
    os->lacing_storage=lacing_storage;
  }
  return 0;
}

/* checksum the page */
/* Direct table CRC; note that this will be faster in the future if we
   perform the checksum simultaneously with other copies */

static ogg_uint32_t _os_update_crc(ogg_uint32_t crc, unsigned char *buffer, int size){
  while (size>=8){
    crc^=((ogg_uint32_t)buffer[0]<<24)|((ogg_uint32_t)buffer[1]<<16)|((ogg_uint32_t)buffer[2]<<8)|((ogg_uint32_t)buffer[3]);

    crc=crc_lookup[7][ crc>>24      ]^crc_lookup[6][(crc>>16)&0xFF]^
        crc_lookup[5][(crc>> 8)&0xFF]^crc_lookup[4][ crc     &0xFF]^
        crc_lookup[3][buffer[4]     ]^crc_lookup[2][buffer[5]     ]^
        crc_lookup[1][buffer[6]     ]^crc_lookup[0][buffer[7]     ];

    buffer+=8;
    size-=8;
  }

  while (size--)
    crc=(crc<<8)^crc_lookup[0][((crc >> 24)&0xff)^*buffer++];
  return crc;
}

void ogg_page_checksum_set(ogg_page *og){
  if(og){
    ogg_uint32_t crc_reg=0;

    /* safety; needed for API behavior, but not framing code */
    og->header[22]=0;
    og->header[23]=0;
    og->header[24]=0;
    og->header[25]=0;

    crc_reg=_os_update_crc(crc_reg,og->header,og->header_len);
    crc_reg=_os_update_crc(crc_reg,og->body,og->body_len);

    og->header[22]=(unsigned char)(crc_reg&0xff);
    og->header[23]=(unsigned char)((crc_reg>>8)&0xff);
    og->header[24]=(unsigned char)((crc_reg>>16)&0xff);
    og->header[25]=(unsigned char)((crc_reg>>24)&0xff);
  }
}

/* submit data to the internal buffer of the framing engine */
int ogg_stream_iovecin(ogg_stream_state *os, ogg_iovec_t *iov, int count,
                       long e_o_s, ogg_int64_t granulepos){

  long bytes = 0, lacing_vals;
  int i;

  if(ogg_stream_check(os)) return -1;
  if(!iov) return 0;

  for (i = 0; i < count; ++i){
    if(iov[i].iov_len>LONG_MAX) return -1;
    if(bytes>LONG_MAX-(long)iov[i].iov_len) return -1;
    bytes += (long)iov[i].iov_len;
  }
  lacing_vals=bytes/255+1;

  if(os->body_returned){
    /* advance packet data according to the body_returned pointer. We
       had to keep it around to return a pointer into the buffer last
       call */

    os->body_fill-=os->body_returned;
    if(os->body_fill)
      memmove(os->body_data,os->body_data+os->body_returned,
              os->body_fill);
    os->body_returned=0;
  }

  /* make sure we have the buffer storage */
  if(_os_body_expand(os,bytes) || _os_lacing_expand(os,lacing_vals))
    return -1;

  /* Copy in the submitted packet.  Yes, the copy is a waste; this is
     the liability of overly clean abstraction for the time being.  It
     will actually be fairly easy to eliminate the extra copy in the
     future */

  for (i = 0; i < count; ++i) {
    memcpy(os->body_data+os->body_fill, iov[i].iov_base, iov[i].iov_len);
    os->body_fill += (int)iov[i].iov_len;
  }

  /* Store lacing vals for this packet */
  for(i=0;i<lacing_vals-1;i++){
    os->lacing_vals[os->lacing_fill+i]=255;
    os->granule_vals[os->lacing_fill+i]=os->granulepos;
  }
  os->lacing_vals[os->lacing_fill+i]=bytes%255;
  os->granulepos=os->granule_vals[os->lacing_fill+i]=granulepos;

  /* flag the first segment as the beginning of the packet */
  os->lacing_vals[os->lacing_fill]|= 0x100;

  os->lacing_fill+=lacing_vals;

  /* for the sake of completeness */
  os->packetno++;

  if(e_o_s)os->e_o_s=1;

  return(0);
}

int ogg_stream_packetin(ogg_stream_state *os,ogg_packet *op){
  ogg_iovec_t iov;
  iov.iov_base = op->packet;
  iov.iov_len = op->bytes;
  return ogg_stream_iovecin(os, &iov, 1, op->e_o_s, op->granulepos);
}

/* Conditionally flush a page; force==0 will only flush nominal-size
   pages, force==1 forces us to flush a page regardless of page size
   so long as there's any data available at all. */
static int ogg_stream_flush_i(ogg_stream_state *os,ogg_page *og, int force, int nfill){
  int i;
  int vals=0;
  int maxvals=(os->lacing_fill>255?255:os->lacing_fill);
  int bytes=0;
  long acc=0;
  ogg_int64_t granule_pos=-1;

  if(ogg_stream_check(os)) return(0);
  if(maxvals==0) return(0);

  /* construct a page */
  /* decide how many segments to include */

  /* If this is the initial header case, the first page must only include
     the initial header packet */
  if(os->b_o_s==0){  /* 'initial header page' case */
    granule_pos=0;
    for(vals=0;vals<maxvals;vals++){
      if((os->lacing_vals[vals]&0x0ff)<255){
        vals++;
        break;
      }
    }
  }else{

    /* The extra packets_done, packet_just_done logic here attempts to do two things:
       1) Don't unnecessarily span pages.
       2) Unless necessary, don't flush pages if there are less than four packets on
          them; this expands page size to reduce unnecessary overhead if incoming packets
          are large.
       These are not necessary behaviors, just 'always better than naive flushing'
       without requiring an application to explicitly request a specific optimized
       behavior. We'll want an explicit behavior setup pathway eventually as well. */

    int packets_done=0;
    int packet_just_done=0;
    for(vals=0;vals<maxvals;vals++){
      if(acc>nfill && packet_just_done>=4){
        force=1;
        break;
      }
      acc+=os->lacing_vals[vals]&0x0ff;
      if((os->lacing_vals[vals]&0xff)<255){
        granule_pos=os->granule_vals[vals];
        packet_just_done=++packets_done;
      }else
        packet_just_done=0;
    }
    if(vals==255)force=1;
  }

  if(!force) return(0);

  /* construct the header in temp storage */
  memcpy(os->header,"OggS",4);

  /* stream structure version */
  os->header[4]=0x00;

  /* continued packet flag? */
  os->header[5]=0x00;
  if((os->lacing_vals[0]&0x100)==0)os->header[5]|=0x01;
  /* first page flag? */
  if(os->b_o_s==0)os->header[5]|=0x02;
  /* last page flag? */
  if(os->e_o_s && os->lacing_fill==vals)os->header[5]|=0x04;
  os->b_o_s=1;

  /* 64 bits of PCM position */
  for(i=6;i<14;i++){
    os->header[i]=(unsigned char)(granule_pos&0xff);
    granule_pos>>=8;
  }

  /* 32 bits of stream serial number */
  {
    long serialno=os->serialno;
    for(i=14;i<18;i++){
      os->header[i]=(unsigned char)(serialno&0xff);
      serialno>>=8;
    }
  }

  /* 32 bits of page counter (we have both counter and page header
     because this val can roll over) */
  if(os->pageno==-1)os->pageno=0; /* because someone called
                                     stream_reset; this would be a
                                     strange thing to do in an
                                     encode stream, but it has
                                     plausible uses */
  {
    long pageno=os->pageno++;
    for(i=18;i<22;i++){
      os->header[i]=(unsigned char)(pageno&0xff);
      pageno>>=8;
    }
  }

  /* zero for computation; filled in later */
  os->header[22]=0;
  os->header[23]=0;
  os->header[24]=0;
  os->header[25]=0;

  /* segment table */
  os->header[26]=(unsigned char)(vals&0xff);
  for(i=0;i<vals;i++)
    bytes+=os->header[i+27]=(unsigned char)(os->lacing_vals[i]&0xff);

  /* set pointers in the ogg_page struct */
  og->header=os->header;
  og->header_len=os->header_fill=vals+27;
  og->body=os->body_data+os->body_returned;
  og->body_len=bytes;

  /* advance the lacing data and set the body_returned pointer */

  os->lacing_fill-=vals;
  memmove(os->lacing_vals,os->lacing_vals+vals,os->lacing_fill*sizeof(*os->lacing_vals));
  memmove(os->granule_vals,os->granule_vals+vals,os->lacing_fill*sizeof(*os->granule_vals));
  os->body_returned+=bytes;

  /* calculate the checksum */

  ogg_page_checksum_set(og);

  /* done */
  return(1);
}

/* This will flush remaining packets into a page (returning nonzero),
   even if there is not enough data to trigger a flush normally
   (undersized page). If there are no packets or partial packets to
   flush, ogg_stream_flush returns 0.  Note that ogg_stream_flush will
   try to flush a normal sized page like ogg_stream_pageout; a call to
   ogg_stream_flush does not guarantee that all packets have flushed.
   Only a return value of 0 from ogg_stream_flush indicates all packet
   data is flushed into pages.

   since ogg_stream_flush will flush the last page in a stream even if
   it's undersized, you almost certainly want to use ogg_stream_pageout
   (and *not* ogg_stream_flush) unless you specifically need to flush
   a page regardless of size in the middle of a stream. */

int ogg_stream_flush(ogg_stream_state *os,ogg_page *og){
  return ogg_stream_flush_i(os,og,1,4096);
}

/* Like the above, but an argument is provided to adjust the nominal
   page size for applications which are smart enough to provide their
   own delay based flushing */

int ogg_stream_flush_fill(ogg_stream_state *os,ogg_page *og, int nfill){
  return ogg_stream_flush_i(os,og,1,nfill);
}

/* This constructs pages from buffered packet segments.  The pointers
returned are to static buffers; do not free. The returned buffers are
good only until the next call (using the same ogg_stream_state) */

int ogg_stream_pageout(ogg_stream_state *os, ogg_page *og){
  int force=0;
  if(ogg_stream_check(os)) return 0;

  if((os->e_o_s&&os->lacing_fill) ||          /* 'were done, now flush' case */
     (os->lacing_fill&&!os->b_o_s))           /* 'initial header page' case */
    force=1;

  return(ogg_stream_flush_i(os,og,force,4096));
}

/* Like the above, but an argument is provided to adjust the nominal
page size for applications which are smart enough to provide their
own delay based flushing */

int ogg_stream_pageout_fill(ogg_stream_state *os, ogg_page *og, int nfill){
  int force=0;
  if(ogg_stream_check(os)) return 0;

  if((os->e_o_s&&os->lacing_fill) ||          /* 'were done, now flush' case */
     (os->lacing_fill&&!os->b_o_s))           /* 'initial header page' case */
    force=1;

  return(ogg_stream_flush_i(os,og,force,nfill));
}

int ogg_stream_eos(ogg_stream_state *os){
  if(ogg_stream_check(os)) return 1;
  return os->e_o_s;
}

/* DECODING PRIMITIVES: packet streaming layer **********************/

/* This has two layers to place more of the multi-serialno and paging
   control in the application's hands.  First, we expose a data buffer
   using ogg_sync_buffer().  The app either copies into the
   buffer, or passes it directly to read(), etc.  We then call
   ogg_sync_wrote() to tell how many bytes we just added.

   Pages are returned (pointers into the buffer in ogg_sync_state)
   by ogg_sync_pageout().  The page is then submitted to
   ogg_stream_pagein() along with the appropriate
   ogg_stream_state* (ie, matching serialno).  We then get raw
   packets out calling ogg_stream_packetout() with a
   ogg_stream_state. */

/* initialize the struct to a known state */
int ogg_sync_init(ogg_sync_state *oy){
  if(oy){
    oy->storage = -1; /* used as a readiness flag */
    memset(oy,0,sizeof(*oy));
  }
  return(0);
}

/* clear non-flat storage within */
int ogg_sync_clear(ogg_sync_state *oy){
  if(oy){
    if(oy->data)_ogg_free(oy->data);
    memset(oy,0,sizeof(*oy));
  }
  return(0);
}

int ogg_sync_destroy(ogg_sync_state *oy){
  if(oy){
    ogg_sync_clear(oy);
    _ogg_free(oy);
  }
  return(0);
}

int ogg_sync_check(ogg_sync_state *oy){
  if(oy->storage<0) return -1;
  return 0;
}

char *ogg_sync_buffer(ogg_sync_state *oy, long size){
  if(ogg_sync_check(oy)) return NULL;

  /* first, clear out any space that has been previously returned */
  if(oy->returned){
    oy->fill-=oy->returned;
    if(oy->fill>0)
      memmove(oy->data,oy->data+oy->returned,oy->fill);
    oy->returned=0;
  }

  if(size>oy->storage-oy->fill){
    /* We need to extend the internal buffer */
    long newsize=size+oy->fill+4096; /* an extra page to be nice */
    void *ret;

    if(oy->data)
      ret=_ogg_realloc(oy->data,newsize);
    else
      ret=_ogg_malloc(newsize);
    if(!ret){
      ogg_sync_clear(oy);
      return NULL;
    }
    oy->data=ret;
    oy->storage=newsize;
  }

  /* expose a segment at least as large as requested at the fill mark */
  return((char *)oy->data+oy->fill);
}

int ogg_sync_wrote(ogg_sync_state *oy, long bytes){
  if(ogg_sync_check(oy))return -1;
  if(oy->fill+bytes>oy->storage)return -1;
  oy->fill+=bytes;
  return(0);
}

/* sync the stream.  This is meant to be useful for finding page
   boundaries.

   return values for this:
  -n) skipped n bytes
   0) page not ready; more data (no bytes skipped)
   n) page synced at current location; page length n bytes

*/

long ogg_sync_pageseek(ogg_sync_state *oy,ogg_page *og){
  unsigned char *page=oy->data+oy->returned;
  unsigned char *next;
  long bytes=oy->fill-oy->returned;

  if(ogg_sync_check(oy))return 0;

  if(oy->headerbytes==0){
    int headerbytes,i;
    if(bytes<27)return(0); /* not enough for a header */

    /* verify capture pattern */
    if(memcmp(page,"OggS",4))goto sync_fail;

    headerbytes=page[26]+27;
    if(bytes<headerbytes)return(0); /* not enough for header + seg table */

    /* count up body length in the segment table */

    for(i=0;i<page[26];i++)
      oy->bodybytes+=page[27+i];
    oy->headerbytes=headerbytes;
  }

  if(oy->bodybytes+oy->headerbytes>bytes)return(0);

  /* The whole test page is buffered.  Verify the checksum */
  {
    /* Grab the checksum bytes, set the header field to zero */
    char chksum[4];
    ogg_page log;

    memcpy(chksum,page+22,4);
    memset(page+22,0,4);

    /* set up a temp page struct and recompute the checksum */
    log.header=page;
    log.header_len=oy->headerbytes;
    log.body=page+oy->headerbytes;
    log.body_len=oy->bodybytes;
    ogg_page_checksum_set(&log);

    /* Compare */
    if(memcmp(chksum,page+22,4)){
      /* D'oh.  Mismatch! Corrupt page (or miscapture and not a page
         at all) */
      /* replace the computed checksum with the one actually read in */
      memcpy(page+22,chksum,4);

#ifndef DISABLE_CRC
      /* Bad checksum. Lose sync */
      goto sync_fail;
#endif
    }
  }

  /* yes, have a whole page all ready to go */
  {
    if(og){
      og->header=page;
      og->header_len=oy->headerbytes;
      og->body=page+oy->headerbytes;
      og->body_len=oy->bodybytes;
    }

    oy->unsynced=0;
    oy->returned+=(bytes=oy->headerbytes+oy->bodybytes);
    oy->headerbytes=0;
    oy->bodybytes=0;
    return(bytes);
  }

 sync_fail:

  oy->headerbytes=0;
  oy->bodybytes=0;

  /* search for possible capture */
  next=memchr(page+1,'O',bytes-1);
  if(!next)
    next=oy->data+oy->fill;

  oy->returned=(int)(next-oy->data);
  return((long)-(next-page));
}

/* sync the stream and get a page.  Keep trying until we find a page.
   Suppress 'sync errors' after reporting the first.

   return values:
   -1) recapture (hole in data)
    0) need more data
    1) page returned

   Returns pointers into buffered data; invalidated by next call to
   _stream, _clear, _init, or _buffer */

int ogg_sync_pageout(ogg_sync_state *oy, ogg_page *og){

  if(ogg_sync_check(oy))return 0;

  /* all we need to do is verify a page at the head of the stream
     buffer.  If it doesn't verify, we look for the next potential
     frame */

  for(;;){
    long ret=ogg_sync_pageseek(oy,og);
    if(ret>0){
      /* have a page */
      return(1);
    }
    if(ret==0){
      /* need more data */
      return(0);
    }

    /* head did not start a synced page... skipped some bytes */
    if(!oy->unsynced){
      oy->unsynced=1;
      return(-1);
    }

    /* loop. keep looking */

  }
}

/* add the incoming page to the stream state; we decompose the page
   into packet segments here as well. */

int ogg_stream_pagein(ogg_stream_state *os, ogg_page *og){
  unsigned char *header=og->header;
  unsigned char *body=og->body;
  long           bodysize=og->body_len;
  int            segptr=0;

  int version=ogg_page_version(og);
  int continued=ogg_page_continued(og);
  int bos=ogg_page_bos(og);
  int eos=ogg_page_eos(og);
  ogg_int64_t granulepos=ogg_page_granulepos(og);
  int serialno=ogg_page_serialno(og);
  long pageno=ogg_page_pageno(og);
  int segments=header[26];

  if(ogg_stream_check(os)) return -1;

  /* clean up 'returned data' */
  {
    long lr=os->lacing_returned;
    long br=os->body_returned;

    /* body data */
    if(br){
      os->body_fill-=br;
      if(os->body_fill)
        memmove(os->body_data,os->body_data+br,os->body_fill);
      os->body_returned=0;
    }

    if(lr){
      /* segment table */
      if(os->lacing_fill-lr){
        memmove(os->lacing_vals,os->lacing_vals+lr,
                (os->lacing_fill-lr)*sizeof(*os->lacing_vals));
        memmove(os->granule_vals,os->granule_vals+lr,
                (os->lacing_fill-lr)*sizeof(*os->granule_vals));
      }
      os->lacing_fill-=lr;
      os->lacing_packet-=lr;
      os->lacing_returned=0;
    }
  }

  /* check the serial number */
  if(serialno!=os->serialno)return(-1);
  if(version>0)return(-1);

  if(_os_lacing_expand(os,segments+1)) return -1;

  /* are we in sequence? */
  if(pageno!=os->pageno){
    int i;

    /* unroll previous partial packet (if any) */
    for(i=os->lacing_packet;i<os->lacing_fill;i++)
      os->body_fill-=os->lacing_vals[i]&0xff;
    os->lacing_fill=os->lacing_packet;

    /* make a note of dropped data in segment table */
    if(os->pageno!=-1){
      os->lacing_vals[os->lacing_fill++]=0x400;
      os->lacing_packet++;
    }
  }

  /* are we a 'continued packet' page?  If so, we may need to skip
     some segments */
  if(continued){
    if(os->lacing_fill<1 ||
       (os->lacing_vals[os->lacing_fill-1]&0xff)<255 ||
       os->lacing_vals[os->lacing_fill-1]==0x400){
      bos=0;
      for(;segptr<segments;segptr++){
        int val=header[27+segptr];
        body+=val;
        bodysize-=val;
        if(val<255){
          segptr++;
          break;
        }
      }
    }
  }

  if(bodysize){
    if(_os_body_expand(os,bodysize)) return -1;
    memcpy(os->body_data+os->body_fill,body,bodysize);
    os->body_fill+=bodysize;
  }

  {
    int saved=-1;
    while(segptr<segments){
      int val=header[27+segptr];
      os->lacing_vals[os->lacing_fill]=val;
      os->granule_vals[os->lacing_fill]=-1;

      if(bos){
        os->lacing_vals[os->lacing_fill]|=0x100;
        bos=0;
      }

      if(val<255)saved=os->lacing_fill;

      os->lacing_fill++;
      segptr++;

      if(val<255)os->lacing_packet=os->lacing_fill;
    }

    /* set the granulepos on the last granuleval of the last full packet */
    if(saved!=-1){
      os->granule_vals[saved]=granulepos;
    }

  }

  if(eos){
    os->e_o_s=1;
    if(os->lacing_fill>0)
      os->lacing_vals[os->lacing_fill-1]|=0x200;
  }

  os->pageno=pageno+1;

  return(0);
}

/* clear things to an initial state.  Good to call, eg, before seeking */
int ogg_sync_reset(ogg_sync_state *oy){
  if(ogg_sync_check(oy))return -1;

  oy->fill=0;
  oy->returned=0;
  oy->unsynced=0;
  oy->headerbytes=0;
  oy->bodybytes=0;
  return(0);
}

int ogg_stream_reset(ogg_stream_state *os){
  if(ogg_stream_check(os)) return -1;

  os->body_fill=0;
  os->body_returned=0;

  os->lacing_fill=0;
  os->lacing_packet=0;
  os->lacing_returned=0;

  os->header_fill=0;

  os->e_o_s=0;
  os->b_o_s=0;
  os->pageno=-1;
  os->packetno=0;
  os->granulepos=0;

  return(0);
}

int ogg_stream_reset_serialno(ogg_stream_state *os,int serialno){
  if(ogg_stream_check(os)) return -1;
  ogg_stream_reset(os);
  os->serialno=serialno;
  return(0);
}

static int _packetout(ogg_stream_state *os,ogg_packet *op,int adv){

  /* The last part of decode. We have the stream broken into packet
     segments.  Now we need to group them into packets (or return the
     out of sync markers) */

  int ptr=os->lacing_returned;

  if(os->lacing_packet<=ptr)return(0);

  if(os->lacing_vals[ptr]&0x400){
    /* we need to tell the codec there's a gap; it might need to
       handle previous packet dependencies. */
    os->lacing_returned++;
    os->packetno++;
    return(-1);
  }

  if(!op && !adv)return(1); /* just using peek as an inexpensive way
                               to ask if there's a whole packet
                               waiting */

  /* Gather the whole packet. We'll have no holes or a partial packet */
  {
    int size=os->lacing_vals[ptr]&0xff;
    long bytes=size;
    int eos=os->lacing_vals[ptr]&0x200; /* last packet of the stream? */
    int bos=os->lacing_vals[ptr]&0x100; /* first packet of the stream? */

    while(size==255){
      int val=os->lacing_vals[++ptr];
      size=val&0xff;
      if(val&0x200)eos=0x200;
      bytes+=size;
    }

    if(op){
      op->e_o_s=eos;
      op->b_o_s=bos;
      op->packet=os->body_data+os->body_returned;
      op->packetno=os->packetno;
      op->granulepos=os->granule_vals[ptr];
      op->bytes=bytes;
    }

    if(adv){
      os->body_returned+=bytes;
      os->lacing_returned=ptr+1;
      os->packetno++;
    }
  }
  return(1);
}

int ogg_stream_packetout(ogg_stream_state *os,ogg_packet *op){
  if(ogg_stream_check(os)) return 0;
  return _packetout(os,op,1);
}

int ogg_stream_packetpeek(ogg_stream_state *os,ogg_packet *op){
  if(ogg_stream_check(os)) return 0;
  return _packetout(os,op,0);
}

void ogg_packet_clear(ogg_packet *op) {
  _ogg_free(op->packet);
  memset(op, 0, sizeof(*op));
}

#ifdef _V_SELFTEST
#include <stdio.h>

ogg_stream_state os_en, os_de;
ogg_sync_state oy;

void checkpacket(ogg_packet *op,long len, int no, long pos){
  long j;
  static int sequence=0;
  static int lastno=0;

  if(op->bytes!=len){
    fprintf(stderr,"incorrect packet length (%ld != %ld)!\n",op->bytes,len);
    exit(1);
  }
  if(op->granulepos!=pos){
    fprintf(stderr,"incorrect packet granpos (%ld != %ld)!\n",(long)op->granulepos,pos);
    exit(1);
  }

  /* packet number just follows sequence/gap; adjust the input number
     for that */
  if(no==0){
    sequence=0;
  }else{
    sequence++;
    if(no>lastno+1)
      sequence++;
  }
  lastno=no;
  if(op->packetno!=sequence){
    fprintf(stderr,"incorrect packet sequence %ld != %d\n",
            (long)(op->packetno),sequence);
    exit(1);
  }

  /* Test data */
  for(j=0;j<op->bytes;j++)
    if(op->packet[j]!=((j+no)&0xff)){
      fprintf(stderr,"body data mismatch (1) at pos %ld: %x!=%lx!\n\n",
              j,op->packet[j],(j+no)&0xff);
      exit(1);
    }
}

void check_page(unsigned char *data,const int *header,ogg_page *og){
  long j;
  /* Test data */
  for(j=0;j<og->body_len;j++)
    if(og->body[j]!=data[j]){
      fprintf(stderr,"body data mismatch (2) at pos %ld: %x!=%x!\n\n",
              j,data[j],og->body[j]);
      exit(1);
    }

  /* Test header */
  for(j=0;j<og->header_len;j++){
    if(og->header[j]!=header[j]){
      fprintf(stderr,"header content mismatch at pos %ld:\n",j);
      for(j=0;j<header[26]+27;j++)
        fprintf(stderr," (%ld)%02x:%02x",j,header[j],og->header[j]);
      fprintf(stderr,"\n");
      exit(1);
    }
  }
  if(og->header_len!=header[26]+27){
    fprintf(stderr,"header length incorrect! (%ld!=%d)\n",
            og->header_len,header[26]+27);
    exit(1);
  }
}

void print_header(ogg_page *og){
  int j;
  fprintf(stderr,"\nHEADER:\n");
  fprintf(stderr,"  capture: %c %c %c %c  version: %d  flags: %x\n",
          og->header[0],og->header[1],og->header[2],og->header[3],
          (int)og->header[4],(int)og->header[5]);

  fprintf(stderr,"  granulepos: %d  serialno: %d  pageno: %ld\n",
          (og->header[9]<<24)|(og->header[8]<<16)|
          (og->header[7]<<8)|og->header[6],
          (og->header[17]<<24)|(og->header[16]<<16)|
          (og->header[15]<<8)|og->header[14],
          ((long)(og->header[21])<<24)|(og->header[20]<<16)|
          (og->header[19]<<8)|og->header[18]);

  fprintf(stderr,"  checksum: %02x:%02x:%02x:%02x\n  segments: %d (",
          (int)og->header[22],(int)og->header[23],
          (int)og->header[24],(int)og->header[25],
          (int)og->header[26]);

  for(j=27;j<og->header_len;j++)
    fprintf(stderr,"%d ",(int)og->header[j]);
  fprintf(stderr,")\n\n");
}

void copy_page(ogg_page *og){
  unsigned char *temp=_ogg_malloc(og->header_len);
  memcpy(temp,og->header,og->header_len);
  og->header=temp;

  temp=_ogg_malloc(og->body_len);
  memcpy(temp,og->body,og->body_len);
  og->body=temp;
}

void free_page(ogg_page *og){
  _ogg_free (og->header);
  _ogg_free (og->body);
}

void error(void){
  fprintf(stderr,"error!\n");
  exit(1);
}

/* 17 only */
const int head1_0[] = {0x4f,0x67,0x67,0x53,0,0x06,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0x15,0xed,0xec,0x91,
                       1,
                       17};

/* 17, 254, 255, 256, 500, 510, 600 byte, pad */
const int head1_1[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0x59,0x10,0x6c,0x2c,
                       1,
                       17};
const int head2_1[] = {0x4f,0x67,0x67,0x53,0,0x04,
                       0x07,0x18,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0x89,0x33,0x85,0xce,
                       13,
                       254,255,0,255,1,255,245,255,255,0,
                       255,255,90};

/* nil packets; beginning,middle,end */
const int head1_2[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0xff,0x7b,0x23,0x17,
                       1,
                       0};
const int head2_2[] = {0x4f,0x67,0x67,0x53,0,0x04,
                       0x07,0x28,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0x5c,0x3f,0x66,0xcb,
                       17,
                       17,254,255,0,0,255,1,0,255,245,255,255,0,
                       255,255,90,0};

/* large initial packet */
const int head1_3[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0x01,0x27,0x31,0xaa,
                       18,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,255,10};

const int head2_3[] = {0x4f,0x67,0x67,0x53,0,0x04,
                       0x07,0x08,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0x7f,0x4e,0x8a,0xd2,
                       4,
                       255,4,255,0};


/* continuing packet test */
const int head1_4[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0xff,0x7b,0x23,0x17,
                       1,
                       0};

const int head2_4[] = {0x4f,0x67,0x67,0x53,0,0x00,
                       0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0xf8,0x3c,0x19,0x79,
                       255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255};

const int head3_4[] = {0x4f,0x67,0x67,0x53,0,0x05,
                       0x07,0x0c,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,2,0,0,0,
                       0x38,0xe6,0xb6,0x28,
                       6,
                       255,220,255,4,255,0};


/* spill expansion test */
const int head1_4b[] = {0x4f,0x67,0x67,0x53,0,0x02,
                        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                        0x01,0x02,0x03,0x04,0,0,0,0,
                        0xff,0x7b,0x23,0x17,
                        1,
                        0};

const int head2_4b[] = {0x4f,0x67,0x67,0x53,0,0x00,
                        0x07,0x10,0x00,0x00,0x00,0x00,0x00,0x00,
                        0x01,0x02,0x03,0x04,1,0,0,0,
                        0xce,0x8f,0x17,0x1a,
                        23,
                        255,255,255,255,255,255,255,255,
                        255,255,255,255,255,255,255,255,255,10,255,4,255,0,0};


const int head3_4b[] = {0x4f,0x67,0x67,0x53,0,0x04,
                        0x07,0x14,0x00,0x00,0x00,0x00,0x00,0x00,
                        0x01,0x02,0x03,0x04,2,0,0,0,
                        0x9b,0xb2,0x50,0xa1,
                        1,
                        0};

/* page with the 255 segment limit */
const int head1_5[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0xff,0x7b,0x23,0x17,
                       1,
                       0};

const int head2_5[] = {0x4f,0x67,0x67,0x53,0,0x00,
                       0x07,0xfc,0x03,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0xed,0x2a,0x2e,0xa7,
                       255,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10,10,
                       10,10,10,10,10,10,10};

const int head3_5[] = {0x4f,0x67,0x67,0x53,0,0x04,
                       0x07,0x00,0x04,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,2,0,0,0,
                       0x6c,0x3b,0x82,0x3d,
                       1,
                       50};


/* packet that overspans over an entire page */
const int head1_6[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0xff,0x7b,0x23,0x17,
                       1,
                       0};

const int head2_6[] = {0x4f,0x67,0x67,0x53,0,0x00,
                       0x07,0x04,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0x68,0x22,0x7c,0x3d,
                       255,
                       100,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255};

const int head3_6[] = {0x4f,0x67,0x67,0x53,0,0x01,
                       0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
                       0x01,0x02,0x03,0x04,2,0,0,0,
                       0xf4,0x87,0xba,0xf3,
                       255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255};

const int head4_6[] = {0x4f,0x67,0x67,0x53,0,0x05,
                       0x07,0x10,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,3,0,0,0,
                       0xf7,0x2f,0x6c,0x60,
                       5,
                       254,255,4,255,0};

/* packet that overspans over an entire page */
const int head1_7[] = {0x4f,0x67,0x67,0x53,0,0x02,
                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,0,0,0,0,
                       0xff,0x7b,0x23,0x17,
                       1,
                       0};

const int head2_7[] = {0x4f,0x67,0x67,0x53,0,0x00,
                       0x07,0x04,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,1,0,0,0,
                       0x68,0x22,0x7c,0x3d,
                       255,
                       100,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255,255,255,
                       255,255,255,255,255,255};

const int head3_7[] = {0x4f,0x67,0x67,0x53,0,0x05,
                       0x07,0x08,0x00,0x00,0x00,0x00,0x00,0x00,
                       0x01,0x02,0x03,0x04,2,0,0,0,
                       0xd4,0xe0,0x60,0xe5,
                       1,
                       0};

int compare_packet(const ogg_packet *op1, const ogg_packet *op2){
  if(op1->packet!=op2->packet){
    fprintf(stderr,"op1->packet != op2->packet\n");
    return(1);
  }
  if(op1->bytes!=op2->bytes){
    fprintf(stderr,"op1->bytes != op2->bytes\n");
    return(1);
  }
  if(op1->b_o_s!=op2->b_o_s){
    fprintf(stderr,"op1->b_o_s != op2->b_o_s\n");
    return(1);
  }
  if(op1->e_o_s!=op2->e_o_s){
    fprintf(stderr,"op1->e_o_s != op2->e_o_s\n");
    return(1);
  }
  if(op1->granulepos!=op2->granulepos){
    fprintf(stderr,"op1->granulepos != op2->granulepos\n");
    return(1);
  }
  if(op1->packetno!=op2->packetno){
    fprintf(stderr,"op1->packetno != op2->packetno\n");
    return(1);
  }
  return(0);
}

void test_pack(const int *pl, const int **headers, int byteskip,
               int pageskip, int packetskip){
  unsigned char *data=_ogg_malloc(1024*1024); /* for scripted test cases only */
  long inptr=0;
  long outptr=0;
  long deptr=0;
  long depacket=0;
  long granule_pos=7,pageno=0;
  int i,j,packets,pageout=pageskip;
  int eosflag=0;
  int bosflag=0;

  int byteskipcount=0;

  ogg_stream_reset(&os_en);
  ogg_stream_reset(&os_de);
  ogg_sync_reset(&oy);

  for(packets=0;packets<packetskip;packets++)
    depacket+=pl[packets];

  for(packets=0;;packets++)if(pl[packets]==-1)break;

  for(i=0;i<packets;i++){
    /* construct a test packet */
    ogg_packet op;
    int len=pl[i];

    op.packet=data+inptr;
    op.bytes=len;
    op.e_o_s=(pl[i+1]<0?1:0);
    op.granulepos=granule_pos;

    granule_pos+=1024;

    for(j=0;j<len;j++)data[inptr++]=i+j;

    /* submit the test packet */
    ogg_stream_packetin(&os_en,&op);

    /* retrieve any finished pages */
    {
      ogg_page og;

      while(ogg_stream_pageout(&os_en,&og)){
        /* We have a page.  Check it carefully */

        fprintf(stderr,"%ld, ",pageno);

        if(headers[pageno]==NULL){
          fprintf(stderr,"coded too many pages!\n");
          exit(1);
        }

        check_page(data+outptr,headers[pageno],&og);

        outptr+=og.body_len;
        pageno++;
        if(pageskip){
          bosflag=1;
          pageskip--;
          deptr+=og.body_len;
        }

        /* have a complete page; submit it to sync/decode */

        {
          ogg_page og_de;
          ogg_packet op_de,op_de2;
          char *buf=ogg_sync_buffer(&oy,og.header_len+og.body_len);
          char *next=buf;
          byteskipcount+=og.header_len;
          if(byteskipcount>byteskip){
            memcpy(next,og.header,byteskipcount-byteskip);
            next+=byteskipcount-byteskip;
            byteskipcount=byteskip;
          }

          byteskipcount+=og.body_len;
          if(byteskipcount>byteskip){
            memcpy(next,og.body,byteskipcount-byteskip);
            next+=byteskipcount-byteskip;
            byteskipcount=byteskip;
          }

          ogg_sync_wrote(&oy,next-buf);

          while(1){
            int ret=ogg_sync_pageout(&oy,&og_de);
            if(ret==0)break;
            if(ret<0)continue;
            /* got a page.  Happy happy.  Verify that it's good. */

            fprintf(stderr,"(%d), ",pageout);

            check_page(data+deptr,headers[pageout],&og_de);
            deptr+=og_de.body_len;
            pageout++;

            /* submit it to deconstitution */
            ogg_stream_pagein(&os_de,&og_de);

            /* packets out? */
            while(ogg_stream_packetpeek(&os_de,&op_de2)>0){
              ogg_stream_packetpeek(&os_de,NULL);
              ogg_stream_packetout(&os_de,&op_de); /* just catching them all */

              /* verify peek and out match */
              if(compare_packet(&op_de,&op_de2)){
                fprintf(stderr,"packetout != packetpeek! pos=%ld\n",
                        depacket);
                exit(1);
              }

              /* verify the packet! */
              /* check data */
              if(memcmp(data+depacket,op_de.packet,op_de.bytes)){
                fprintf(stderr,"packet data mismatch in decode! pos=%ld\n",
                        depacket);
                exit(1);
              }
              /* check bos flag */
              if(bosflag==0 && op_de.b_o_s==0){
                fprintf(stderr,"b_o_s flag not set on packet!\n");
                exit(1);
              }
              if(bosflag && op_de.b_o_s){
                fprintf(stderr,"b_o_s flag incorrectly set on packet!\n");
                exit(1);
              }
              bosflag=1;
              depacket+=op_de.bytes;

              /* check eos flag */
              if(eosflag){
                fprintf(stderr,"Multiple decoded packets with eos flag!\n");
                exit(1);
              }

              if(op_de.e_o_s)eosflag=1;

              /* check granulepos flag */
              if(op_de.granulepos!=-1){
                fprintf(stderr," granule:%ld ",(long)op_de.granulepos);
              }
            }
          }
        }
      }
    }
  }
  _ogg_free(data);
  if(headers[pageno]!=NULL){
    fprintf(stderr,"did not write last page!\n");
    exit(1);
  }
  if(headers[pageout]!=NULL){
    fprintf(stderr,"did not decode last page!\n");
    exit(1);
  }
  if(inptr!=outptr){
    fprintf(stderr,"encoded page data incomplete!\n");
    exit(1);
  }
  if(inptr!=deptr){
    fprintf(stderr,"decoded page data incomplete!\n");
    exit(1);
  }
  if(inptr!=depacket){
    fprintf(stderr,"decoded packet data incomplete!\n");
    exit(1);
  }
  if(!eosflag){
    fprintf(stderr,"Never got a packet with EOS set!\n");
    exit(1);
  }
  fprintf(stderr,"ok.\n");
}

int main(void){

  ogg_stream_init(&os_en,0x04030201);
  ogg_stream_init(&os_de,0x04030201);
  ogg_sync_init(&oy);

  /* Exercise each code path in the framing code.  Also verify that
     the checksums are working.  */

  {
    /* 17 only */
    const int packets[]={17, -1};
    const int *headret[]={head1_0,NULL};

    fprintf(stderr,"testing single page encoding... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* 17, 254, 255, 256, 500, 510, 600 byte, pad */
    const int packets[]={17, 254, 255, 256, 500, 510, 600, -1};
    const int *headret[]={head1_1,head2_1,NULL};

    fprintf(stderr,"testing basic page encoding... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* nil packets; beginning,middle,end */
    const int packets[]={0,17, 254, 255, 0, 256, 0, 500, 510, 600, 0, -1};
    const int *headret[]={head1_2,head2_2,NULL};

    fprintf(stderr,"testing basic nil packets... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* large initial packet */
    const int packets[]={4345,259,255,-1};
    const int *headret[]={head1_3,head2_3,NULL};

    fprintf(stderr,"testing initial-packet lacing > 4k... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* continuing packet test; with page spill expansion, we have to
       overflow the lacing table. */
    const int packets[]={0,65500,259,255,-1};
    const int *headret[]={head1_4,head2_4,head3_4,NULL};

    fprintf(stderr,"testing single packet page span... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* spill expand packet test */
    const int packets[]={0,4345,259,255,0,0,-1};
    const int *headret[]={head1_4b,head2_4b,head3_4b,NULL};

    fprintf(stderr,"testing page spill expansion... ");
    test_pack(packets,headret,0,0,0);
  }

  /* page with the 255 segment limit */
  {

    const int packets[]={0,10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,10,
                   10,10,10,10,10,10,10,50,-1};
    const int *headret[]={head1_5,head2_5,head3_5,NULL};

    fprintf(stderr,"testing max packet segments... ");
    test_pack(packets,headret,0,0,0);
  }

  {
    /* packet that overspans over an entire page */
    const int packets[]={0,100,130049,259,255,-1};
    const int *headret[]={head1_6,head2_6,head3_6,head4_6,NULL};

    fprintf(stderr,"testing very large packets... ");
    test_pack(packets,headret,0,0,0);
  }

#ifndef DISABLE_CRC
  {
    /* test for the libogg 1.1.1 resync in large continuation bug
       found by Josh Coalson)  */
    const int packets[]={0,100,130049,259,255,-1};
    const int *headret[]={head1_6,head2_6,head3_6,head4_6,NULL};

    fprintf(stderr,"testing continuation resync in very large packets... ");
    test_pack(packets,headret,100,2,3);
  }
#else
    fprintf(stderr,"Skipping continuation resync test due to --disable-crc\n");
#endif

  {
    /* term only page.  why not? */
    const int packets[]={0,100,64770,-1};
    const int *headret[]={head1_7,head2_7,head3_7,NULL};

    fprintf(stderr,"testing zero data page (1 nil packet)... ");
    test_pack(packets,headret,0,0,0);
  }



  {
    /* build a bunch of pages for testing */
    unsigned char *data=_ogg_malloc(1024*1024);
    int pl[]={0, 1,1,98,4079, 1,1,2954,2057, 76,34,912,0,234,1000,1000, 1000,300,-1};
    int inptr=0,i,j;
    ogg_page og[5];

    ogg_stream_reset(&os_en);

    for(i=0;pl[i]!=-1;i++){
      ogg_packet op;
      int len=pl[i];

      op.packet=data+inptr;
      op.bytes=len;
      op.e_o_s=(pl[i+1]<0?1:0);
      op.granulepos=(i+1)*1000;

      for(j=0;j<len;j++)data[inptr++]=i+j;
      ogg_stream_packetin(&os_en,&op);
    }

    _ogg_free(data);

    /* retrieve finished pages */
    for(i=0;i<5;i++){
      if(ogg_stream_pageout(&os_en,&og[i])==0){
        fprintf(stderr,"Too few pages output building sync tests!\n");
        exit(1);
      }
      copy_page(&og[i]);
    }

    /* Test lost pages on pagein/packetout: no rollback */
    {
      ogg_page temp;
      ogg_packet test;

      fprintf(stderr,"Testing loss of pages... ");

      ogg_sync_reset(&oy);
      ogg_stream_reset(&os_de);
      for(i=0;i<5;i++){
        memcpy(ogg_sync_buffer(&oy,og[i].header_len),og[i].header,
               og[i].header_len);
        ogg_sync_wrote(&oy,og[i].header_len);
        memcpy(ogg_sync_buffer(&oy,og[i].body_len),og[i].body,og[i].body_len);
        ogg_sync_wrote(&oy,og[i].body_len);
      }

      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);
      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);
      ogg_sync_pageout(&oy,&temp);
      /* skip */
      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);

      /* do we get the expected results/packets? */

      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,0,0,0);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,1,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,2,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,98,3,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,4079,4,5000);
      if(ogg_stream_packetout(&os_de,&test)!=-1){
        fprintf(stderr,"Error: loss of page did not return error\n");
        exit(1);
      }
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,76,9,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,34,10,-1);
      fprintf(stderr,"ok.\n");
    }

    /* Test lost pages on pagein/packetout: rollback with continuation */
    {
      ogg_page temp;
      ogg_packet test;

      fprintf(stderr,"Testing loss of pages (rollback required)... ");

      ogg_sync_reset(&oy);
      ogg_stream_reset(&os_de);
      for(i=0;i<5;i++){
        memcpy(ogg_sync_buffer(&oy,og[i].header_len),og[i].header,
               og[i].header_len);
        ogg_sync_wrote(&oy,og[i].header_len);
        memcpy(ogg_sync_buffer(&oy,og[i].body_len),og[i].body,og[i].body_len);
        ogg_sync_wrote(&oy,og[i].body_len);
      }

      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);
      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);
      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);
      ogg_sync_pageout(&oy,&temp);
      /* skip */
      ogg_sync_pageout(&oy,&temp);
      ogg_stream_pagein(&os_de,&temp);

      /* do we get the expected results/packets? */

      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,0,0,0);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,1,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,2,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,98,3,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,4079,4,5000);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,5,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,1,6,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,2954,7,-1);
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,2057,8,9000);
      if(ogg_stream_packetout(&os_de,&test)!=-1){
        fprintf(stderr,"Error: loss of page did not return error\n");
        exit(1);
      }
      if(ogg_stream_packetout(&os_de,&test)!=1)error();
      checkpacket(&test,300,17,18000);
      fprintf(stderr,"ok.\n");
    }

    /* the rest only test sync */
    {
      ogg_page og_de;
      /* Test fractional page inputs: incomplete capture */
      fprintf(stderr,"Testing sync on partial inputs... ");
      ogg_sync_reset(&oy);
      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header,
             3);
      ogg_sync_wrote(&oy,3);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      /* Test fractional page inputs: incomplete fixed header */
      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header+3,
             20);
      ogg_sync_wrote(&oy,20);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      /* Test fractional page inputs: incomplete header */
      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header+23,
             5);
      ogg_sync_wrote(&oy,5);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      /* Test fractional page inputs: incomplete body */

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header+28,
             og[1].header_len-28);
      ogg_sync_wrote(&oy,og[1].header_len-28);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,1000);
      ogg_sync_wrote(&oy,1000);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body+1000,
             og[1].body_len-1000);
      ogg_sync_wrote(&oy,og[1].body_len-1000);
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();

      fprintf(stderr,"ok.\n");
    }

    /* Test fractional page inputs: page + incomplete capture */
    {
      ogg_page og_de;
      fprintf(stderr,"Testing sync on 1+partial inputs... ");
      ogg_sync_reset(&oy);

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header,
             og[1].header_len);
      ogg_sync_wrote(&oy,og[1].header_len);

      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,
             og[1].body_len);
      ogg_sync_wrote(&oy,og[1].body_len);

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header,
             20);
      ogg_sync_wrote(&oy,20);
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header+20,
             og[1].header_len-20);
      ogg_sync_wrote(&oy,og[1].header_len-20);
      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,
             og[1].body_len);
      ogg_sync_wrote(&oy,og[1].body_len);
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();

      fprintf(stderr,"ok.\n");
    }

    /* Test recapture: garbage + page */
    {
      ogg_page og_de;
      fprintf(stderr,"Testing search for capture... ");
      ogg_sync_reset(&oy);

      /* 'garbage' */
      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,
             og[1].body_len);
      ogg_sync_wrote(&oy,og[1].body_len);

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header,
             og[1].header_len);
      ogg_sync_wrote(&oy,og[1].header_len);

      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,
             og[1].body_len);
      ogg_sync_wrote(&oy,og[1].body_len);

      memcpy(ogg_sync_buffer(&oy,og[2].header_len),og[2].header,
             20);
      ogg_sync_wrote(&oy,20);
      if(ogg_sync_pageout(&oy,&og_de)>0)error();
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();
      if(ogg_sync_pageout(&oy,&og_de)>0)error();

      memcpy(ogg_sync_buffer(&oy,og[2].header_len),og[2].header+20,
             og[2].header_len-20);
      ogg_sync_wrote(&oy,og[2].header_len-20);
      memcpy(ogg_sync_buffer(&oy,og[2].body_len),og[2].body,
             og[2].body_len);
      ogg_sync_wrote(&oy,og[2].body_len);
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();

      fprintf(stderr,"ok.\n");
    }

#ifndef DISABLE_CRC
    /* Test recapture: page + garbage + page */
    {
      ogg_page og_de;
      fprintf(stderr,"Testing recapture... ");
      ogg_sync_reset(&oy);

      memcpy(ogg_sync_buffer(&oy,og[1].header_len),og[1].header,
             og[1].header_len);
      ogg_sync_wrote(&oy,og[1].header_len);

      memcpy(ogg_sync_buffer(&oy,og[1].body_len),og[1].body,
             og[1].body_len);
      ogg_sync_wrote(&oy,og[1].body_len);

      memcpy(ogg_sync_buffer(&oy,og[2].header_len),og[2].header,
             og[2].header_len);
      ogg_sync_wrote(&oy,og[2].header_len);

      memcpy(ogg_sync_buffer(&oy,og[2].header_len),og[2].header,
             og[2].header_len);
      ogg_sync_wrote(&oy,og[2].header_len);

      if(ogg_sync_pageout(&oy,&og_de)<=0)error();

      memcpy(ogg_sync_buffer(&oy,og[2].body_len),og[2].body,
             og[2].body_len-5);
      ogg_sync_wrote(&oy,og[2].body_len-5);

      memcpy(ogg_sync_buffer(&oy,og[3].header_len),og[3].header,
             og[3].header_len);
      ogg_sync_wrote(&oy,og[3].header_len);

      memcpy(ogg_sync_buffer(&oy,og[3].body_len),og[3].body,
             og[3].body_len);
      ogg_sync_wrote(&oy,og[3].body_len);

      if(ogg_sync_pageout(&oy,&og_de)>0)error();
      if(ogg_sync_pageout(&oy,&og_de)<=0)error();

      fprintf(stderr,"ok.\n");
    }
#else
    fprintf(stderr,"Skipping recapture test due to --disable-crc\n");
#endif

    /* Free page data that was previously copied */
    {
      for(i=0;i<5;i++){
        free_page(&og[i]);
      }
    }
  }
  ogg_sync_clear(&oy);
  ogg_stream_clear(&os_en);
  ogg_stream_clear(&os_de);

  return(0);
}

#endif
