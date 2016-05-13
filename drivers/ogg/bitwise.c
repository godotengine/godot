/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE Ogg CONTAINER SOURCE CODE.              *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2010             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function: packing variable sized words into an octet stream
  last mod: $Id: bitwise.c 17287 2010-06-10 13:42:06Z tterribe $

 ********************************************************************/

/* We're 'LSb' endian; if we write a word but read individual bits,
   then we'll read the lsb first */

#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <ogg/ogg.h>

#define BUFFER_INCREMENT 256

static const unsigned long mask[]=
{0x00000000,0x00000001,0x00000003,0x00000007,0x0000000f,
 0x0000001f,0x0000003f,0x0000007f,0x000000ff,0x000001ff,
 0x000003ff,0x000007ff,0x00000fff,0x00001fff,0x00003fff,
 0x00007fff,0x0000ffff,0x0001ffff,0x0003ffff,0x0007ffff,
 0x000fffff,0x001fffff,0x003fffff,0x007fffff,0x00ffffff,
 0x01ffffff,0x03ffffff,0x07ffffff,0x0fffffff,0x1fffffff,
 0x3fffffff,0x7fffffff,0xffffffff };

static const unsigned int mask8B[]=
{0x00,0x80,0xc0,0xe0,0xf0,0xf8,0xfc,0xfe,0xff};

void oggpack_writeinit(oggpack_buffer *b){
  memset(b,0,sizeof(*b));
  b->ptr=b->buffer=_ogg_malloc(BUFFER_INCREMENT);
  b->buffer[0]='\0';
  b->storage=BUFFER_INCREMENT;
}

void oggpackB_writeinit(oggpack_buffer *b){
  oggpack_writeinit(b);
}

int oggpack_writecheck(oggpack_buffer *b){
  if(!b->ptr || !b->storage)return -1;
  return 0;
}

int oggpackB_writecheck(oggpack_buffer *b){
  return oggpack_writecheck(b);
}

void oggpack_writetrunc(oggpack_buffer *b,long bits){
  long bytes=bits>>3;
  if(b->ptr){
    bits-=bytes*8;
    b->ptr=b->buffer+bytes;
    b->endbit=bits;
    b->endbyte=bytes;
    *b->ptr&=mask[bits];
  }
}

void oggpackB_writetrunc(oggpack_buffer *b,long bits){
  long bytes=bits>>3;
  if(b->ptr){
    bits-=bytes*8;
    b->ptr=b->buffer+bytes;
    b->endbit=bits;
    b->endbyte=bytes;
    *b->ptr&=mask8B[bits];
  }
}

/* Takes only up to 32 bits. */
void oggpack_write(oggpack_buffer *b,unsigned long value,int bits){
  if(bits<0 || bits>32) goto err;
  if(b->endbyte>=b->storage-4){
    void *ret;
    if(!b->ptr)return;
    if(b->storage>LONG_MAX-BUFFER_INCREMENT) goto err;
    ret=_ogg_realloc(b->buffer,b->storage+BUFFER_INCREMENT);
    if(!ret) goto err;
    b->buffer=ret;
    b->storage+=BUFFER_INCREMENT;
    b->ptr=b->buffer+b->endbyte;
  }

  value&=mask[bits]; 
  bits+=b->endbit;

  b->ptr[0]|=value<<b->endbit;  
  
  if(bits>=8){
    b->ptr[1]=(unsigned char)(value>>(8-b->endbit));
    if(bits>=16){
      b->ptr[2]=(unsigned char)(value>>(16-b->endbit));
      if(bits>=24){
        b->ptr[3]=(unsigned char)(value>>(24-b->endbit));
        if(bits>=32){
          if(b->endbit)
            b->ptr[4]=(unsigned char)(value>>(32-b->endbit));
          else
            b->ptr[4]=0;
        }
      }
    }
  }

  b->endbyte+=bits/8;
  b->ptr+=bits/8;
  b->endbit=bits&7;
  return;
 err:
  oggpack_writeclear(b);
}

/* Takes only up to 32 bits. */
void oggpackB_write(oggpack_buffer *b,unsigned long value,int bits){
  if(bits<0 || bits>32) goto err;
  if(b->endbyte>=b->storage-4){
    void *ret;
    if(!b->ptr)return;
    if(b->storage>LONG_MAX-BUFFER_INCREMENT) goto err;
    ret=_ogg_realloc(b->buffer,b->storage+BUFFER_INCREMENT);
    if(!ret) goto err;
    b->buffer=ret;
    b->storage+=BUFFER_INCREMENT;
    b->ptr=b->buffer+b->endbyte;
  }

  value=(value&mask[bits])<<(32-bits); 
  bits+=b->endbit;

  b->ptr[0]|=value>>(24+b->endbit);  
  
  if(bits>=8){
    b->ptr[1]=(unsigned char)(value>>(16+b->endbit));
    if(bits>=16){
      b->ptr[2]=(unsigned char)(value>>(8+b->endbit));
      if(bits>=24){
        b->ptr[3]=(unsigned char)(value>>(b->endbit));
        if(bits>=32){
          if(b->endbit)
            b->ptr[4]=(unsigned char)(value<<(8-b->endbit));
          else
            b->ptr[4]=0;
        }
      }
    }
  }

  b->endbyte+=bits/8;
  b->ptr+=bits/8;
  b->endbit=bits&7;
  return;
 err:
  oggpack_writeclear(b);
}

void oggpack_writealign(oggpack_buffer *b){
  int bits=8-b->endbit;
  if(bits<8)
    oggpack_write(b,0,bits);
}

void oggpackB_writealign(oggpack_buffer *b){
  int bits=8-b->endbit;
  if(bits<8)
    oggpackB_write(b,0,bits);
}

static void oggpack_writecopy_helper(oggpack_buffer *b,
                                     void *source,
                                     long bits,
                                     void (*w)(oggpack_buffer *,
                                               unsigned long,
                                               int),
                                     int msb){
  unsigned char *ptr=(unsigned char *)source;

  long bytes=bits/8;
  bits-=bytes*8;

  if(b->endbit){
    int i;
    /* unaligned copy.  Do it the hard way. */
    for(i=0;i<bytes;i++)
      w(b,(unsigned long)(ptr[i]),8);    
  }else{
    /* aligned block copy */
    if(b->endbyte+bytes+1>=b->storage){
      void *ret;
      if(!b->ptr) goto err;
      if(b->endbyte+bytes+BUFFER_INCREMENT>b->storage) goto err;
      b->storage=b->endbyte+bytes+BUFFER_INCREMENT;
      ret=_ogg_realloc(b->buffer,b->storage);
      if(!ret) goto err;
      b->buffer=ret;
      b->ptr=b->buffer+b->endbyte;
    }

    memmove(b->ptr,source,bytes);
    b->ptr+=bytes;
    b->endbyte+=bytes;
    *b->ptr=0;

  }
  if(bits){
    if(msb)
      w(b,(unsigned long)(ptr[bytes]>>(8-bits)),bits);    
    else
      w(b,(unsigned long)(ptr[bytes]),bits);    
  }
  return;
 err:
  oggpack_writeclear(b);
}

void oggpack_writecopy(oggpack_buffer *b,void *source,long bits){
  oggpack_writecopy_helper(b,source,bits,oggpack_write,0);
}

void oggpackB_writecopy(oggpack_buffer *b,void *source,long bits){
  oggpack_writecopy_helper(b,source,bits,oggpackB_write,1);
}

void oggpack_reset(oggpack_buffer *b){
  if(!b->ptr)return;
  b->ptr=b->buffer;
  b->buffer[0]=0;
  b->endbit=b->endbyte=0;
}

void oggpackB_reset(oggpack_buffer *b){
  oggpack_reset(b);
}

void oggpack_writeclear(oggpack_buffer *b){
  if(b->buffer)_ogg_free(b->buffer);
  memset(b,0,sizeof(*b));
}

void oggpackB_writeclear(oggpack_buffer *b){
  oggpack_writeclear(b);
}

void oggpack_readinit(oggpack_buffer *b,unsigned char *buf,int bytes){
  memset(b,0,sizeof(*b));
  b->buffer=b->ptr=buf;
  b->storage=bytes;
}

void oggpackB_readinit(oggpack_buffer *b,unsigned char *buf,int bytes){
  oggpack_readinit(b,buf,bytes);
}

/* Read in bits without advancing the bitptr; bits <= 32 */
long oggpack_look(oggpack_buffer *b,int bits){
  unsigned long ret;
  unsigned long m;

  if(bits<0 || bits>32) return -1;
  m=mask[bits];
  bits+=b->endbit;

  if(b->endbyte >= b->storage-4){
    /* not the main path */
    if(b->endbyte > b->storage-((bits+7)>>3)) return -1;
    /* special case to avoid reading b->ptr[0], which might be past the end of
        the buffer; also skips some useless accounting */
    else if(!bits)return(0L);
  }

  ret=b->ptr[0]>>b->endbit;
  if(bits>8){
    ret|=b->ptr[1]<<(8-b->endbit);  
    if(bits>16){
      ret|=b->ptr[2]<<(16-b->endbit);  
      if(bits>24){
        ret|=b->ptr[3]<<(24-b->endbit);  
        if(bits>32 && b->endbit)
          ret|=b->ptr[4]<<(32-b->endbit);
      }
    }
  }
  return(m&ret);
}

/* Read in bits without advancing the bitptr; bits <= 32 */
long oggpackB_look(oggpack_buffer *b,int bits){
  unsigned long ret;
  int m=32-bits;

  if(m<0 || m>32) return -1;
  bits+=b->endbit;

  if(b->endbyte >= b->storage-4){
    /* not the main path */
    if(b->endbyte > b->storage-((bits+7)>>3)) return -1;
    /* special case to avoid reading b->ptr[0], which might be past the end of
        the buffer; also skips some useless accounting */
    else if(!bits)return(0L);
  }

  ret=b->ptr[0]<<(24+b->endbit);
  if(bits>8){
    ret|=b->ptr[1]<<(16+b->endbit);  
    if(bits>16){
      ret|=b->ptr[2]<<(8+b->endbit);  
      if(bits>24){
        ret|=b->ptr[3]<<(b->endbit);  
        if(bits>32 && b->endbit)
          ret|=b->ptr[4]>>(8-b->endbit);
      }
    }
  }
  return ((ret&0xffffffff)>>(m>>1))>>((m+1)>>1);
}

long oggpack_look1(oggpack_buffer *b){
  if(b->endbyte>=b->storage)return(-1);
  return((b->ptr[0]>>b->endbit)&1);
}

long oggpackB_look1(oggpack_buffer *b){
  if(b->endbyte>=b->storage)return(-1);
  return((b->ptr[0]>>(7-b->endbit))&1);
}

void oggpack_adv(oggpack_buffer *b,int bits){
  bits+=b->endbit;

  if(b->endbyte > b->storage-((bits+7)>>3)) goto overflow;

  b->ptr+=bits/8;
  b->endbyte+=bits/8;
  b->endbit=bits&7;
  return;

 overflow:
  b->ptr=NULL;
  b->endbyte=b->storage;
  b->endbit=1;
}

void oggpackB_adv(oggpack_buffer *b,int bits){
  oggpack_adv(b,bits);
}

void oggpack_adv1(oggpack_buffer *b){
  if(++(b->endbit)>7){
    b->endbit=0;
    b->ptr++;
    b->endbyte++;
  }
}

void oggpackB_adv1(oggpack_buffer *b){
  oggpack_adv1(b);
}

/* bits <= 32 */
long oggpack_read(oggpack_buffer *b,int bits){
  long ret;
  unsigned long m;

  if(bits<0 || bits>32) goto err;
  m=mask[bits];
  bits+=b->endbit;

  if(b->endbyte >= b->storage-4){
    /* not the main path */
    if(b->endbyte > b->storage-((bits+7)>>3)) goto overflow;
    /* special case to avoid reading b->ptr[0], which might be past the end of
        the buffer; also skips some useless accounting */
    else if(!bits)return(0L);
  }

  ret=b->ptr[0]>>b->endbit;
  if(bits>8){
    ret|=b->ptr[1]<<(8-b->endbit);  
    if(bits>16){
      ret|=b->ptr[2]<<(16-b->endbit);  
      if(bits>24){
        ret|=b->ptr[3]<<(24-b->endbit);  
        if(bits>32 && b->endbit){
          ret|=b->ptr[4]<<(32-b->endbit);
        }
      }
    }
  }
  ret&=m;
  b->ptr+=bits/8;
  b->endbyte+=bits/8;
  b->endbit=bits&7;
  return ret;

 overflow:
 err:
  b->ptr=NULL;
  b->endbyte=b->storage;
  b->endbit=1;
  return -1L;
}

/* bits <= 32 */
long oggpackB_read(oggpack_buffer *b,int bits){
  long ret;
  long m=32-bits;

  if(m<0 || m>32) goto err;
  bits+=b->endbit;

  if(b->endbyte+4>=b->storage){
    /* not the main path */
    if(b->endbyte > b->storage-((bits+7)>>3)) goto overflow;
    /* special case to avoid reading b->ptr[0], which might be past the end of
        the buffer; also skips some useless accounting */
    else if(!bits)return(0L);
  }

  ret=b->ptr[0]<<(24+b->endbit);
  if(bits>8){
    ret|=b->ptr[1]<<(16+b->endbit);  
    if(bits>16){
      ret|=b->ptr[2]<<(8+b->endbit);  
      if(bits>24){
        ret|=b->ptr[3]<<(b->endbit);  
        if(bits>32 && b->endbit)
          ret|=b->ptr[4]>>(8-b->endbit);
      }
    }
  }
  ret=((ret&0xffffffffUL)>>(m>>1))>>((m+1)>>1);

  b->ptr+=bits/8;
  b->endbyte+=bits/8;
  b->endbit=bits&7;
  return ret;

 overflow:
 err:
  b->ptr=NULL;
  b->endbyte=b->storage;
  b->endbit=1;
  return -1L;
}

long oggpack_read1(oggpack_buffer *b){
  long ret;

  if(b->endbyte >= b->storage) goto overflow;
  ret=(b->ptr[0]>>b->endbit)&1;

  b->endbit++;
  if(b->endbit>7){
    b->endbit=0;
    b->ptr++;
    b->endbyte++;
  }
  return ret;

 overflow:
  b->ptr=NULL;
  b->endbyte=b->storage;
  b->endbit=1;
  return -1L;
}

long oggpackB_read1(oggpack_buffer *b){
  long ret;

  if(b->endbyte >= b->storage) goto overflow;
  ret=(b->ptr[0]>>(7-b->endbit))&1;

  b->endbit++;
  if(b->endbit>7){
    b->endbit=0;
    b->ptr++;
    b->endbyte++;
  }
  return ret;

 overflow:
  b->ptr=NULL;
  b->endbyte=b->storage;
  b->endbit=1;
  return -1L;
}

long oggpack_bytes(oggpack_buffer *b){
  return(b->endbyte+(b->endbit+7)/8);
}

long oggpack_bits(oggpack_buffer *b){
  return(b->endbyte*8+b->endbit);
}

long oggpackB_bytes(oggpack_buffer *b){
  return oggpack_bytes(b);
}

long oggpackB_bits(oggpack_buffer *b){
  return oggpack_bits(b);
}
  
unsigned char *oggpack_get_buffer(oggpack_buffer *b){
  return(b->buffer);
}

unsigned char *oggpackB_get_buffer(oggpack_buffer *b){
  return oggpack_get_buffer(b);
}

/* Self test of the bitwise routines; everything else is based on
   them, so they damned well better be solid. */

#ifdef _V_SELFTEST
#include <stdio.h>

static int ilog(unsigned int v){
  int ret=0;
  while(v){
    ret++;
    v>>=1;
  }
  return(ret);
}
      
oggpack_buffer o;
oggpack_buffer r;

void report(char *in){
  fprintf(stderr,"%s",in);
  exit(1);
}

void cliptest(unsigned long *b,int vals,int bits,int *comp,int compsize){
  long bytes,i;
  unsigned char *buffer;

  oggpack_reset(&o);
  for(i=0;i<vals;i++)
    oggpack_write(&o,b[i],bits?bits:ilog(b[i]));
  buffer=oggpack_get_buffer(&o);
  bytes=oggpack_bytes(&o);
  if(bytes!=compsize)report("wrong number of bytes!\n");
  for(i=0;i<bytes;i++)if(buffer[i]!=comp[i]){
    for(i=0;i<bytes;i++)fprintf(stderr,"%x %x\n",(int)buffer[i],(int)comp[i]);
    report("wrote incorrect value!\n");
  }
  oggpack_readinit(&r,buffer,bytes);
  for(i=0;i<vals;i++){
    int tbit=bits?bits:ilog(b[i]);
    if(oggpack_look(&r,tbit)==-1)
      report("out of data!\n");
    if(oggpack_look(&r,tbit)!=(b[i]&mask[tbit]))
      report("looked at incorrect value!\n");
    if(tbit==1)
      if(oggpack_look1(&r)!=(b[i]&mask[tbit]))
        report("looked at single bit incorrect value!\n");
    if(tbit==1){
      if(oggpack_read1(&r)!=(b[i]&mask[tbit]))
        report("read incorrect single bit value!\n");
    }else{
    if(oggpack_read(&r,tbit)!=(b[i]&mask[tbit]))
      report("read incorrect value!\n");
    }
  }
  if(oggpack_bytes(&r)!=bytes)report("leftover bytes after read!\n");
}

void cliptestB(unsigned long *b,int vals,int bits,int *comp,int compsize){
  long bytes,i;
  unsigned char *buffer;
  
  oggpackB_reset(&o);
  for(i=0;i<vals;i++)
    oggpackB_write(&o,b[i],bits?bits:ilog(b[i]));
  buffer=oggpackB_get_buffer(&o);
  bytes=oggpackB_bytes(&o);
  if(bytes!=compsize)report("wrong number of bytes!\n");
  for(i=0;i<bytes;i++)if(buffer[i]!=comp[i]){
    for(i=0;i<bytes;i++)fprintf(stderr,"%x %x\n",(int)buffer[i],(int)comp[i]);
    report("wrote incorrect value!\n");
  }
  oggpackB_readinit(&r,buffer,bytes);
  for(i=0;i<vals;i++){
    int tbit=bits?bits:ilog(b[i]);
    if(oggpackB_look(&r,tbit)==-1)
      report("out of data!\n");
    if(oggpackB_look(&r,tbit)!=(b[i]&mask[tbit]))
      report("looked at incorrect value!\n");
    if(tbit==1)
      if(oggpackB_look1(&r)!=(b[i]&mask[tbit]))
        report("looked at single bit incorrect value!\n");
    if(tbit==1){
      if(oggpackB_read1(&r)!=(b[i]&mask[tbit]))
        report("read incorrect single bit value!\n");
    }else{
    if(oggpackB_read(&r,tbit)!=(b[i]&mask[tbit]))
      report("read incorrect value!\n");
    }
  }
  if(oggpackB_bytes(&r)!=bytes)report("leftover bytes after read!\n");
}

int main(void){
  unsigned char *buffer;
  long bytes,i;
  static unsigned long testbuffer1[]=
    {18,12,103948,4325,543,76,432,52,3,65,4,56,32,42,34,21,1,23,32,546,456,7,
       567,56,8,8,55,3,52,342,341,4,265,7,67,86,2199,21,7,1,5,1,4};
  int test1size=43;

  static unsigned long testbuffer2[]=
    {216531625L,1237861823,56732452,131,3212421,12325343,34547562,12313212,
       1233432,534,5,346435231,14436467,7869299,76326614,167548585,
       85525151,0,12321,1,349528352};
  int test2size=21;

  static unsigned long testbuffer3[]=
    {1,0,14,0,1,0,12,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,0,0,1,
       0,1,30,1,1,1,0,0,1,0,0,0,12,0,11,0,1,0,0,1};
  int test3size=56;

  static unsigned long large[]=
    {2136531625L,2137861823,56732452,131,3212421,12325343,34547562,12313212,
       1233432,534,5,2146435231,14436467,7869299,76326614,167548585,
       85525151,0,12321,1,2146528352};

  int onesize=33;
  static int one[33]={146,25,44,151,195,15,153,176,233,131,196,65,85,172,47,40,
                    34,242,223,136,35,222,211,86,171,50,225,135,214,75,172,
                    223,4};
  static int oneB[33]={150,101,131,33,203,15,204,216,105,193,156,65,84,85,222,
                       8,139,145,227,126,34,55,244,171,85,100,39,195,173,18,
                       245,251,128};

  int twosize=6;
  static int two[6]={61,255,255,251,231,29};
  static int twoB[6]={247,63,255,253,249,120};

  int threesize=54;
  static int three[54]={169,2,232,252,91,132,156,36,89,13,123,176,144,32,254,
                      142,224,85,59,121,144,79,124,23,67,90,90,216,79,23,83,
                      58,135,196,61,55,129,183,54,101,100,170,37,127,126,10,
                      100,52,4,14,18,86,77,1};
  static int threeB[54]={206,128,42,153,57,8,183,251,13,89,36,30,32,144,183,
                         130,59,240,121,59,85,223,19,228,180,134,33,107,74,98,
                         233,253,196,135,63,2,110,114,50,155,90,127,37,170,104,
                         200,20,254,4,58,106,176,144,0};

  int foursize=38;
  static int four[38]={18,6,163,252,97,194,104,131,32,1,7,82,137,42,129,11,72,
                     132,60,220,112,8,196,109,64,179,86,9,137,195,208,122,169,
                     28,2,133,0,1};
  static int fourB[38]={36,48,102,83,243,24,52,7,4,35,132,10,145,21,2,93,2,41,
                        1,219,184,16,33,184,54,149,170,132,18,30,29,98,229,67,
                        129,10,4,32};

  int fivesize=45;
  static int five[45]={169,2,126,139,144,172,30,4,80,72,240,59,130,218,73,62,
                     241,24,210,44,4,20,0,248,116,49,135,100,110,130,181,169,
                     84,75,159,2,1,0,132,192,8,0,0,18,22};
  static int fiveB[45]={1,84,145,111,245,100,128,8,56,36,40,71,126,78,213,226,
                        124,105,12,0,133,128,0,162,233,242,67,152,77,205,77,
                        172,150,169,129,79,128,0,6,4,32,0,27,9,0};

  int sixsize=7;
  static int six[7]={17,177,170,242,169,19,148};
  static int sixB[7]={136,141,85,79,149,200,41};

  /* Test read/write together */
  /* Later we test against pregenerated bitstreams */
  oggpack_writeinit(&o);

  fprintf(stderr,"\nSmall preclipped packing (LSb): ");
  cliptest(testbuffer1,test1size,0,one,onesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nNull bit call (LSb): ");
  cliptest(testbuffer3,test3size,0,two,twosize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nLarge preclipped packing (LSb): ");
  cliptest(testbuffer2,test2size,0,three,threesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\n32 bit preclipped packing (LSb): ");
  oggpack_reset(&o);
  for(i=0;i<test2size;i++)
    oggpack_write(&o,large[i],32);
  buffer=oggpack_get_buffer(&o);
  bytes=oggpack_bytes(&o);
  oggpack_readinit(&r,buffer,bytes);
  for(i=0;i<test2size;i++){
    if(oggpack_look(&r,32)==-1)report("out of data. failed!");
    if(oggpack_look(&r,32)!=large[i]){
      fprintf(stderr,"%ld != %ld (%lx!=%lx):",oggpack_look(&r,32),large[i],
              oggpack_look(&r,32),large[i]);
      report("read incorrect value!\n");
    }
    oggpack_adv(&r,32);
  }
  if(oggpack_bytes(&r)!=bytes)report("leftover bytes after read!\n");
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nSmall unclipped packing (LSb): ");
  cliptest(testbuffer1,test1size,7,four,foursize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nLarge unclipped packing (LSb): ");
  cliptest(testbuffer2,test2size,17,five,fivesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nSingle bit unclipped packing (LSb): ");
  cliptest(testbuffer3,test3size,1,six,sixsize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nTesting read past end (LSb): ");
  oggpack_readinit(&r,(unsigned char *)"\0\0\0\0\0\0\0\0",8);
  for(i=0;i<64;i++){
    if(oggpack_read(&r,1)!=0){
      fprintf(stderr,"failed; got -1 prematurely.\n");
      exit(1);
    }
  }
  if(oggpack_look(&r,1)!=-1 ||
     oggpack_read(&r,1)!=-1){
      fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  oggpack_readinit(&r,(unsigned char *)"\0\0\0\0\0\0\0\0",8);
  if(oggpack_read(&r,30)!=0 || oggpack_read(&r,16)!=0){
      fprintf(stderr,"failed 2; got -1 prematurely.\n");
      exit(1);
  }

  if(oggpack_look(&r,18)!=0 ||
     oggpack_look(&r,18)!=0){
    fprintf(stderr,"failed 3; got -1 prematurely.\n");
      exit(1);
  }
  if(oggpack_look(&r,19)!=-1 ||
     oggpack_look(&r,19)!=-1){
    fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  if(oggpack_look(&r,32)!=-1 ||
     oggpack_look(&r,32)!=-1){
    fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  oggpack_writeclear(&o);
  fprintf(stderr,"ok.\n");

  /********** lazy, cut-n-paste retest with MSb packing ***********/

  /* Test read/write together */
  /* Later we test against pregenerated bitstreams */
  oggpackB_writeinit(&o);

  fprintf(stderr,"\nSmall preclipped packing (MSb): ");
  cliptestB(testbuffer1,test1size,0,oneB,onesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nNull bit call (MSb): ");
  cliptestB(testbuffer3,test3size,0,twoB,twosize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nLarge preclipped packing (MSb): ");
  cliptestB(testbuffer2,test2size,0,threeB,threesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\n32 bit preclipped packing (MSb): ");
  oggpackB_reset(&o);
  for(i=0;i<test2size;i++)
    oggpackB_write(&o,large[i],32);
  buffer=oggpackB_get_buffer(&o);
  bytes=oggpackB_bytes(&o);
  oggpackB_readinit(&r,buffer,bytes);
  for(i=0;i<test2size;i++){
    if(oggpackB_look(&r,32)==-1)report("out of data. failed!");
    if(oggpackB_look(&r,32)!=large[i]){
      fprintf(stderr,"%ld != %ld (%lx!=%lx):",oggpackB_look(&r,32),large[i],
              oggpackB_look(&r,32),large[i]);
      report("read incorrect value!\n");
    }
    oggpackB_adv(&r,32);
  }
  if(oggpackB_bytes(&r)!=bytes)report("leftover bytes after read!\n");
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nSmall unclipped packing (MSb): ");
  cliptestB(testbuffer1,test1size,7,fourB,foursize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nLarge unclipped packing (MSb): ");
  cliptestB(testbuffer2,test2size,17,fiveB,fivesize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nSingle bit unclipped packing (MSb): ");
  cliptestB(testbuffer3,test3size,1,sixB,sixsize);
  fprintf(stderr,"ok.");

  fprintf(stderr,"\nTesting read past end (MSb): ");
  oggpackB_readinit(&r,(unsigned char *)"\0\0\0\0\0\0\0\0",8);
  for(i=0;i<64;i++){
    if(oggpackB_read(&r,1)!=0){
      fprintf(stderr,"failed; got -1 prematurely.\n");
      exit(1);
    }
  }
  if(oggpackB_look(&r,1)!=-1 ||
     oggpackB_read(&r,1)!=-1){
      fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  oggpackB_readinit(&r,(unsigned char *)"\0\0\0\0\0\0\0\0",8);
  if(oggpackB_read(&r,30)!=0 || oggpackB_read(&r,16)!=0){
      fprintf(stderr,"failed 2; got -1 prematurely.\n");
      exit(1);
  }

  if(oggpackB_look(&r,18)!=0 ||
     oggpackB_look(&r,18)!=0){
    fprintf(stderr,"failed 3; got -1 prematurely.\n");
      exit(1);
  }
  if(oggpackB_look(&r,19)!=-1 ||
     oggpackB_look(&r,19)!=-1){
    fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  if(oggpackB_look(&r,32)!=-1 ||
     oggpackB_look(&r,32)!=-1){
    fprintf(stderr,"failed; read past end without -1.\n");
      exit(1);
  }
  oggpackB_writeclear(&o);
  fprintf(stderr,"ok.\n\n");


  return(0);
}  
#endif  /* _V_SELFTEST */

#undef BUFFER_INCREMENT
