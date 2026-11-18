/* Copyright (c) 2001-2011 Timothy B. Terriberry
   Copyright (c) 2008-2009 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif
#include "os_support.h"
#include "arch.h"
#include "entenc.h"
#include "mfrngcod.h"

/*A range encoder.
  See entdec.c and the references for implementation details \cite{Mar79,MNW98}.

  @INPROCEEDINGS{Mar79,
   author="Martin, G.N.N.",
   title="Range encoding: an algorithm for removing redundancy from a digitised
    message",
   booktitle="Video \& Data Recording Conference",
   year=1979,
   address="Southampton",
   month=Jul
  }
  @ARTICLE{MNW98,
   author="Alistair Moffat and Radford Neal and Ian H. Witten",
   title="Arithmetic Coding Revisited",
   journal="{ACM} Transactions on Information Systems",
   year=1998,
   volume=16,
   number=3,
   pages="256--294",
   month=Jul,
   URL="http://www.stanford.edu/class/ee398/handouts/papers/Moffat98ArithmCoding.pdf"
  }*/

static int ec_write_byte(ec_enc *_this,unsigned _value){
  if(_this->offs+_this->end_offs>=_this->storage)return -1;
  _this->buf[_this->offs++]=(unsigned char)_value;
  return 0;
}

static int ec_write_byte_at_end(ec_enc *_this,unsigned _value){
  if(_this->offs+_this->end_offs>=_this->storage)return -1;
  _this->buf[_this->storage-++(_this->end_offs)]=(unsigned char)_value;
  return 0;
}

/*Outputs a symbol, with a carry bit.
  If there is a potential to propagate a carry over several symbols, they are
   buffered until it can be determined whether or not an actual carry will
   occur.
  If the counter for the buffered symbols overflows, then the stream becomes
   undecodable.
  This gives a theoretical limit of a few billion symbols in a single packet on
   32-bit systems.
  The alternative is to truncate the range in order to force a carry, but
   requires similar carry tracking in the decoder, needlessly slowing it down.*/
static void ec_enc_carry_out(ec_enc *_this,int _c){
  if(_c!=EC_SYM_MAX){
    /*No further carry propagation possible, flush buffer.*/
    int carry;
    carry=_c>>EC_SYM_BITS;
    /*Don't output a byte on the first write.
      This compare should be taken care of by branch-prediction thereafter.*/
    if(_this->rem>=0)_this->error|=ec_write_byte(_this,_this->rem+carry);
    if(_this->ext>0){
      unsigned sym;
      sym=(EC_SYM_MAX+carry)&EC_SYM_MAX;
      do _this->error|=ec_write_byte(_this,sym);
      while(--(_this->ext)>0);
    }
    _this->rem=_c&EC_SYM_MAX;
  }
  else _this->ext++;
}

static OPUS_INLINE void ec_enc_normalize(ec_enc *_this){
  /*If the range is too small, output some bits and rescale it.*/
  while(_this->rng<=EC_CODE_BOT){
    ec_enc_carry_out(_this,(int)(_this->val>>EC_CODE_SHIFT));
    /*Move the next-to-high-order symbol into the high-order position.*/
    _this->val=(_this->val<<EC_SYM_BITS)&(EC_CODE_TOP-1);
    _this->rng<<=EC_SYM_BITS;
    _this->nbits_total+=EC_SYM_BITS;
  }
}

void ec_enc_init(ec_enc *_this,unsigned char *_buf,opus_uint32 _size){
  _this->buf=_buf;
  _this->end_offs=0;
  _this->end_window=0;
  _this->nend_bits=0;
  /*This is the offset from which ec_tell() will subtract partial bits.*/
  _this->nbits_total=EC_CODE_BITS+1;
  _this->offs=0;
  _this->rng=EC_CODE_TOP;
  _this->rem=-1;
  _this->val=0;
  _this->ext=0;
  _this->storage=_size;
  _this->error=0;
}

void ec_encode(ec_enc *_this,unsigned _fl,unsigned _fh,unsigned _ft){
  opus_uint32 r;
  r=celt_udiv(_this->rng,_ft);
  if(_fl>0){
    _this->val+=_this->rng-IMUL32(r,(_ft-_fl));
    _this->rng=IMUL32(r,(_fh-_fl));
  }
  else _this->rng-=IMUL32(r,(_ft-_fh));
  ec_enc_normalize(_this);
}

void ec_encode_bin(ec_enc *_this,unsigned _fl,unsigned _fh,unsigned _bits){
  opus_uint32 r;
  r=_this->rng>>_bits;
  if(_fl>0){
    _this->val+=_this->rng-IMUL32(r,((1U<<_bits)-_fl));
    _this->rng=IMUL32(r,(_fh-_fl));
  }
  else _this->rng-=IMUL32(r,((1U<<_bits)-_fh));
  ec_enc_normalize(_this);
}

/*The probability of having a "one" is 1/(1<<_logp).*/
void ec_enc_bit_logp(ec_enc *_this,int _val,unsigned _logp){
  opus_uint32 r;
  opus_uint32 s;
  opus_uint32 l;
  r=_this->rng;
  l=_this->val;
  s=r>>_logp;
  r-=s;
  if(_val)_this->val=l+r;
  _this->rng=_val?s:r;
  ec_enc_normalize(_this);
}

void ec_enc_icdf(ec_enc *_this,int _s,const unsigned char *_icdf,unsigned _ftb){
  opus_uint32 r;
  r=_this->rng>>_ftb;
  if(_s>0){
    _this->val+=_this->rng-IMUL32(r,_icdf[_s-1]);
    _this->rng=IMUL32(r,_icdf[_s-1]-_icdf[_s]);
  }
  else _this->rng-=IMUL32(r,_icdf[_s]);
  ec_enc_normalize(_this);
}

void ec_enc_icdf16(ec_enc *_this,int _s,const opus_uint16 *_icdf,unsigned _ftb){
  opus_uint32 r;
  r=_this->rng>>_ftb;
  if(_s>0){
    _this->val+=_this->rng-IMUL32(r,_icdf[_s-1]);
    _this->rng=IMUL32(r,_icdf[_s-1]-_icdf[_s]);
  }
  else _this->rng-=IMUL32(r,_icdf[_s]);
  ec_enc_normalize(_this);
}

void ec_enc_uint(ec_enc *_this,opus_uint32 _fl,opus_uint32 _ft){
  unsigned  ft;
  unsigned  fl;
  int       ftb;
  /*In order to optimize EC_ILOG(), it is undefined for the value 0.*/
  celt_assert(_ft>1);
  _ft--;
  ftb=EC_ILOG(_ft);
  if(ftb>EC_UINT_BITS){
    ftb-=EC_UINT_BITS;
    ft=(_ft>>ftb)+1;
    fl=(unsigned)(_fl>>ftb);
    ec_encode(_this,fl,fl+1,ft);
    ec_enc_bits(_this,_fl&(((opus_uint32)1<<ftb)-1U),ftb);
  }
  else ec_encode(_this,_fl,_fl+1,_ft+1);
}

void ec_enc_bits(ec_enc *_this,opus_uint32 _fl,unsigned _bits){
  ec_window window;
  int       used;
  window=_this->end_window;
  used=_this->nend_bits;
  celt_assert(_bits>0);
  if(used+_bits>EC_WINDOW_SIZE){
    do{
      _this->error|=ec_write_byte_at_end(_this,(unsigned)window&EC_SYM_MAX);
      window>>=EC_SYM_BITS;
      used-=EC_SYM_BITS;
    }
    while(used>=EC_SYM_BITS);
  }
  window|=(ec_window)_fl<<used;
  used+=_bits;
  _this->end_window=window;
  _this->nend_bits=used;
  _this->nbits_total+=_bits;
}

void ec_enc_patch_initial_bits(ec_enc *_this,unsigned _val,unsigned _nbits){
  int      shift;
  unsigned mask;
  celt_assert(_nbits<=EC_SYM_BITS);
  shift=EC_SYM_BITS-_nbits;
  mask=((1<<_nbits)-1)<<shift;
  if(_this->offs>0){
    /*The first byte has been finalized.*/
    _this->buf[0]=(unsigned char)((_this->buf[0]&~mask)|_val<<shift);
  }
  else if(_this->rem>=0){
    /*The first byte is still awaiting carry propagation.*/
    _this->rem=(_this->rem&~mask)|_val<<shift;
  }
  else if(_this->rng<=(EC_CODE_TOP>>_nbits)){
    /*The renormalization loop has never been run.*/
    _this->val=(_this->val&~((opus_uint32)mask<<EC_CODE_SHIFT))|
     (opus_uint32)_val<<(EC_CODE_SHIFT+shift);
  }
  /*The encoder hasn't even encoded _nbits of data yet.*/
  else _this->error=-1;
}

void ec_enc_shrink(ec_enc *_this,opus_uint32 _size){
  celt_assert(_this->offs+_this->end_offs<=_size);
  OPUS_MOVE(_this->buf+_size-_this->end_offs,
   _this->buf+_this->storage-_this->end_offs,_this->end_offs);
  _this->storage=_size;
}

void ec_enc_done(ec_enc *_this){
  ec_window   window;
  int         used;
  opus_uint32 msk;
  opus_uint32 end;
  int         l;
  /*We output the minimum number of bits that ensures that the symbols encoded
     thus far will be decoded correctly regardless of the bits that follow.*/
  l=EC_CODE_BITS-EC_ILOG(_this->rng);
  msk=(EC_CODE_TOP-1)>>l;
  end=(_this->val+msk)&~msk;
  if((end|msk)>=_this->val+_this->rng){
    l++;
    msk>>=1;
    end=(_this->val+msk)&~msk;
  }
  while(l>0){
    ec_enc_carry_out(_this,(int)(end>>EC_CODE_SHIFT));
    end=(end<<EC_SYM_BITS)&(EC_CODE_TOP-1);
    l-=EC_SYM_BITS;
  }
  /*If we have a buffered byte flush it into the output buffer.*/
  if(_this->rem>=0||_this->ext>0)ec_enc_carry_out(_this,0);
  /*If we have buffered extra bits, flush them as well.*/
  window=_this->end_window;
  used=_this->nend_bits;
  while(used>=EC_SYM_BITS){
    _this->error|=ec_write_byte_at_end(_this,(unsigned)window&EC_SYM_MAX);
    window>>=EC_SYM_BITS;
    used-=EC_SYM_BITS;
  }
  /*Clear any excess space and add any remaining extra bits to the last byte.*/
  if(!_this->error){
    if (_this->buf) OPUS_CLEAR(_this->buf+_this->offs,
     _this->storage-_this->offs-_this->end_offs);
    if(used>0){
      /*If there's no range coder data at all, give up.*/
      if(_this->end_offs>=_this->storage)_this->error=-1;
      else{
        l=-l;
        /*If we've busted, don't add too many extra bits to the last byte; it
           would corrupt the range coder data, and that's more important.*/
        if(_this->offs+_this->end_offs>=_this->storage&&l<used){
          window&=(1<<l)-1;
          _this->error=-1;
        }
        _this->buf[_this->storage-_this->end_offs-1]|=(unsigned char)window;
      }
    }
  }
}
