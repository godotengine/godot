#include "mathops.h"
#include <limits.h>

/*The fastest fallback strategy for platforms with fast multiplication appears
   to be based on de Bruijn sequences~\cite{LP98}.
  Tests confirmed this to be true even on an ARM11, where it is actually faster
   than using the native clz instruction.
  Define OC_ILOG_NODEBRUIJN to use a simpler fallback on platforms where
   multiplication or table lookups are too expensive.

  @UNPUBLISHED{LP98,
    author="Charles E. Leiserson and Harald Prokop",
    title="Using de {Bruijn} Sequences to Index a 1 in a Computer Word",
    month=Jun,
    year=1998,
    note="\url{http://supertech.csail.mit.edu/papers/debruijn.pdf}"
  }*/
#if !defined(OC_ILOG_NODEBRUIJN)&& \
 !defined(OC_CLZ32)||!defined(OC_CLZ64)&&LONG_MAX<9223372036854775807LL
static const unsigned char OC_DEBRUIJN_IDX32[32]={
   0, 1,28, 2,29,14,24, 3,30,22,20,15,25,17, 4, 8,
  31,27,13,23,21,19,16, 7,26,12,18, 6,11, 5,10, 9
};
#endif

int oc_ilog32(ogg_uint32_t _v){
#if defined(OC_CLZ32)
  return (OC_CLZ32_OFFS-OC_CLZ32(_v))&-!!_v;
#else
/*On a Pentium M, this branchless version tested as the fastest version without
   multiplications on 1,000,000,000 random 32-bit integers, edging out a
   similar version with branches, and a 256-entry LUT version.*/
# if defined(OC_ILOG_NODEBRUIJN)
  int ret;
  int m;
  ret=_v>0;
  m=(_v>0xFFFFU)<<4;
  _v>>=m;
  ret|=m;
  m=(_v>0xFFU)<<3;
  _v>>=m;
  ret|=m;
  m=(_v>0xFU)<<2;
  _v>>=m;
  ret|=m;
  m=(_v>3)<<1;
  _v>>=m;
  ret|=m;
  ret+=_v>1;
  return ret;
/*This de Bruijn sequence version is faster if you have a fast multiplier.*/
# else
  int ret;
  ret=_v>0;
  _v|=_v>>1;
  _v|=_v>>2;
  _v|=_v>>4;
  _v|=_v>>8;
  _v|=_v>>16;
  _v=(_v>>1)+1;
  ret+=OC_DEBRUIJN_IDX32[_v*0x77CB531U>>27&0x1F];
  return ret;
# endif
#endif
}

int oc_ilog64(ogg_int64_t _v){
#if defined(OC_CLZ64)
  return (OC_CLZ64_OFFS-OC_CLZ64(_v))&-!!_v;
#else
# if defined(OC_ILOG_NODEBRUIJN)
  ogg_uint32_t v;
  int          ret;
  int          m;
  ret=_v>0;
  m=(_v>0xFFFFFFFFU)<<5;
  v=(ogg_uint32_t)(_v>>m);
  ret|=m;
  m=(v>0xFFFFU)<<4;
  v>>=m;
  ret|=m;
  m=(v>0xFFU)<<3;
  v>>=m;
  ret|=m;
  m=(v>0xFU)<<2;
  v>>=m;
  ret|=m;
  m=(v>3)<<1;
  v>>=m;
  ret|=m;
  ret+=v>1;
  return ret;
# else
/*If we don't have a 64-bit word, split it into two 32-bit halves.*/
#  if LONG_MAX<9223372036854775807LL
  ogg_uint32_t v;
  int          ret;
  int          m;
  ret=_v>0;
  m=(_v>0xFFFFFFFFU)<<5;
  v=(ogg_uint32_t)(_v>>m);
  ret|=m;
  v|=v>>1;
  v|=v>>2;
  v|=v>>4;
  v|=v>>8;
  v|=v>>16;
  v=(v>>1)+1;
  ret+=OC_DEBRUIJN_IDX32[v*0x77CB531U>>27&0x1F];
  return ret;
/*Otherwise do it in one 64-bit operation.*/
#  else
  static const unsigned char OC_DEBRUIJN_IDX64[64]={
     0, 1, 2, 7, 3,13, 8,19, 4,25,14,28, 9,34,20,40,
     5,17,26,38,15,46,29,48,10,31,35,54,21,50,41,57,
    63, 6,12,18,24,27,33,39,16,37,45,47,30,53,49,56,
    62,11,23,32,36,44,52,55,61,22,43,51,60,42,59,58
  };
  int ret;
  ret=_v>0;
  _v|=_v>>1;
  _v|=_v>>2;
  _v|=_v>>4;
  _v|=_v>>8;
  _v|=_v>>16;
  _v|=_v>>32;
  _v=(_v>>1)+1;
  ret+=OC_DEBRUIJN_IDX64[_v*0x218A392CD3D5DBF>>58&0x3F];
  return ret;
#  endif
# endif
#endif
}

/*round(2**(62+i)*atanh(2**(-(i+1)))/log(2))*/
static const ogg_int64_t OC_ATANH_LOG2[32]={
  0x32B803473F7AD0F4LL,0x2F2A71BD4E25E916LL,0x2E68B244BB93BA06LL,
  0x2E39FB9198CE62E4LL,0x2E2E683F68565C8FLL,0x2E2B850BE2077FC1LL,
  0x2E2ACC58FE7B78DBLL,0x2E2A9E2DE52FD5F2LL,0x2E2A92A338D53EECLL,
  0x2E2A8FC08F5E19B6LL,0x2E2A8F07E51A485ELL,0x2E2A8ED9BA8AF388LL,
  0x2E2A8ECE2FE7384ALL,0x2E2A8ECB4D3E4B1ALL,0x2E2A8ECA94940FE8LL,
  0x2E2A8ECA6669811DLL,0x2E2A8ECA5ADEDD6ALL,0x2E2A8ECA57FC347ELL,
  0x2E2A8ECA57438A43LL,0x2E2A8ECA57155FB4LL,0x2E2A8ECA5709D510LL,
  0x2E2A8ECA5706F267LL,0x2E2A8ECA570639BDLL,0x2E2A8ECA57060B92LL,
  0x2E2A8ECA57060008LL,0x2E2A8ECA5705FD25LL,0x2E2A8ECA5705FC6CLL,
  0x2E2A8ECA5705FC3ELL,0x2E2A8ECA5705FC33LL,0x2E2A8ECA5705FC30LL,
  0x2E2A8ECA5705FC2FLL,0x2E2A8ECA5705FC2FLL
};

/*Computes the binary exponential of _z, a log base 2 in Q57 format.*/
ogg_int64_t oc_bexp64(ogg_int64_t _z){
  ogg_int64_t w;
  ogg_int64_t z;
  int         ipart;
  ipart=(int)(_z>>57);
  if(ipart<0)return 0;
  if(ipart>=63)return 0x7FFFFFFFFFFFFFFFLL;
  z=_z-OC_Q57(ipart);
  if(z){
    ogg_int64_t mask;
    long        wlo;
    int         i;
    /*C doesn't give us 64x64->128 muls, so we use CORDIC.
      This is not particularly fast, but it's not being used in time-critical
       code; it is very accurate.*/
    /*z is the fractional part of the log in Q62 format.
      We need 1 bit of headroom since the magnitude can get larger than 1
       during the iteration, and a sign bit.*/
    z<<=5;
    /*w is the exponential in Q61 format (since it also needs headroom and can
       get as large as 2.0); we could get another bit if we dropped the sign,
       but we'll recover that bit later anyway.
      Ideally this should start out as
        \lim_{n->\infty} 2^{61}/\product_{i=1}^n \sqrt{1-2^{-2i}}
       but in order to guarantee convergence we have to repeat iterations 4,
        13 (=3*4+1), and 40 (=3*13+1, etc.), so it winds up somewhat larger.*/
    w=0x26A3D0E401DD846DLL;
    for(i=0;;i++){
      mask=-(z<0);
      w+=(w>>i+1)+mask^mask;
      z-=OC_ATANH_LOG2[i]+mask^mask;
      /*Repeat iteration 4.*/
      if(i>=3)break;
      z<<=1;
    }
    for(;;i++){
      mask=-(z<0);
      w+=(w>>i+1)+mask^mask;
      z-=OC_ATANH_LOG2[i]+mask^mask;
      /*Repeat iteration 13.*/
      if(i>=12)break;
      z<<=1;
    }
    for(;i<32;i++){
      mask=-(z<0);
      w+=(w>>i+1)+mask^mask;
      z=z-(OC_ATANH_LOG2[i]+mask^mask)<<1;
    }
    wlo=0;
    /*Skip the remaining iterations unless we really require that much
       precision.
      We could have bailed out earlier for smaller iparts, but that would
       require initializing w from a table, as the limit doesn't converge to
       61-bit precision until n=30.*/
    if(ipart>30){
      /*For these iterations, we just update the low bits, as the high bits
         can't possibly be affected.
        OC_ATANH_LOG2 has also converged (it actually did so one iteration
         earlier, but that's no reason for an extra special case).*/
      for(;;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z-=OC_ATANH_LOG2[31]+mask^mask;
        /*Repeat iteration 40.*/
        if(i>=39)break;
        z<<=1;
      }
      for(;i<61;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z=z-(OC_ATANH_LOG2[31]+mask^mask)<<1;
      }
    }
    w=(w<<1)+wlo;
  }
  else w=(ogg_int64_t)1<<62;
  if(ipart<62)w=(w>>61-ipart)+1>>1;
  return w;
}

/*Computes the binary logarithm of _w, returned in Q57 format.*/
ogg_int64_t oc_blog64(ogg_int64_t _w){
  ogg_int64_t z;
  int         ipart;
  if(_w<=0)return -1;
  ipart=OC_ILOGNZ_64(_w)-1;
  if(ipart>61)_w>>=ipart-61;
  else _w<<=61-ipart;
  z=0;
  if(_w&_w-1){
    ogg_int64_t x;
    ogg_int64_t y;
    ogg_int64_t u;
    ogg_int64_t mask;
    int         i;
    /*C doesn't give us 64x64->128 muls, so we use CORDIC.
      This is not particularly fast, but it's not being used in time-critical
       code; it is very accurate.*/
    /*z is the fractional part of the log in Q61 format.*/
    /*x and y are the cosh() and sinh(), respectively, in Q61 format.
      We are computing z=2*atanh(y/x)=2*atanh((_w-1)/(_w+1)).*/
    x=_w+((ogg_int64_t)1<<61);
    y=_w-((ogg_int64_t)1<<61);
    for(i=0;i<4;i++){
      mask=-(y<0);
      z+=(OC_ATANH_LOG2[i]>>i)+mask^mask;
      u=x>>i+1;
      x-=(y>>i+1)+mask^mask;
      y-=u+mask^mask;
    }
    /*Repeat iteration 4.*/
    for(i--;i<13;i++){
      mask=-(y<0);
      z+=(OC_ATANH_LOG2[i]>>i)+mask^mask;
      u=x>>i+1;
      x-=(y>>i+1)+mask^mask;
      y-=u+mask^mask;
    }
    /*Repeat iteration 13.*/
    for(i--;i<32;i++){
      mask=-(y<0);
      z+=(OC_ATANH_LOG2[i]>>i)+mask^mask;
      u=x>>i+1;
      x-=(y>>i+1)+mask^mask;
      y-=u+mask^mask;
    }
    /*OC_ATANH_LOG2 has converged.*/
    for(;i<40;i++){
      mask=-(y<0);
      z+=(OC_ATANH_LOG2[31]>>i)+mask^mask;
      u=x>>i+1;
      x-=(y>>i+1)+mask^mask;
      y-=u+mask^mask;
    }
    /*Repeat iteration 40.*/
    for(i--;i<62;i++){
      mask=-(y<0);
      z+=(OC_ATANH_LOG2[31]>>i)+mask^mask;
      u=x>>i+1;
      x-=(y>>i+1)+mask^mask;
      y-=u+mask^mask;
    }
    z=z+8>>4;
  }
  return OC_Q57(ipart)+z;
}
