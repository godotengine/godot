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
  last mod: $Id$

 ********************************************************************/
#include "encint.h"
#include "dct.h"



/*Performs a forward 8 point Type-II DCT transform.
  The output is scaled by a factor of 2 from the orthonormal version of the
   transform.
  _y: The buffer to store the result in.
      Data will be placed the first 8 entries (e.g., in a row of an 8x8 block).
  _x: The input coefficients.
      Every 8th entry is used (e.g., from a column of an 8x8 block).*/
static void oc_fdct8(ogg_int16_t _y[8],const ogg_int16_t *_x){
  int t0;
  int t1;
  int t2;
  int t3;
  int t4;
  int t5;
  int t6;
  int t7;
  int r;
  int s;
  int u;
  int v;
  /*Stage 1:*/
  /*0-7 butterfly.*/
  t0=_x[0<<3]+(int)_x[7<<3];
  t7=_x[0<<3]-(int)_x[7<<3];
  /*1-6 butterfly.*/
  t1=_x[1<<3]+(int)_x[6<<3];
  t6=_x[1<<3]-(int)_x[6<<3];
  /*2-5 butterfly.*/
  t2=_x[2<<3]+(int)_x[5<<3];
  t5=_x[2<<3]-(int)_x[5<<3];
  /*3-4 butterfly.*/
  t3=_x[3<<3]+(int)_x[4<<3];
  t4=_x[3<<3]-(int)_x[4<<3];
  /*Stage 2:*/
  /*0-3 butterfly.*/
  r=t0+t3;
  t3=t0-t3;
  t0=r;
  /*1-2 butterfly.*/
  r=t1+t2;
  t2=t1-t2;
  t1=r;
  /*6-5 butterfly.*/
  r=t6+t5;
  t5=t6-t5;
  t6=r;
  /*Stages 3 and 4 are where all the approximation occurs.
    These are chosen to be as close to an exact inverse of the approximations
     made in the iDCT as possible, while still using mostly 16-bit arithmetic.
    We use some 16x16->32 signed MACs, but those still commonly execute in 1
     cycle on a 16-bit DSP.
    For example, s=(27146*t5+0x4000>>16)+t5+(t5!=0) is an exact inverse of
     t5=(OC_C4S4*s>>16).
    That is, applying the latter to the output of the former will recover t5
     exactly (over the valid input range of t5, -23171...23169).
    We increase the rounding bias to 0xB500 in this particular case so that
     errors inverting the subsequent butterfly are not one-sided (e.g., the
     mean error is very close to zero).
    The (t5!=0) term could be replaced simply by 1, but we want to send 0 to 0.
    The fDCT of an all-zeros block will still not be zero, because of the
     biases we added at the very beginning of the process, but it will be close
     enough that it is guaranteed to round to zero.*/
  /*Stage 3:*/
  /*4-5 butterfly.*/
  s=(27146*t5+0xB500>>16)+t5+(t5!=0)>>1;
  r=t4+s;
  t5=t4-s;
  t4=r;
  /*7-6 butterfly.*/
  s=(27146*t6+0xB500>>16)+t6+(t6!=0)>>1;
  r=t7+s;
  t6=t7-s;
  t7=r;
  /*Stage 4:*/
  /*0-1 butterfly.*/
  r=(27146*t0+0x4000>>16)+t0+(t0!=0);
  s=(27146*t1+0xB500>>16)+t1+(t1!=0);
  u=r+s>>1;
  v=r-u;
  _y[0]=u;
  _y[4]=v;
  /*3-2 rotation by 6pi/16*/
  u=(OC_C6S2*t2+OC_C2S6*t3+0x6CB7>>16)+(t3!=0);
  s=(OC_C6S2*u>>16)-t2;
  v=(s*21600+0x2800>>18)+s+(s!=0);
  _y[2]=u;
  _y[6]=v;
  /*6-5 rotation by 3pi/16*/
  u=(OC_C5S3*t6+OC_C3S5*t5+0x0E3D>>16)+(t5!=0);
  s=t6-(OC_C5S3*u>>16);
  v=(s*26568+0x3400>>17)+s+(s!=0);
  _y[5]=u;
  _y[3]=v;
  /*7-4 rotation by 7pi/16*/
  u=(OC_C7S1*t4+OC_C1S7*t7+0x7B1B>>16)+(t7!=0);
  s=(OC_C7S1*u>>16)-t4;
  v=(s*20539+0x3000>>20)+s+(s!=0);
  _y[1]=u;
  _y[7]=v;
}

/*Performs a forward 8x8 Type-II DCT transform.
  The output is scaled by a factor of 4 relative to the orthonormal version
   of the transform.
  _y: The buffer to store the result in.
      This may be the same as _x.
  _x: The input coefficients. */
void oc_enc_fdct8x8_c(ogg_int16_t _y[64],const ogg_int16_t _x[64]){
  const ogg_int16_t *in;
  ogg_int16_t       *end;
  ogg_int16_t       *out;
  ogg_int16_t        w[64];
  int                i;
  /*Add two extra bits of working precision to improve accuracy; any more and
     we could overflow.*/
  for(i=0;i<64;i++)w[i]=_x[i]<<2;
  /*These biases correct for some systematic error that remains in the full
     fDCT->iDCT round trip.*/
  w[0]+=(w[0]!=0)+1;
  w[1]++;
  w[8]--;
  /*Transform columns of w into rows of _y.*/
  for(in=w,out=_y,end=out+64;out<end;in++,out+=8)oc_fdct8(out,in);
  /*Transform columns of _y into rows of w.*/
  for(in=_y,out=w,end=out+64;out<end;in++,out+=8)oc_fdct8(out,in);
  /*Round the result back to the external working precision (which is still
     scaled by four relative to the orthogonal result).
    TODO: We should just update the external working precision.*/
  for(i=0;i<64;i++)_y[i]=w[OC_FZIG_ZAG[i]]+2>>2;
}



/*This does not seem to outperform simple LFE border padding before MC.
  It yields higher PSNR, but much higher bitrate usage.*/
#if 0
typedef struct oc_extension_info oc_extension_info;



/*Information needed to pad boundary blocks.
  We multiply each row/column by an extension matrix that fills in the padding
   values as a linear combination of the active values, so that an equivalent
   number of coefficients are forced to zero.
  This costs at most 16 multiplies, the same as a 1-D fDCT itself, and as
   little as 7 multiplies.
  We compute the extension matrices for every possible shape in advance, as
   there are only 35.
  The coefficients for all matrices are stored in a single array to take
   advantage of the overlap and repetitiveness of many of the shapes.
  A similar technique is applied to the offsets into this array.
  This reduces the required table storage by about 48%.
  See tools/extgen.c for details.
  We could conceivably do the same for all 256 possible shapes.*/
struct oc_extension_info{
  /*The mask of the active pixels in the shape.*/
  short                     mask;
  /*The number of active pixels in the shape.*/
  short                     na;
  /*The extension matrix.
    This is (8-na)xna*/
  const ogg_int16_t *const *ext;
  /*The pixel indices: na active pixels followed by 8-na padding pixels.*/
  unsigned char             pi[8];
  /*The coefficient indices: na unconstrained coefficients followed by 8-na
     coefficients to be forced to zero.*/
  unsigned char             ci[8];
};


/*The number of shapes we need.*/
#define OC_NSHAPES   (35)

static const ogg_int16_t OC_EXT_COEFFS[229]={
  0x7FFF,0xE1F8,0x6903,0xAA79,0x5587,0x7FFF,0x1E08,0x7FFF,
  0x5587,0xAA79,0x6903,0xE1F8,0x7FFF,0x0000,0x0000,0x0000,
  0x7FFF,0x0000,0x0000,0x7FFF,0x8000,0x7FFF,0x0000,0x0000,
  0x7FFF,0xE1F8,0x1E08,0xB0A7,0xAA1D,0x337C,0x7FFF,0x4345,
  0x2267,0x4345,0x7FFF,0x337C,0xAA1D,0xB0A7,0x8A8C,0x4F59,
  0x03B4,0xE2D6,0x7FFF,0x2CF3,0x7FFF,0xE2D6,0x03B4,0x4F59,
  0x8A8C,0x1103,0x7AEF,0x5225,0xDF60,0xC288,0xDF60,0x5225,
  0x7AEF,0x1103,0x668A,0xD6EE,0x3A16,0x0E6C,0xFA07,0x0E6C,
  0x3A16,0xD6EE,0x668A,0x2A79,0x2402,0x980F,0x50F5,0x4882,
  0x50F5,0x980F,0x2402,0x2A79,0xF976,0x2768,0x5F22,0x2768,
  0xF976,0x1F91,0x76C1,0xE9AE,0x76C1,0x1F91,0x7FFF,0xD185,
  0x0FC8,0xD185,0x7FFF,0x4F59,0x4345,0xED62,0x4345,0x4F59,
  0xF574,0x5D99,0x2CF3,0x5D99,0xF574,0x5587,0x3505,0x30FC,
  0xF482,0x953C,0xEAC4,0x7FFF,0x4F04,0x7FFF,0xEAC4,0x953C,
  0xF482,0x30FC,0x4F04,0x273D,0xD8C3,0x273D,0x1E09,0x61F7,
  0x1E09,0x273D,0xD8C3,0x273D,0x4F04,0x30FC,0xA57E,0x153C,
  0x6AC4,0x3C7A,0x1E08,0x3C7A,0x6AC4,0x153C,0xA57E,0x7FFF,
  0xA57E,0x5A82,0x6AC4,0x153C,0xC386,0xE1F8,0xC386,0x153C,
  0x6AC4,0x5A82,0xD8C3,0x273D,0x7FFF,0xE1F7,0x7FFF,0x273D,
  0xD8C3,0x4F04,0x30FC,0xD8C3,0x273D,0xD8C3,0x30FC,0x4F04,
  0x1FC8,0x67AD,0x1853,0xE038,0x1853,0x67AD,0x1FC8,0x4546,
  0xE038,0x1FC8,0x3ABA,0x1FC8,0xE038,0x4546,0x3505,0x5587,
  0xF574,0xBC11,0x78F4,0x4AFB,0xE6F3,0x4E12,0x3C11,0xF8F4,
  0x4AFB,0x3C7A,0xF88B,0x3C11,0x78F4,0xCAFB,0x7FFF,0x08CC,
  0x070C,0x236D,0x5587,0x236D,0x070C,0xF88B,0x3C7A,0x4AFB,
  0xF8F4,0x3C11,0x7FFF,0x153C,0xCAFB,0x153C,0x7FFF,0x1E08,
  0xE1F8,0x7FFF,0x08CC,0x7FFF,0xCAFB,0x78F4,0x3C11,0x4E12,
  0xE6F3,0x4AFB,0x78F4,0xBC11,0xFE3D,0x7FFF,0xFE3D,0x2F3A,
  0x7FFF,0x2F3A,0x89BC,0x7FFF,0x89BC
};

static const ogg_int16_t *const OC_EXT_ROWS[96]={
  OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   0,
  OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   0,OC_EXT_COEFFS+   6,
  OC_EXT_COEFFS+  27,OC_EXT_COEFFS+  38,OC_EXT_COEFFS+  43,OC_EXT_COEFFS+  32,
  OC_EXT_COEFFS+  49,OC_EXT_COEFFS+  58,OC_EXT_COEFFS+  67,OC_EXT_COEFFS+  71,
  OC_EXT_COEFFS+  62,OC_EXT_COEFFS+  53,OC_EXT_COEFFS+  12,OC_EXT_COEFFS+  15,
  OC_EXT_COEFFS+  14,OC_EXT_COEFFS+  13,OC_EXT_COEFFS+  76,OC_EXT_COEFFS+  81,
  OC_EXT_COEFFS+  86,OC_EXT_COEFFS+  91,OC_EXT_COEFFS+  96,OC_EXT_COEFFS+  98,
  OC_EXT_COEFFS+  93,OC_EXT_COEFFS+  88,OC_EXT_COEFFS+  83,OC_EXT_COEFFS+  78,
  OC_EXT_COEFFS+  12,OC_EXT_COEFFS+  15,OC_EXT_COEFFS+  15,OC_EXT_COEFFS+  12,
  OC_EXT_COEFFS+  12,OC_EXT_COEFFS+  15,OC_EXT_COEFFS+  12,OC_EXT_COEFFS+  15,
  OC_EXT_COEFFS+  15,OC_EXT_COEFFS+  12,OC_EXT_COEFFS+ 103,OC_EXT_COEFFS+ 108,
  OC_EXT_COEFFS+ 126,OC_EXT_COEFFS+  16,OC_EXT_COEFFS+ 137,OC_EXT_COEFFS+ 141,
  OC_EXT_COEFFS+  20,OC_EXT_COEFFS+ 130,OC_EXT_COEFFS+ 113,OC_EXT_COEFFS+ 116,
  OC_EXT_COEFFS+ 146,OC_EXT_COEFFS+ 153,OC_EXT_COEFFS+ 160,OC_EXT_COEFFS+ 167,
  OC_EXT_COEFFS+ 170,OC_EXT_COEFFS+ 163,OC_EXT_COEFFS+ 156,OC_EXT_COEFFS+ 149,
  OC_EXT_COEFFS+ 119,OC_EXT_COEFFS+ 122,OC_EXT_COEFFS+ 174,OC_EXT_COEFFS+ 177,
  OC_EXT_COEFFS+ 182,OC_EXT_COEFFS+ 187,OC_EXT_COEFFS+ 192,OC_EXT_COEFFS+ 197,
  OC_EXT_COEFFS+ 202,OC_EXT_COEFFS+ 207,OC_EXT_COEFFS+ 210,OC_EXT_COEFFS+ 215,
  OC_EXT_COEFFS+ 179,OC_EXT_COEFFS+ 189,OC_EXT_COEFFS+  24,OC_EXT_COEFFS+ 204,
  OC_EXT_COEFFS+ 184,OC_EXT_COEFFS+ 194,OC_EXT_COEFFS+ 212,OC_EXT_COEFFS+ 199,
  OC_EXT_COEFFS+ 217,OC_EXT_COEFFS+ 100,OC_EXT_COEFFS+ 134,OC_EXT_COEFFS+ 135,
  OC_EXT_COEFFS+ 135,OC_EXT_COEFFS+  12,OC_EXT_COEFFS+  15,OC_EXT_COEFFS+ 134,
  OC_EXT_COEFFS+ 134,OC_EXT_COEFFS+ 135,OC_EXT_COEFFS+ 220,OC_EXT_COEFFS+ 223,
  OC_EXT_COEFFS+ 226,OC_EXT_COEFFS+ 227,OC_EXT_COEFFS+ 224,OC_EXT_COEFFS+ 221
};

static const oc_extension_info OC_EXTENSION_INFO[OC_NSHAPES]={
  {0x7F,7,OC_EXT_ROWS+  0,{0,1,2,3,4,5,6,7},{0,1,2,4,5,6,7,3}},
  {0xFE,7,OC_EXT_ROWS+  7,{1,2,3,4,5,6,7,0},{0,1,2,4,5,6,7,3}},
  {0x3F,6,OC_EXT_ROWS+  8,{0,1,2,3,4,5,7,6},{0,1,3,4,6,7,5,2}},
  {0xFC,6,OC_EXT_ROWS+ 10,{2,3,4,5,6,7,1,0},{0,1,3,4,6,7,5,2}},
  {0x1F,5,OC_EXT_ROWS+ 12,{0,1,2,3,4,7,6,5},{0,2,3,5,7,6,4,1}},
  {0xF8,5,OC_EXT_ROWS+ 15,{3,4,5,6,7,2,1,0},{0,2,3,5,7,6,4,1}},
  {0x0F,4,OC_EXT_ROWS+ 18,{0,1,2,3,7,6,5,4},{0,2,4,6,7,5,3,1}},
  {0xF0,4,OC_EXT_ROWS+ 18,{4,5,6,7,3,2,1,0},{0,2,4,6,7,5,3,1}},
  {0x07,3,OC_EXT_ROWS+ 22,{0,1,2,7,6,5,4,3},{0,3,6,7,5,4,2,1}},
  {0xE0,3,OC_EXT_ROWS+ 27,{5,6,7,4,3,2,1,0},{0,3,6,7,5,4,2,1}},
  {0x03,2,OC_EXT_ROWS+ 32,{0,1,7,6,5,4,3,2},{0,4,7,6,5,3,2,1}},
  {0xC0,2,OC_EXT_ROWS+ 32,{6,7,5,4,3,2,1,0},{0,4,7,6,5,3,2,1}},
  {0x01,1,OC_EXT_ROWS+  0,{0,7,6,5,4,3,2,1},{0,7,6,5,4,3,2,1}},
  {0x80,1,OC_EXT_ROWS+  0,{7,6,5,4,3,2,1,0},{0,7,6,5,4,3,2,1}},
  {0x7E,6,OC_EXT_ROWS+ 42,{1,2,3,4,5,6,7,0},{0,1,2,5,6,7,4,3}},
  {0x7C,5,OC_EXT_ROWS+ 44,{2,3,4,5,6,7,1,0},{0,1,4,5,7,6,3,2}},
  {0x3E,5,OC_EXT_ROWS+ 47,{1,2,3,4,5,7,6,0},{0,1,4,5,7,6,3,2}},
  {0x78,4,OC_EXT_ROWS+ 50,{3,4,5,6,7,2,1,0},{0,4,5,7,6,3,2,1}},
  {0x3C,4,OC_EXT_ROWS+ 54,{2,3,4,5,7,6,1,0},{0,3,4,7,6,5,2,1}},
  {0x1E,4,OC_EXT_ROWS+ 58,{1,2,3,4,7,6,5,0},{0,4,5,7,6,3,2,1}},
  {0x70,3,OC_EXT_ROWS+ 62,{4,5,6,7,3,2,1,0},{0,5,7,6,4,3,2,1}},
  {0x38,3,OC_EXT_ROWS+ 67,{3,4,5,7,6,2,1,0},{0,5,6,7,4,3,2,1}},
  {0x1C,3,OC_EXT_ROWS+ 72,{2,3,4,7,6,5,1,0},{0,5,6,7,4,3,2,1}},
  {0x0E,3,OC_EXT_ROWS+ 77,{1,2,3,7,6,5,4,0},{0,5,7,6,4,3,2,1}},
  {0x60,2,OC_EXT_ROWS+ 82,{5,6,7,4,3,2,1,0},{0,2,7,6,5,4,3,1}},
  {0x30,2,OC_EXT_ROWS+ 36,{4,5,7,6,3,2,1,0},{0,4,7,6,5,3,2,1}},
  {0x18,2,OC_EXT_ROWS+ 90,{3,4,7,6,5,2,1,0},{0,1,7,6,5,4,3,2}},
  {0x0C,2,OC_EXT_ROWS+ 34,{2,3,7,6,5,4,1,0},{0,4,7,6,5,3,2,1}},
  {0x06,2,OC_EXT_ROWS+ 84,{1,2,7,6,5,4,3,0},{0,2,7,6,5,4,3,1}},
  {0x40,1,OC_EXT_ROWS+  0,{6,7,5,4,3,2,1,0},{0,7,6,5,4,3,2,1}},
  {0x20,1,OC_EXT_ROWS+  0,{5,7,6,4,3,2,1,0},{0,7,6,5,4,3,2,1}},
  {0x10,1,OC_EXT_ROWS+  0,{4,7,6,5,3,2,1,0},{0,7,6,5,4,3,2,1}},
  {0x08,1,OC_EXT_ROWS+  0,{3,7,6,5,4,2,1,0},{0,7,6,5,4,3,2,1}},
  {0x04,1,OC_EXT_ROWS+  0,{2,7,6,5,4,3,1,0},{0,7,6,5,4,3,2,1}},
  {0x02,1,OC_EXT_ROWS+  0,{1,7,6,5,4,3,2,0},{0,7,6,5,4,3,2,1}}
};



/*Pads a single column of a partial block and then performs a forward Type-II
   DCT on the result.
  The input is scaled by a factor of 4 and biased appropriately for the current
   fDCT implementation.
  The output is scaled by an additional factor of 2 from the orthonormal
   version of the transform.
  _y: The buffer to store the result in.
      Data will be placed the first 8 entries (e.g., in a row of an 8x8 block).
  _x: The input coefficients.
      Every 8th entry is used (e.g., from a column of an 8x8 block).
  _e: The extension information for the shape.*/
static void oc_fdct8_ext(ogg_int16_t _y[8],ogg_int16_t *_x,
 const oc_extension_info *_e){
  const unsigned char *pi;
  int                  na;
  na=_e->na;
  pi=_e->pi;
  if(na==1){
    int ci;
    /*While the branch below is still correct for shapes with na==1, we can
       perform the entire transform with just 1 multiply in this case instead
       of 23.*/
    _y[0]=(ogg_int16_t)(OC_DIV2_16(OC_C4S4*(_x[pi[0]])));
    for(ci=1;ci<8;ci++)_y[ci]=0;
  }
  else{
    const ogg_int16_t *const *ext;
    int                       zpi;
    int                       api;
    int                       nz;
    /*First multiply by the extension matrix to compute the padding values.*/
    nz=8-na;
    ext=_e->ext;
    for(zpi=0;zpi<nz;zpi++){
      ogg_int32_t v;
      v=0;
      for(api=0;api<na;api++){
        v+=ext[zpi][api]*(ogg_int32_t)(_x[pi[api]<<3]<<1);
      }
      _x[pi[na+zpi]<<3]=(ogg_int16_t)(v+0x8000>>16)+1>>1;
    }
    oc_fdct8(_y,_x);
  }
}

/*Performs a forward 8x8 Type-II DCT transform on blocks which overlap the
   border of the picture region.
  This method ONLY works with rectangular regions.
  _border: A description of which pixels are inside the border.
  _y:      The buffer to store the result in.
           This may be the same as _x.
  _x:      The input pixel values.
           Pixel values outside the border will be ignored.*/
void oc_fdct8x8_border(const oc_border_info *_border,
 ogg_int16_t _y[64],const ogg_int16_t _x[64]){
  ogg_int16_t             *in;
  ogg_int16_t             *out;
  ogg_int16_t              w[64];
  ogg_int64_t              mask;
  const oc_extension_info *cext;
  const oc_extension_info *rext;
  int                      cmask;
  int                      rmask;
  int                      ri;
  int                      ci;
  /*Identify the shapes of the non-zero rows and columns.*/
  rmask=cmask=0;
  mask=_border->mask;
  for(ri=0;ri<8;ri++){
    /*This aggregation is _only_ correct for rectangular masks.*/
    cmask|=((mask&0xFF)!=0)<<ri;
    rmask|=mask&0xFF;
    mask>>=8;
  }
  /*Find the associated extension info for these shapes.*/
  if(cmask==0xFF)cext=NULL;
  else for(cext=OC_EXTENSION_INFO;cext->mask!=cmask;){
    /*If we somehow can't find the shape, then just do an unpadded fDCT.
      It won't be efficient, but it should still be correct.*/
    if(++cext>=OC_EXTENSION_INFO+OC_NSHAPES){
      oc_enc_fdct8x8_c(_y,_x);
      return;
    }
  }
  if(rmask==0xFF)rext=NULL;
  else for(rext=OC_EXTENSION_INFO;rext->mask!=rmask;){
    /*If we somehow can't find the shape, then just do an unpadded fDCT.
      It won't be efficient, but it should still be correct.*/
    if(++rext>=OC_EXTENSION_INFO+OC_NSHAPES){
      oc_enc_fdct8x8_c(_y,_x);
      return;
    }
  }
  /*Add two extra bits of working precision to improve accuracy; any more and
     we could overflow.*/
  for(ci=0;ci<64;ci++)w[ci]=_x[ci]<<2;
  /*These biases correct for some systematic error that remains in the full
     fDCT->iDCT round trip.
    We can safely add them before padding, since if these pixel values are
     overwritten, we didn't care what they were anyway (and the unbiased values
     will usually yield smaller DCT coefficient magnitudes).*/
  w[0]+=(w[0]!=0)+1;
  w[1]++;
  w[8]--;
  /*Transform the columns.
    We can ignore zero columns without a problem.*/
  in=w;
  out=_y;
  if(cext==NULL)for(ci=0;ci<8;ci++)oc_fdct8(out+(ci<<3),in+ci);
  else for(ci=0;ci<8;ci++)if(rmask&(1<<ci))oc_fdct8_ext(out+(ci<<3),in+ci,cext);
  /*Transform the rows.
    We transform even rows that are supposedly zero, because rounding errors
     may make them slightly non-zero, and this will give a more precise
     reconstruction with very small quantizers.*/
  in=_y;
  out=w;
  if(rext==NULL)for(ri=0;ri<8;ri++)oc_fdct8(out+(ri<<3),in+ri);
  else for(ri=0;ri<8;ri++)oc_fdct8_ext(out+(ri<<3),in+ri,rext);
  /*Round the result back to the external working precision (which is still
     scaled by four relative to the orthogonal result).
    TODO: We should just update the external working precision.*/
  for(ci=0;ci<64;ci++)_y[ci]=w[ci]+2>>2;
}
#endif
