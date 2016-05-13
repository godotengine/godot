/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2009             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function: lookup based functions
  last mod: $Id: lookup.c 16227 2009-07-08 06:58:46Z xiphmont $

 ********************************************************************/

#include <math.h>
#include "lookup.h"
#include "lookup_data.h"
#include "os.h"
#include "misc.h"

#ifdef FLOAT_LOOKUP

/* interpolated lookup based cos function, domain 0 to PI only */
float vorbis_coslook(float a){
  double d=a*(.31830989*(float)COS_LOOKUP_SZ);
  int i=vorbis_ftoi(d-.5);

  return COS_LOOKUP[i]+ (d-i)*(COS_LOOKUP[i+1]-COS_LOOKUP[i]);
}

/* interpolated 1./sqrt(p) where .5 <= p < 1. */
float vorbis_invsqlook(float a){
  double d=a*(2.f*(float)INVSQ_LOOKUP_SZ)-(float)INVSQ_LOOKUP_SZ;
  int i=vorbis_ftoi(d-.5f);
  return INVSQ_LOOKUP[i]+ (d-i)*(INVSQ_LOOKUP[i+1]-INVSQ_LOOKUP[i]);
}

/* interpolated 1./sqrt(p) where .5 <= p < 1. */
float vorbis_invsq2explook(int a){
  return INVSQ2EXP_LOOKUP[a-INVSQ2EXP_LOOKUP_MIN];
}

#include <stdio.h>
/* interpolated lookup based fromdB function, domain -140dB to 0dB only */
float vorbis_fromdBlook(float a){
  int i=vorbis_ftoi(a*((float)(-(1<<FROMdB2_SHIFT)))-.5f);
  return (i<0)?1.f:
    ((i>=(FROMdB_LOOKUP_SZ<<FROMdB_SHIFT))?0.f:
     FROMdB_LOOKUP[i>>FROMdB_SHIFT]*FROMdB2_LOOKUP[i&FROMdB2_MASK]);
}

#endif

#ifdef INT_LOOKUP
/* interpolated 1./sqrt(p) where .5 <= a < 1. (.100000... to .111111...) in
   16.16 format

   returns in m.8 format */
long vorbis_invsqlook_i(long a,long e){
  long i=(a&0x7fff)>>(INVSQ_LOOKUP_I_SHIFT-1);
  long d=(a&INVSQ_LOOKUP_I_MASK)<<(16-INVSQ_LOOKUP_I_SHIFT); /*  0.16 */
  long val=INVSQ_LOOKUP_I[i]-                                /*  1.16 */
    (((INVSQ_LOOKUP_I[i]-INVSQ_LOOKUP_I[i+1])*               /*  0.16 */
      d)>>16);                                               /* result 1.16 */

  e+=32;
  if(e&1)val=(val*5792)>>13; /* multiply val by 1/sqrt(2) */
  e=(e>>1)-8;

  return(val>>e);
}

/* interpolated lookup based fromdB function, domain -140dB to 0dB only */
/* a is in n.12 format */
float vorbis_fromdBlook_i(long a){
  int i=(-a)>>(12-FROMdB2_SHIFT);
  return (i<0)?1.f:
    ((i>=(FROMdB_LOOKUP_SZ<<FROMdB_SHIFT))?0.f:
     FROMdB_LOOKUP[i>>FROMdB_SHIFT]*FROMdB2_LOOKUP[i&FROMdB2_MASK]);
}

/* interpolated lookup based cos function, domain 0 to PI only */
/* a is in 0.16 format, where 0==0, 2^^16-1==PI, return 0.14 */
long vorbis_coslook_i(long a){
  int i=a>>COS_LOOKUP_I_SHIFT;
  int d=a&COS_LOOKUP_I_MASK;
  return COS_LOOKUP_I[i]- ((d*(COS_LOOKUP_I[i]-COS_LOOKUP_I[i+1]))>>
                           COS_LOOKUP_I_SHIFT);
}

#endif
