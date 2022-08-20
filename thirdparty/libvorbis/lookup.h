/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2009             *
 * by the Xiph.Org Foundation https://xiph.org/                     *
 *                                                                  *
 ********************************************************************

  function: lookup based functions

 ********************************************************************/

#ifndef _V_LOOKUP_H_

#ifdef FLOAT_LOOKUP
extern float vorbis_coslook(float a);
extern float vorbis_invsqlook(float a);
extern float vorbis_invsq2explook(int a);
extern float vorbis_fromdBlook(float a);
#endif
#ifdef INT_LOOKUP
extern long vorbis_invsqlook_i(long a,long e);
extern long vorbis_coslook_i(long a);
extern float vorbis_fromdBlook_i(long a);
#endif

#endif
