/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2001             *
 * by the XIPHOPHORUS Company http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: fft transform
 last mod: $Id: smallft.h,v 1.3 2003/09/16 18:35:45 jm Exp $

 ********************************************************************/
/**
   @file smallft.h
   @brief Discrete Rotational Fourier Transform (DRFT)
*/

#ifndef _V_SMFT_H_
#define _V_SMFT_H_


#ifdef __cplusplus
extern "C" {
#endif

/** Discrete Rotational Fourier Transform lookup */
struct drft_lookup{
  int n;
  float *trigcache;
  int *splitcache;
};

extern void spx_drft_forward(struct drft_lookup *l,float *data);
extern void spx_drft_backward(struct drft_lookup *l,float *data);
extern void spx_drft_init(struct drft_lookup *l,int n);
extern void spx_drft_clear(struct drft_lookup *l);

#ifdef __cplusplus
}
#endif

#endif
