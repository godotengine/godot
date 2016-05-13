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

  function: LSP (also called LSF) conversion routines
  last mod: $Id: lsp.h 16227 2009-07-08 06:58:46Z xiphmont $

 ********************************************************************/


#ifndef _V_LSP_H_
#define _V_LSP_H_

extern int vorbis_lpc_to_lsp(float *lpc,float *lsp,int m);

extern void vorbis_lsp_to_curve(float *curve,int *map,int n,int ln,
                                float *lsp,int m,
                                float amp,float ampoffset);

#endif
