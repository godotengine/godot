/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2007             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: bitrate tracking and management

 ********************************************************************/

#ifndef _V_BITRATE_H_
#define _V_BITRATE_H_

#include "vorbis/codec.h"
#include "codec_internal.h"
#include "os.h"

/* encode side bitrate tracking */
typedef struct bitrate_manager_state {
  int            managed;

  long           avg_reservoir;
  long           minmax_reservoir;
  long           avg_bitsper;
  long           min_bitsper;
  long           max_bitsper;

  long           short_per_long;
  double         avgfloat;

  vorbis_block  *vb;
  int            choice;
} bitrate_manager_state;

typedef struct bitrate_manager_info{
  long           avg_rate;
  long           min_rate;
  long           max_rate;
  long           reservoir_bits;
  double         reservoir_bias;

  double         slew_damp;

} bitrate_manager_info;

extern void vorbis_bitrate_init(vorbis_info *vi,bitrate_manager_state *bs);
extern void vorbis_bitrate_clear(bitrate_manager_state *bs);
extern int vorbis_bitrate_managed(vorbis_block *vb);
extern int vorbis_bitrate_addblock(vorbis_block *vb);
extern int vorbis_bitrate_flushpacket(vorbis_dsp_state *vd, ogg_packet *op);

#endif
