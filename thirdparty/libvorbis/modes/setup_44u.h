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

 function: toplevel settings for 44.1/48kHz uncoupled modes

 ********************************************************************/

#include "modes/residue_44u.h"

static const double rate_mapping_44_un[12]={
  32000.,48000.,60000.,70000.,80000.,86000.,
  96000.,110000.,120000.,140000.,160000.,240001.
};

static const ve_setup_data_template ve_setup_44_uncoupled={
  11,
  rate_mapping_44_un,
  quality_mapping_44,
  -1,
  40000,
  50000,

  blocksize_short_44,
  blocksize_long_44,

  _psy_tone_masteratt_44,
  _psy_tone_0dB,
  _psy_tone_suppress,

  _vp_tonemask_adj_otherblock,
  _vp_tonemask_adj_longblock,
  _vp_tonemask_adj_otherblock,

  _psy_noiseguards_44,
  _psy_noisebias_impulse,
  _psy_noisebias_padding,
  _psy_noisebias_trans,
  _psy_noisebias_long,
  _psy_noise_suppress,

  _psy_compand_44,
  _psy_compand_short_mapping,
  _psy_compand_long_mapping,

  {_noise_start_short_44,_noise_start_long_44},
  {_noise_part_short_44,_noise_part_long_44},
  _noise_thresh_44,

  _psy_ath_floater,
  _psy_ath_abs,

  _psy_lowpass_44,

  _psy_global_44,
  _global_mapping_44,
  _psy_stereo_modes_44,

  _floor_books,
  _floor,
  2,
  _floor_mapping_44,

  _mapres_template_44_uncoupled
};
