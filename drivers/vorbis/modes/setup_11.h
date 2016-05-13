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

 function: 11kHz settings
 last mod: $Id: setup_11.h 16894 2010-02-12 20:32:12Z xiphmont $

 ********************************************************************/

#include "psych_11.h"

static const int blocksize_11[2]={
  512,512
};

static const int _floor_mapping_11a[]={
  6,6
};
static const int *_floor_mapping_11[]={
  _floor_mapping_11a
};

static const double rate_mapping_11[3]={
  8000.,13000.,44000.,
};

static const double rate_mapping_11_uncoupled[3]={
  12000.,20000.,50000.,
};

static const double quality_mapping_11[3]={
  -.1,.0,1.
};

static const ve_setup_data_template ve_setup_11_stereo={
  2,
  rate_mapping_11,
  quality_mapping_11,
  2,
  9000,
  15000,

  blocksize_11,
  blocksize_11,

  _psy_tone_masteratt_11,
  _psy_tone_0dB,
  _psy_tone_suppress,

  _vp_tonemask_adj_11,
  NULL,
  _vp_tonemask_adj_11,

  _psy_noiseguards_8,
  _psy_noisebias_11,
  _psy_noisebias_11,
  NULL,
  NULL,
  _psy_noise_suppress,

  _psy_compand_8,
  _psy_compand_8_mapping,
  NULL,

  {_noise_start_8,_noise_start_8},
  {_noise_part_8,_noise_part_8},
  _noise_thresh_11,

  _psy_ath_floater_8,
  _psy_ath_abs_8,

  _psy_lowpass_11,

  _psy_global_44,
  _global_mapping_8,
  _psy_stereo_modes_8,

  _floor_books,
  _floor,
  1,
  _floor_mapping_11,

  _mapres_template_8_stereo
};

static const ve_setup_data_template ve_setup_11_uncoupled={
  2,
  rate_mapping_11_uncoupled,
  quality_mapping_11,
  -1,
  9000,
  15000,

  blocksize_11,
  blocksize_11,

  _psy_tone_masteratt_11,
  _psy_tone_0dB,
  _psy_tone_suppress,

  _vp_tonemask_adj_11,
  NULL,
  _vp_tonemask_adj_11,

  _psy_noiseguards_8,
  _psy_noisebias_11,
  _psy_noisebias_11,
  NULL,
  NULL,
  _psy_noise_suppress,

  _psy_compand_8,
  _psy_compand_8_mapping,
  NULL,

  {_noise_start_8,_noise_start_8},
  {_noise_part_8,_noise_part_8},
  _noise_thresh_11,

  _psy_ath_floater_8,
  _psy_ath_abs_8,

  _psy_lowpass_11,

  _psy_global_44,
  _global_mapping_8,
  _psy_stereo_modes_8,

  _floor_books,
  _floor,
  1,
  _floor_mapping_11,

  _mapres_template_8_uncoupled
};
