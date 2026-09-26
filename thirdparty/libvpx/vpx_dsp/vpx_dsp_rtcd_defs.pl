##
##  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

sub vpx_dsp_forward_decls() {
print <<EOF
/*
 * DSP
 */

#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#if CONFIG_VP9_ENCODER
 struct macroblock_plane;
 struct ScanOrder;
#endif

EOF
}
forward_decls qw/vpx_dsp_forward_decls/;

# functions that are 64 bit only.
$mmx_x86_64 = $sse2_x86_64 = $ssse3_x86_64 = $avx_x86_64 = $avx2_x86_64 = '';
if ($opts{arch} eq "x86_64") {
  $mmx_x86_64 = 'mmx';
  $sse2_x86_64 = 'sse2';
  $ssse3_x86_64 = 'ssse3';
  $avx_x86_64 = 'avx';
  $avx2_x86_64 = 'avx2';
  $avx512_x86_64 = 'avx512';
}

#
# Intra prediction
#

add_proto qw/void vpx_d207_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d207_predictor_4x4 neon sse2/;

add_proto qw/void vpx_d45_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d45_predictor_4x4 neon sse2/;

add_proto qw/void vpx_d45e_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";

add_proto qw/void vpx_d63_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d63_predictor_4x4 neon ssse3/;

add_proto qw/void vpx_d63e_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";

add_proto qw/void vpx_h_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_h_predictor_4x4 neon dspr2 msa sse2/;

add_proto qw/void vpx_he_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";

add_proto qw/void vpx_d117_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d117_predictor_4x4 neon/;

add_proto qw/void vpx_d135_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d135_predictor_4x4 neon/;

add_proto qw/void vpx_d153_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d153_predictor_4x4 neon ssse3/;

add_proto qw/void vpx_v_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_v_predictor_4x4 neon msa sse2/;

add_proto qw/void vpx_ve_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";

add_proto qw/void vpx_tm_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_tm_predictor_4x4 neon dspr2 msa sse2/;

add_proto qw/void vpx_dc_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_predictor_4x4 dspr2 msa neon sse2/;

add_proto qw/void vpx_dc_top_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_top_predictor_4x4 msa neon sse2/;

add_proto qw/void vpx_dc_left_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_left_predictor_4x4 msa neon sse2/;

add_proto qw/void vpx_dc_128_predictor_4x4/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_128_predictor_4x4 msa neon sse2/;

add_proto qw/void vpx_d207_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d207_predictor_8x8 neon ssse3/;

add_proto qw/void vpx_d45_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_d45_predictor_8x8 neon sse2/;

add_proto qw/void vpx_d63_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_d63_predictor_8x8 neon ssse3/;

add_proto qw/void vpx_h_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_h_predictor_8x8 neon dspr2 msa sse2/;

add_proto qw/void vpx_d117_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d117_predictor_8x8 neon/;

add_proto qw/void vpx_d135_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d135_predictor_8x8 neon/;

add_proto qw/void vpx_d153_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d153_predictor_8x8 neon ssse3/;

add_proto qw/void vpx_v_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_v_predictor_8x8 neon msa sse2/;

add_proto qw/void vpx_tm_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_tm_predictor_8x8 neon dspr2 msa sse2/;

add_proto qw/void vpx_dc_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
# TODO(crbug.com/webm/1522): Re-enable vsx implementation.
specialize qw/vpx_dc_predictor_8x8 dspr2 neon msa sse2 lsx/;

add_proto qw/void vpx_dc_top_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_top_predictor_8x8 neon msa sse2/;

add_proto qw/void vpx_dc_left_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_left_predictor_8x8 neon msa sse2/;

add_proto qw/void vpx_dc_128_predictor_8x8/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_128_predictor_8x8 neon msa sse2/;

add_proto qw/void vpx_d207_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d207_predictor_16x16 neon ssse3/;

add_proto qw/void vpx_d45_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d45_predictor_16x16 neon ssse3 vsx/;

add_proto qw/void vpx_d63_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d63_predictor_16x16 neon ssse3 vsx/;

add_proto qw/void vpx_h_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_h_predictor_16x16 neon dspr2 msa sse2 vsx/;

add_proto qw/void vpx_d117_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d117_predictor_16x16 neon/;

add_proto qw/void vpx_d135_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d135_predictor_16x16 neon/;

add_proto qw/void vpx_d153_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d153_predictor_16x16 neon ssse3/;

add_proto qw/void vpx_v_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_v_predictor_16x16 neon msa sse2 vsx/;

add_proto qw/void vpx_tm_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_tm_predictor_16x16 neon msa sse2 vsx/;

add_proto qw/void vpx_dc_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_predictor_16x16 dspr2 neon msa sse2 vsx lsx/;

add_proto qw/void vpx_dc_top_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_top_predictor_16x16 neon msa sse2 vsx/;

add_proto qw/void vpx_dc_left_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_left_predictor_16x16 neon msa sse2 vsx/;

add_proto qw/void vpx_dc_128_predictor_16x16/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_128_predictor_16x16 neon msa sse2 vsx/;

add_proto qw/void vpx_d207_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d207_predictor_32x32 neon ssse3/;

add_proto qw/void vpx_d45_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d45_predictor_32x32 neon ssse3 vsx/;

add_proto qw/void vpx_d63_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d63_predictor_32x32 neon ssse3 vsx/;

add_proto qw/void vpx_h_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_h_predictor_32x32 neon msa sse2 vsx/;

add_proto qw/void vpx_d117_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d117_predictor_32x32 neon/;

add_proto qw/void vpx_d135_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d135_predictor_32x32 neon/;

add_proto qw/void vpx_d153_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_d153_predictor_32x32 neon ssse3/;

add_proto qw/void vpx_v_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_v_predictor_32x32 neon msa sse2 vsx/;

add_proto qw/void vpx_tm_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_tm_predictor_32x32 neon msa sse2 vsx/;

add_proto qw/void vpx_dc_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_predictor_32x32 msa neon sse2 vsx/;

add_proto qw/void vpx_dc_top_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_top_predictor_32x32 msa neon sse2 vsx/;

add_proto qw/void vpx_dc_left_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_left_predictor_32x32 msa neon sse2 vsx/;

add_proto qw/void vpx_dc_128_predictor_32x32/, "uint8_t *dst, ptrdiff_t stride, const uint8_t *above, const uint8_t *left";
specialize qw/vpx_dc_128_predictor_32x32 msa neon sse2 vsx/;

# High bitdepth functions
if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  add_proto qw/void vpx_highbd_d207_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d207_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_d45_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d45_predictor_4x4 neon ssse3/;

  add_proto qw/void vpx_highbd_d63_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d63_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_h_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_h_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_d117_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d117_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_d135_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d135_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_d153_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d153_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_v_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_v_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_tm_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_tm_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_dc_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_dc_top_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_top_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_dc_left_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_left_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_dc_128_predictor_4x4/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_128_predictor_4x4 neon sse2/;

  add_proto qw/void vpx_highbd_d207_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d207_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_d45_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d45_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_d63_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d63_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_h_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_h_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_d117_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d117_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_d135_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d135_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_d153_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d153_predictor_8x8 neon ssse3/;

  add_proto qw/void vpx_highbd_v_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_v_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_tm_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_tm_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_dc_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_dc_top_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_top_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_dc_left_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_left_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_dc_128_predictor_8x8/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_128_predictor_8x8 neon sse2/;

  add_proto qw/void vpx_highbd_d207_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d207_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_d45_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d45_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_d63_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d63_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_h_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_h_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_d117_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d117_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_d135_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d135_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_d153_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d153_predictor_16x16 neon ssse3/;

  add_proto qw/void vpx_highbd_v_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_v_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_tm_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_tm_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_dc_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_dc_top_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_top_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_dc_left_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_left_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_dc_128_predictor_16x16/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_128_predictor_16x16 neon sse2/;

  add_proto qw/void vpx_highbd_d207_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d207_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_d45_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d45_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_d63_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d63_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_h_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_h_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_d117_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d117_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_d135_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d135_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_d153_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_d153_predictor_32x32 neon ssse3/;

  add_proto qw/void vpx_highbd_v_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_v_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_tm_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_tm_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_dc_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_dc_top_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_top_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_dc_left_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_left_predictor_32x32 neon sse2/;

  add_proto qw/void vpx_highbd_dc_128_predictor_32x32/, "uint16_t *dst, ptrdiff_t stride, const uint16_t *above, const uint16_t *left, int bd";
  specialize qw/vpx_highbd_dc_128_predictor_32x32 neon sse2/;
}  # CONFIG_VP9_HIGHBITDEPTH

if (vpx_config("CONFIG_VP9") eq "yes") {
#
# Sub Pixel Filters
#
add_proto qw/void vpx_convolve_copy/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve_copy neon dspr2 msa sse2 vsx lsx/;

add_proto qw/void vpx_convolve_avg/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve_avg neon dspr2 msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_convolve8/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8 sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_convolve8_horiz/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8_horiz sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_convolve8_vert/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8_vert sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_convolve8_avg/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8_avg sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_convolve8_avg_horiz/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8_avg_horiz sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_convolve8_avg_vert/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_convolve8_avg_vert sse2 ssse3 avx2 neon neon_dotprod neon_i8mm dspr2 msa vsx mmi lsx/;

add_proto qw/void vpx_scaled_2d/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
specialize qw/vpx_scaled_2d ssse3 neon msa/;

add_proto qw/void vpx_scaled_horiz/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";

add_proto qw/void vpx_scaled_vert/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";

add_proto qw/void vpx_scaled_avg_2d/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";

add_proto qw/void vpx_scaled_avg_horiz/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";

add_proto qw/void vpx_scaled_avg_vert/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
} #CONFIG_VP9

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  #
  # Sub Pixel Filters
  #
  add_proto qw/void vpx_highbd_convolve_copy/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve_copy sse2 avx2 neon/;

  add_proto qw/void vpx_highbd_convolve_avg/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve_avg sse2 avx2 neon/;

  add_proto qw/void vpx_highbd_convolve8/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8 avx2 neon sve2/, "$sse2_x86_64";

  add_proto qw/void vpx_highbd_convolve8_horiz/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8_horiz avx2 neon sve/, "$sse2_x86_64";

  add_proto qw/void vpx_highbd_convolve8_vert/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8_vert avx2 neon sve2/, "$sse2_x86_64";

  add_proto qw/void vpx_highbd_convolve8_avg/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8_avg avx2 neon sve2/, "$sse2_x86_64";

  add_proto qw/void vpx_highbd_convolve8_avg_horiz/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8_avg_horiz avx2 neon sve/, "$sse2_x86_64";

  add_proto qw/void vpx_highbd_convolve8_avg_vert/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
  specialize qw/vpx_highbd_convolve8_avg_vert avx2 neon sve2/, "$sse2_x86_64";
}  # CONFIG_VP9_HIGHBITDEPTH

if (vpx_config("CONFIG_VP9") eq "yes") {
#
# Loopfilter
#
add_proto qw/void vpx_lpf_vertical_16/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_vertical_16 sse2 neon dspr2 msa/;

add_proto qw/void vpx_lpf_vertical_16_dual/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_vertical_16_dual sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_vertical_8/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_vertical_8 sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_vertical_8_dual/, "uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1";
specialize qw/vpx_lpf_vertical_8_dual sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_vertical_4/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_vertical_4 sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_vertical_4_dual/, "uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1";
specialize qw/vpx_lpf_vertical_4_dual sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_horizontal_16/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_horizontal_16 sse2 avx2 neon dspr2 msa/;

add_proto qw/void vpx_lpf_horizontal_16_dual/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_horizontal_16_dual sse2 avx2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_horizontal_8/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_horizontal_8 sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_horizontal_8_dual/, "uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1";
specialize qw/vpx_lpf_horizontal_8_dual sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_horizontal_4/, "uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh";
specialize qw/vpx_lpf_horizontal_4 sse2 neon dspr2 msa lsx/;

add_proto qw/void vpx_lpf_horizontal_4_dual/, "uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1";
specialize qw/vpx_lpf_horizontal_4_dual sse2 neon dspr2 msa lsx/;
} #CONFIG_VP9

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  add_proto qw/void vpx_highbd_lpf_vertical_16/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_vertical_16 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_vertical_16_dual/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_vertical_16_dual sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_vertical_8/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_vertical_8 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_vertical_8_dual/, "uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1, int bd";
  specialize qw/vpx_highbd_lpf_vertical_8_dual sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_vertical_4/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_vertical_4 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_vertical_4_dual/, "uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1, int bd";
  specialize qw/vpx_highbd_lpf_vertical_4_dual sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_16/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_16 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_16_dual/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_16_dual sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_8/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_8 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_8_dual/, "uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_8_dual sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_4/, "uint16_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_4 sse2 neon/;

  add_proto qw/void vpx_highbd_lpf_horizontal_4_dual/, "uint16_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1, int bd";
  specialize qw/vpx_highbd_lpf_horizontal_4_dual sse2 neon/;
}  # CONFIG_VP9_HIGHBITDEPTH

#
# Encoder functions.
#

#
# Forward transform
#
if (vpx_config("CONFIG_VP9_ENCODER") eq "yes") {
if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  add_proto qw/void vpx_fdct4x4/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct4x4 neon sse2/;

  add_proto qw/void vpx_fdct4x4_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct4x4_1 sse2 neon/;
  specialize qw/vpx_highbd_fdct4x4_1 neon/;
  $vpx_highbd_fdct4x4_1_neon=vpx_fdct4x4_1_neon;

  add_proto qw/void vpx_fdct8x8/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct8x8 neon sse2/;

  add_proto qw/void vpx_fdct8x8_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct8x8_1 neon sse2 msa/;

  add_proto qw/void vpx_fdct16x16/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct16x16 neon sse2/;

  add_proto qw/void vpx_fdct16x16_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct16x16_1 sse2 neon/;

  add_proto qw/void vpx_fdct32x32/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32 neon sse2/;

  add_proto qw/void vpx_fdct32x32_rd/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32_rd neon sse2/;

  add_proto qw/void vpx_fdct32x32_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32_1 sse2 neon/;

  add_proto qw/void vpx_highbd_fdct4x4/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct4x4 sse2 neon/;

  add_proto qw/void vpx_highbd_fdct8x8/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct8x8 sse2 neon/;

  add_proto qw/void vpx_highbd_fdct8x8_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct8x8_1 neon/;
  $vpx_highbd_fdct8x8_1_neon=vpx_fdct8x8_1_neon;

  add_proto qw/void vpx_highbd_fdct16x16/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct16x16 sse2 neon/;

  add_proto qw/void vpx_highbd_fdct16x16_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct16x16_1 neon/;

  add_proto qw/void vpx_highbd_fdct32x32/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct32x32 sse2 neon/;

  add_proto qw/void vpx_highbd_fdct32x32_rd/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct32x32_rd sse2 neon/;

  add_proto qw/void vpx_highbd_fdct32x32_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_highbd_fdct32x32_1 neon/;
} else {
  add_proto qw/void vpx_fdct4x4/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct4x4 neon sse2 msa lsx/;

  add_proto qw/void vpx_fdct4x4_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct4x4_1 sse2 neon/;

  add_proto qw/void vpx_fdct8x8/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct8x8 sse2 neon msa lsx/, "$ssse3_x86_64";

  add_proto qw/void vpx_fdct8x8_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct8x8_1 sse2 neon msa/;

  add_proto qw/void vpx_fdct16x16/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct16x16 neon sse2 avx2 msa lsx/;

  add_proto qw/void vpx_fdct16x16_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct16x16_1 sse2 neon msa/;

  add_proto qw/void vpx_fdct32x32/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32 neon sse2 avx2 msa lsx/;

  add_proto qw/void vpx_fdct32x32_rd/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32_rd sse2 avx2 neon msa vsx lsx/;

  add_proto qw/void vpx_fdct32x32_1/, "const int16_t *input, tran_low_t *output, int stride";
  specialize qw/vpx_fdct32x32_1 sse2 neon msa/;
}  # CONFIG_VP9_HIGHBITDEPTH
}  # CONFIG_VP9_ENCODER

#
# Inverse transform
if (vpx_config("CONFIG_VP9") eq "yes") {

add_proto qw/void vpx_idct4x4_16_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct4x4_1_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct8x8_64_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct8x8_12_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct8x8_1_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct16x16_256_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct16x16_38_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct16x16_10_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct16x16_1_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct32x32_1024_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct32x32_135_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct32x32_34_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_idct32x32_1_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_iwht4x4_16_add/, "const tran_low_t *input, uint8_t *dest, int stride";
add_proto qw/void vpx_iwht4x4_1_add/, "const tran_low_t *input, uint8_t *dest, int stride";

if (vpx_config("CONFIG_EMULATE_HARDWARE") ne "yes") {
  # Note that there are more specializations appended when
  # CONFIG_VP9_HIGHBITDEPTH is off.
  specialize qw/vpx_idct4x4_16_add neon sse2 vsx/;
  specialize qw/vpx_idct4x4_1_add neon sse2/;
  specialize qw/vpx_idct8x8_64_add neon sse2 vsx/;
  specialize qw/vpx_idct8x8_12_add neon sse2 ssse3/;
  specialize qw/vpx_idct8x8_1_add neon sse2/;
  specialize qw/vpx_idct16x16_256_add neon sse2 avx2 vsx/;
  specialize qw/vpx_idct16x16_38_add neon sse2/;
  specialize qw/vpx_idct16x16_10_add neon sse2/;
  specialize qw/vpx_idct16x16_1_add neon sse2/;
  specialize qw/vpx_idct32x32_1024_add neon sse2 avx2 vsx/;
  specialize qw/vpx_idct32x32_135_add neon sse2 ssse3 avx2/;
  specialize qw/vpx_idct32x32_34_add neon sse2 ssse3/;
  specialize qw/vpx_idct32x32_1_add neon sse2/;
  specialize qw/vpx_iwht4x4_16_add sse2 vsx/;

  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") ne "yes") {
    # Note that these specializations are appended to the above ones.
    specialize qw/vpx_idct4x4_16_add dspr2 msa/;
    specialize qw/vpx_idct4x4_1_add dspr2 msa/;
    specialize qw/vpx_idct8x8_64_add dspr2 msa/;
    specialize qw/vpx_idct8x8_12_add dspr2 msa/;
    specialize qw/vpx_idct8x8_1_add dspr2 msa/;
    specialize qw/vpx_idct16x16_256_add dspr2 msa/;
    specialize qw/vpx_idct16x16_38_add dspr2 msa/;
    $vpx_idct16x16_38_add_dspr2=vpx_idct16x16_256_add_dspr2;
    $vpx_idct16x16_38_add_msa=vpx_idct16x16_256_add_msa;
    specialize qw/vpx_idct16x16_10_add dspr2 msa/;
    specialize qw/vpx_idct16x16_1_add dspr2 msa/;
    specialize qw/vpx_idct32x32_1024_add dspr2 msa lsx/;
    specialize qw/vpx_idct32x32_135_add dspr2 msa/;
    $vpx_idct32x32_135_add_dspr2=vpx_idct32x32_1024_add_dspr2;
    $vpx_idct32x32_135_add_msa=vpx_idct32x32_1024_add_msa;
    $vpx_idct32x32_135_add_lsx=vpx_idct32x32_1024_add_lsx;
    specialize qw/vpx_idct32x32_34_add dspr2 msa lsx/;
    specialize qw/vpx_idct32x32_1_add dspr2 msa lsx/;
    specialize qw/vpx_iwht4x4_16_add msa/;
    specialize qw/vpx_iwht4x4_1_add msa/;
  } # !CONFIG_VP9_HIGHBITDEPTH
}  # !CONFIG_EMULATE_HARDWARE

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  # Note as optimized versions of these functions are added we need to add a check to ensure
  # that when CONFIG_EMULATE_HARDWARE is on, it defaults to the C versions only.

  add_proto qw/void vpx_highbd_idct4x4_16_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct4x4_1_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  specialize qw/vpx_highbd_idct4x4_1_add neon sse2/;

  add_proto qw/void vpx_highbd_idct8x8_64_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct8x8_12_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct8x8_1_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  specialize qw/vpx_highbd_idct8x8_1_add neon sse2/;

  add_proto qw/void vpx_highbd_idct16x16_256_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct16x16_38_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct16x16_10_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct16x16_1_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  specialize qw/vpx_highbd_idct16x16_1_add neon sse2/;

  add_proto qw/void vpx_highbd_idct32x32_1024_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct32x32_135_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct32x32_34_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_idct32x32_1_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  specialize qw/vpx_highbd_idct32x32_1_add neon sse2/;

  add_proto qw/void vpx_highbd_iwht4x4_16_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";
  add_proto qw/void vpx_highbd_iwht4x4_1_add/, "const tran_low_t *input, uint16_t *dest, int stride, int bd";

  if (vpx_config("CONFIG_EMULATE_HARDWARE") ne "yes") {
    specialize qw/vpx_highbd_idct4x4_16_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct8x8_64_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct8x8_12_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct16x16_256_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct16x16_38_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct16x16_10_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct32x32_1024_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct32x32_135_add neon sse2 sse4_1/;
    specialize qw/vpx_highbd_idct32x32_34_add neon sse2 sse4_1/;
  }  # !CONFIG_EMULATE_HARDWARE
}  # CONFIG_VP9_HIGHBITDEPTH
}  # CONFIG_VP9

#
# Quantization
#
if (vpx_config("CONFIG_VP9_ENCODER") eq "yes") {
  add_proto qw/void vpx_quantize_b/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
  specialize qw/vpx_quantize_b neon sse2 ssse3 avx avx2 vsx/;

  add_proto qw/void vpx_quantize_b_32x32/, "const tran_low_t *coeff_ptr, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
  specialize qw/vpx_quantize_b_32x32 neon ssse3 avx avx2 vsx/;

  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
    add_proto qw/void vpx_highbd_quantize_b/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
    specialize qw/vpx_highbd_quantize_b neon sse2 avx2/;

    add_proto qw/void vpx_highbd_quantize_b_32x32/, "const tran_low_t *coeff_ptr, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
    specialize qw/vpx_highbd_quantize_b_32x32 neon sse2 avx2/;
  } else {
    specialize qw/vpx_quantize_b lsx/;

    specialize qw/vpx_quantize_b_32x32 lsx/;
  } # CONFIG_VP9_HIGHBITDEPTH
}  # CONFIG_VP9_ENCODER

if (vpx_config("CONFIG_ENCODERS") eq "yes") {
#
# Block subtraction
#
add_proto qw/void vpx_subtract_block/, "int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr, ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride";
specialize qw/vpx_subtract_block neon msa mmi sse2 avx2 vsx lsx/;

add_proto qw/int64_t/, "vpx_sse", "const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, int width, int height";
specialize qw/vpx_sse sse4_1 avx2 neon neon_dotprod/;

#
# Single block SAD
#
add_proto qw/unsigned int vpx_sad64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad64x64 neon neon_dotprod avx512 avx2 msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad64x32 neon neon_dotprod avx512 avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad32x64 neon neon_dotprod avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad32x32 neon neon_dotprod avx2 msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad32x16 neon neon_dotprod avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad16x32 neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad16x16 neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad16x8 neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad8x16 neon msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad8x8 neon msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad8x4 neon msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad4x8 neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad4x4 neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad_skip_64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_64x64 neon neon_dotprod avx512 avx2 sse2/;

add_proto qw/unsigned int vpx_sad_skip_64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_64x32 neon neon_dotprod avx512 avx2 sse2/;

add_proto qw/unsigned int vpx_sad_skip_32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_32x64 neon neon_dotprod avx2 sse2/;

add_proto qw/unsigned int vpx_sad_skip_32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_32x32 neon neon_dotprod avx2 sse2/;

add_proto qw/unsigned int vpx_sad_skip_32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_32x16 neon neon_dotprod avx2 sse2/;

add_proto qw/unsigned int vpx_sad_skip_16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_16x32 neon neon_dotprod sse2/;

add_proto qw/unsigned int vpx_sad_skip_16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_16x16 neon neon_dotprod sse2/;

add_proto qw/unsigned int vpx_sad_skip_16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_16x8 neon neon_dotprod sse2/;

add_proto qw/unsigned int vpx_sad_skip_8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_8x16 neon sse2/;

add_proto qw/unsigned int vpx_sad_skip_8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_8x8 neon sse2/;

add_proto qw/unsigned int vpx_sad_skip_8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_8x4 neon/;

add_proto qw/unsigned int vpx_sad_skip_4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_4x8 neon sse2/;

add_proto qw/unsigned int vpx_sad_skip_4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
specialize qw/vpx_sad_skip_4x4 neon/;

#
# Avg
#
if (vpx_config("CONFIG_VP9_ENCODER") eq "yes") {
  add_proto qw/unsigned int vpx_avg_8x8/, "const uint8_t *, int p";
  specialize qw/vpx_avg_8x8 sse2 neon msa/;

  add_proto qw/unsigned int vpx_avg_4x4/, "const uint8_t *, int p";
  specialize qw/vpx_avg_4x4 sse2 neon msa/;

  add_proto qw/void vpx_minmax_8x8/, "const uint8_t *s, int p, const uint8_t *d, int dp, int *min, int *max";
  specialize qw/vpx_minmax_8x8 sse2 neon msa/;

  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
    add_proto qw/void vpx_hadamard_8x8/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_hadamard_8x8 sse2 neon vsx lsx/, "$ssse3_x86_64";

    add_proto qw/void vpx_hadamard_16x16/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_hadamard_16x16 avx2 sse2 neon vsx lsx/;

    add_proto qw/void vpx_hadamard_32x32/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_hadamard_32x32 sse2 avx2 neon/;

    add_proto qw/void vpx_highbd_hadamard_8x8/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_highbd_hadamard_8x8 avx2 neon/;

    add_proto qw/void vpx_highbd_hadamard_16x16/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_highbd_hadamard_16x16 avx2 neon/;

    add_proto qw/void vpx_highbd_hadamard_32x32/, "const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff";
    specialize qw/vpx_highbd_hadamard_32x32 avx2 neon/;

    add_proto qw/int vpx_satd/, "const tran_low_t *coeff, int length";
    specialize qw/vpx_satd avx2 sse2 neon/;

    add_proto qw/int vpx_highbd_satd/, "const tran_low_t *coeff, int length";
    specialize qw/vpx_highbd_satd avx2 neon/;
  } else {
    add_proto qw/void vpx_hadamard_8x8/, "const int16_t *src_diff, ptrdiff_t src_stride, int16_t *coeff";
    specialize qw/vpx_hadamard_8x8 sse2 neon msa vsx lsx/, "$ssse3_x86_64";

    add_proto qw/void vpx_hadamard_16x16/, "const int16_t *src_diff, ptrdiff_t src_stride, int16_t *coeff";
    specialize qw/vpx_hadamard_16x16 avx2 sse2 neon msa vsx lsx/;

    add_proto qw/void vpx_hadamard_32x32/, "const int16_t *src_diff, ptrdiff_t src_stride, int16_t *coeff";
    specialize qw/vpx_hadamard_32x32 sse2 avx2 neon/;

    add_proto qw/int vpx_satd/, "const int16_t *coeff, int length";
    specialize qw/vpx_satd avx2 sse2 neon msa/;
  }

  add_proto qw/void vpx_int_pro_row/, "int16_t hbuf[16], const uint8_t *ref, const int ref_stride, const int height";
  specialize qw/vpx_int_pro_row neon sse2 msa/;
  add_proto qw/int16_t vpx_int_pro_col/, "const uint8_t *ref, const int width";
  specialize qw/vpx_int_pro_col neon sse2 msa/;

  add_proto qw/int vpx_vector_var/, "const int16_t *ref, const int16_t *src, const int bwl";
  specialize qw/vpx_vector_var neon sse2 msa/;
}  # CONFIG_VP9_ENCODER

add_proto qw/unsigned int vpx_sad64x64_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad64x64_avg neon neon_dotprod avx512 avx2 msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad64x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad64x32_avg neon neon_dotprod avx512 avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad32x64_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad32x64_avg neon neon_dotprod avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad32x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad32x32_avg neon neon_dotprod avx2 msa sse2 vsx mmi lsx/;

add_proto qw/unsigned int vpx_sad32x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad32x16_avg neon neon_dotprod avx2 msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad16x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad16x32_avg neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad16x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad16x16_avg neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad16x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad16x8_avg neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/unsigned int vpx_sad8x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad8x16_avg neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad8x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad8x8_avg neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad8x4_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad8x4_avg neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad4x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad4x8_avg neon msa sse2 mmi/;

add_proto qw/unsigned int vpx_sad4x4_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
specialize qw/vpx_sad4x4_avg neon msa sse2 mmi/;

#
# Multi-block SAD, comparing a reference to N independent blocks
#
add_proto qw/void vpx_sad64x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad64x64x4d avx512 avx2 neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_sad64x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad64x32x4d neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_sad32x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad32x64x4d neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_sad32x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad32x32x4d avx2 neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_sad32x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad32x16x4d neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/void vpx_sad16x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad16x32x4d neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/void vpx_sad16x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad16x16x4d neon neon_dotprod msa sse2 vsx mmi lsx/;

add_proto qw/void vpx_sad16x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad16x8x4d neon neon_dotprod msa sse2 vsx mmi/;

add_proto qw/void vpx_sad8x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad8x16x4d neon msa sse2 mmi/;

add_proto qw/void vpx_sad8x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad8x8x4d neon msa sse2 mmi lsx/;

add_proto qw/void vpx_sad8x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad8x4x4d neon msa sse2 mmi/;

add_proto qw/void vpx_sad4x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad4x8x4d neon msa sse2 mmi/;

add_proto qw/void vpx_sad4x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad4x4x4d neon msa sse2 mmi/;

add_proto qw/void vpx_sad_skip_64x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_64x64x4d neon neon_dotprod avx512 avx2 sse2/;

add_proto qw/void vpx_sad_skip_64x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_64x32x4d neon neon_dotprod avx512 avx2 sse2/;

add_proto qw/void vpx_sad_skip_32x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_32x64x4d neon neon_dotprod avx2 sse2/;

add_proto qw/void vpx_sad_skip_32x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_32x32x4d neon neon_dotprod avx2 sse2/;

add_proto qw/void vpx_sad_skip_32x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_32x16x4d neon neon_dotprod avx2 sse2/;

add_proto qw/void vpx_sad_skip_16x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_16x32x4d neon neon_dotprod sse2/;

add_proto qw/void vpx_sad_skip_16x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_16x16x4d neon neon_dotprod sse2/;

add_proto qw/void vpx_sad_skip_16x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_16x8x4d neon neon_dotprod sse2/;

add_proto qw/void vpx_sad_skip_8x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_8x16x4d neon sse2/;

add_proto qw/void vpx_sad_skip_8x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_8x8x4d neon sse2/;

add_proto qw/void vpx_sad_skip_8x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_8x4x4d neon/;

add_proto qw/void vpx_sad_skip_4x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_4x8x4d neon sse2/;

add_proto qw/void vpx_sad_skip_4x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
specialize qw/vpx_sad_skip_4x4x4d neon/;

add_proto qw/uint64_t vpx_sum_squares_2d_i16/, "const int16_t *src, int stride, int size";
specialize qw/vpx_sum_squares_2d_i16 neon sve sse2 msa/;

#
# Structured Similarity (SSIM)
#
if (vpx_config("CONFIG_INTERNAL_STATS") eq "yes") {
    add_proto qw/void vpx_ssim_parms_8x8/, "const uint8_t *s, int sp, const uint8_t *r, int rp, uint32_t *sum_s, uint32_t *sum_r, uint32_t *sum_sq_s, uint32_t *sum_sq_r, uint32_t *sum_sxr";
    specialize qw/vpx_ssim_parms_8x8/, "$sse2_x86_64";
}

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  #
  # Block subtraction
  #
  add_proto qw/void vpx_highbd_subtract_block/, "int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src8_ptr, ptrdiff_t src_stride, const uint8_t *pred8_ptr, ptrdiff_t pred_stride, int bd";
  specialize qw/vpx_highbd_subtract_block neon avx2/;

  add_proto qw/int64_t/, "vpx_highbd_sse", "const uint8_t *a8, int a_stride, const uint8_t *b8,int b_stride, int width, int height";
  specialize qw/vpx_highbd_sse sse4_1 avx2 neon/;

  #
  # Single block SAD
  #
  add_proto qw/unsigned int vpx_highbd_sad64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad64x64 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad64x32 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad32x64 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad32x32 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad32x16 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad16x32 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad16x16 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad16x8 sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad8x16 sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad8x8 sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad8x4 sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad4x8 neon/;

  add_proto qw/unsigned int vpx_highbd_sad4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad4x4 neon/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_64x64 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_64x32 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_32x64 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_32x32 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_32x16 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_16x32 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_16x16 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_16x8 neon sse2 avx2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_8x16 neon sse2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_8x8 neon sse2/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_8x4 neon/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_4x8 neon/;

  add_proto qw/unsigned int vpx_highbd_sad_skip_4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride";
  specialize qw/vpx_highbd_sad_skip_4x4 neon/;

  #
  # Avg
  #
  add_proto qw/unsigned int vpx_highbd_avg_8x8/, "const uint8_t *s8, int p";
  specialize qw/vpx_highbd_avg_8x8 sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_avg_4x4/, "const uint8_t *s8, int p";
  specialize qw/vpx_highbd_avg_4x4 sse2 neon/;

  add_proto qw/void vpx_highbd_minmax_8x8/, "const uint8_t *s8, int p, const uint8_t *d8, int dp, int *min, int *max";
  specialize qw/vpx_highbd_minmax_8x8 neon/;

  add_proto qw/unsigned int vpx_highbd_sad64x64_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad64x64_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad64x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad64x32_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x64_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad32x64_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad32x32_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad32x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad32x16_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x32_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad16x32_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad16x16_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad16x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad16x8_avg sse2 neon avx2/;

  add_proto qw/unsigned int vpx_highbd_sad8x16_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad8x16_avg sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad8x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad8x8_avg sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad8x4_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad8x4_avg sse2 neon/;

  add_proto qw/unsigned int vpx_highbd_sad4x8_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad4x8_avg neon/;

  add_proto qw/unsigned int vpx_highbd_sad4x4_avg/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, const uint8_t *second_pred";
  specialize qw/vpx_highbd_sad4x4_avg neon/;

  #
  # Multi-block SAD, comparing a reference to N independent blocks
  #
  add_proto qw/void vpx_highbd_sad64x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad64x64x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad64x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad64x32x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad32x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad32x64x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad32x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad32x32x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad32x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad32x16x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad16x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad16x32x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad16x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad16x16x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad16x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad16x8x4d sse2 neon avx2/;

  add_proto qw/void vpx_highbd_sad8x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad8x16x4d sse2 neon/;

  add_proto qw/void vpx_highbd_sad8x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad8x8x4d sse2 neon/;

  add_proto qw/void vpx_highbd_sad8x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad8x4x4d sse2 neon/;

  add_proto qw/void vpx_highbd_sad4x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad4x8x4d sse2 neon/;

  add_proto qw/void vpx_highbd_sad4x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad4x4x4d sse2 neon/;

  add_proto qw/void vpx_highbd_sad_skip_64x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_64x64x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_64x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_64x32x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_32x64x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_32x64x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_32x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_32x32x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_32x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_32x16x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_16x32x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_16x32x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_16x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_16x16x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_16x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_16x8x4d neon sse2 avx2/;

  add_proto qw/void vpx_highbd_sad_skip_8x16x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_8x16x4d neon sse2/;

  add_proto qw/void vpx_highbd_sad_skip_8x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_8x8x4d neon sse2/;

  add_proto qw/void vpx_highbd_sad_skip_8x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_8x4x4d neon/;

  add_proto qw/void vpx_highbd_sad_skip_4x8x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_4x8x4d neon sse2/;

  add_proto qw/void vpx_highbd_sad_skip_4x4x4d/, "const uint8_t *src_ptr, int src_stride, const uint8_t *const ref_array[4], int ref_stride, uint32_t sad_array[4]";
  specialize qw/vpx_highbd_sad_skip_4x4x4d neon/;

  #
  # Structured Similarity (SSIM)
  #
  if (vpx_config("CONFIG_INTERNAL_STATS") eq "yes") {
    add_proto qw/void vpx_highbd_ssim_parms_8x8/, "const uint16_t *s, int sp, const uint16_t *r, int rp, uint32_t *sum_s, uint32_t *sum_r, uint32_t *sum_sq_s, uint32_t *sum_sq_r, uint32_t *sum_sxr";
  }
}  # CONFIG_VP9_HIGHBITDEPTH
}  # CONFIG_ENCODERS

if (vpx_config("CONFIG_ENCODERS") eq "yes" || vpx_config("CONFIG_POSTPROC") eq "yes" || vpx_config("CONFIG_VP9_POSTPROC") eq "yes") {

#
# Variance
#
add_proto qw/unsigned int vpx_variance64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance64x64 sse2 avx2 neon neon_dotprod msa mmi vsx lsx/;

add_proto qw/unsigned int vpx_variance64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance64x32 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance32x64 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance32x32 sse2 avx2 neon neon_dotprod msa mmi vsx lsx/;

add_proto qw/unsigned int vpx_variance32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance32x16 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance16x32 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance16x16 sse2 avx2 neon neon_dotprod msa mmi vsx lsx/;

add_proto qw/unsigned int vpx_variance16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance16x8 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance8x16 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance8x8 sse2 avx2 neon neon_dotprod msa mmi vsx lsx/;

add_proto qw/unsigned int vpx_variance8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance8x4 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance4x8 sse2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_variance4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_variance4x4 sse2 neon neon_dotprod msa mmi vsx/;

#
# Specialty Variance
#
add_proto qw/void vpx_get16x16var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_get16x16var sse2 avx2 neon neon_dotprod msa vsx lsx/;

add_proto qw/void vpx_get8x8var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_get8x8var sse2 neon neon_dotprod msa vsx/;

add_proto qw/unsigned int vpx_mse16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_mse16x16 sse2 avx2 neon neon_dotprod msa mmi vsx lsx/;

add_proto qw/unsigned int vpx_mse16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_mse16x8 sse2 avx2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_mse8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_mse8x16 sse2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_mse8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_mse8x8 sse2 neon neon_dotprod msa mmi vsx/;

add_proto qw/unsigned int vpx_get_mb_ss/, "const int16_t *";
  specialize qw/vpx_get_mb_ss sse2 msa vsx/;

add_proto qw/unsigned int vpx_get4x4sse_cs/, "const unsigned char *src_ptr, int src_stride, const unsigned char *ref_ptr, int ref_stride";
  specialize qw/vpx_get4x4sse_cs neon neon_dotprod msa vsx/;

add_proto qw/void vpx_comp_avg_pred/, "uint8_t *comp_pred, const uint8_t *pred, int width, int height, const uint8_t *ref, int ref_stride";
  specialize qw/vpx_comp_avg_pred neon sse2 avx2 vsx lsx/;

#
# Subpixel Variance
#
add_proto qw/uint32_t vpx_sub_pixel_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance64x64 avx2 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance64x32 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance32x64 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance32x32 avx2 neon msa mmi sse2 ssse3 lsx/;

add_proto qw/uint32_t vpx_sub_pixel_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance32x16 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance16x32 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance16x16 neon msa mmi sse2 ssse3 lsx/;

add_proto qw/uint32_t vpx_sub_pixel_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance16x8 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance8x16 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance8x8 neon msa mmi sse2 ssse3 lsx/;

add_proto qw/uint32_t vpx_sub_pixel_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance8x4 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance4x8 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_sub_pixel_variance4x4 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance64x64 neon avx2 msa mmi sse2 ssse3 lsx/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance64x32 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance32x64 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance32x32 neon avx2 msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance32x16 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance16x32 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance16x16 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance16x8 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance8x16 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance8x8 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance8x4 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance4x8 neon msa mmi sse2 ssse3/;

add_proto qw/uint32_t vpx_sub_pixel_avg_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_sub_pixel_avg_variance4x4 neon msa mmi sse2 ssse3/;

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  add_proto qw/unsigned int vpx_highbd_12_variance64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance64x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance64x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance32x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance32x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance32x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance16x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance16x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance16x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance8x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance8x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_variance8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance8x4 neon sve/;
  add_proto qw/unsigned int vpx_highbd_12_variance4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance4x8 neon sve/;
  add_proto qw/unsigned int vpx_highbd_12_variance4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_variance4x4 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance64x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance64x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance32x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance32x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance32x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance16x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance16x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance16x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance8x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance8x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_variance8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance8x4 neon sve/;
  add_proto qw/unsigned int vpx_highbd_10_variance4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance4x8 neon sve/;
  add_proto qw/unsigned int vpx_highbd_10_variance4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_variance4x4 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance64x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance64x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance64x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance64x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance32x64/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance32x64 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance32x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance32x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance32x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance32x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance16x32/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance16x32 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance16x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance16x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance8x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance8x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_variance8x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance8x4 neon sve/;
  add_proto qw/unsigned int vpx_highbd_8_variance4x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance4x8 neon sve/;
  add_proto qw/unsigned int vpx_highbd_8_variance4x4/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_variance4x4 neon sve/;

  add_proto qw/void vpx_highbd_8_get16x16var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_8_get16x16var sse2 neon sve/;

  add_proto qw/void vpx_highbd_8_get8x8var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_8_get8x8var sse2 neon sve/;

  add_proto qw/void vpx_highbd_10_get16x16var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_10_get16x16var sse2 neon sve/;

  add_proto qw/void vpx_highbd_10_get8x8var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_10_get8x8var sse2 neon sve/;

  add_proto qw/void vpx_highbd_12_get16x16var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_12_get16x16var sse2 neon sve/;

  add_proto qw/void vpx_highbd_12_get8x8var/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse, int *sum";
  specialize qw/vpx_highbd_12_get8x8var sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_8_mse16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_mse16x16 sse2 neon neon_dotprod/;

  add_proto qw/unsigned int vpx_highbd_8_mse16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_mse16x8 neon neon_dotprod/;
  add_proto qw/unsigned int vpx_highbd_8_mse8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_mse8x16 neon neon_dotprod/;
  add_proto qw/unsigned int vpx_highbd_8_mse8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_8_mse8x8 sse2 neon neon_dotprod/;

  add_proto qw/unsigned int vpx_highbd_10_mse16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_mse16x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_10_mse16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_mse16x8 neon sve/;
  add_proto qw/unsigned int vpx_highbd_10_mse8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_mse8x16 neon sve/;
  add_proto qw/unsigned int vpx_highbd_10_mse8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_10_mse8x8 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_mse16x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_mse16x16 sse2 neon sve/;

  add_proto qw/unsigned int vpx_highbd_12_mse16x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_mse16x8 neon sve/;
  add_proto qw/unsigned int vpx_highbd_12_mse8x16/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_mse8x16 neon sve/;
  add_proto qw/unsigned int vpx_highbd_12_mse8x8/, "const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse";
  specialize qw/vpx_highbd_12_mse8x8 sse2 neon sve/;

  add_proto qw/void vpx_highbd_comp_avg_pred/, "uint16_t *comp_pred, const uint16_t *pred, int width, int height, const uint16_t *ref, int ref_stride";
  specialize qw/vpx_highbd_comp_avg_pred neon sse2/;

  #
  # Subpixel Variance
  #
  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_12_sub_pixel_variance4x4 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_10_sub_pixel_variance4x4 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse";
  specialize qw/vpx_highbd_8_sub_pixel_variance4x4 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_12_sub_pixel_avg_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_12_sub_pixel_avg_variance4x4 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_10_sub_pixel_avg_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_10_sub_pixel_avg_variance4x4 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance64x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance64x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance64x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance64x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance32x64/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance32x64 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance32x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance32x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance32x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance32x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance16x32/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance16x32 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance16x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance16x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance16x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance16x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance8x16/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance8x16 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance8x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance8x8 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance8x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance8x4 sse2 neon/;

  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance4x8/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance4x8 neon/;
  add_proto qw/uint32_t vpx_highbd_8_sub_pixel_avg_variance4x4/, "const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, const uint8_t *ref_ptr, int ref_stride, uint32_t *sse, const uint8_t *second_pred";
  specialize qw/vpx_highbd_8_sub_pixel_avg_variance4x4 neon/;

}  # CONFIG_VP9_HIGHBITDEPTH

#
# Post Processing
#
if (vpx_config("CONFIG_POSTPROC") eq "yes" || vpx_config("CONFIG_VP9_POSTPROC") eq "yes") {
    add_proto qw/void vpx_plane_add_noise/, "uint8_t *start, const int8_t *noise, int blackclamp, int whiteclamp, int width, int height, int pitch";
    specialize qw/vpx_plane_add_noise sse2 msa/;

    add_proto qw/void vpx_mbpost_proc_down/, "unsigned char *dst, int pitch, int rows, int cols,int flimit";
    specialize qw/vpx_mbpost_proc_down sse2 neon msa vsx/;

    add_proto qw/void vpx_mbpost_proc_across_ip/, "unsigned char *src, int pitch, int rows, int cols,int flimit";
    specialize qw/vpx_mbpost_proc_across_ip sse2 neon msa vsx/;

    add_proto qw/void vpx_post_proc_down_and_across_mb_row/, "unsigned char *src, unsigned char *dst, int src_pitch, int dst_pitch, int cols, unsigned char *flimits, int size";
    specialize qw/vpx_post_proc_down_and_across_mb_row sse2 neon msa vsx/;

}

}  # CONFIG_ENCODERS || CONFIG_POSTPROC || CONFIG_VP9_POSTPROC

1;
