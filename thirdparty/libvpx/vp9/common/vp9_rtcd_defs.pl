##
##  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

sub vp9_common_forward_decls() {
print <<EOF
/*
 * VP9
 */

#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_filter.h"
#if !CONFIG_REALTIME_ONLY && CONFIG_VP9_ENCODER
#include "vp9/encoder/vp9_temporal_filter.h"
#endif

struct macroblockd;

/* Encoder forward decls */
struct macroblock;
struct macroblock_plane;
struct vp9_sad_table;
struct ScanOrder;
struct search_site_config;
struct mv;
union int_mv;
struct yv12_buffer_config;
EOF
}
forward_decls qw/vp9_common_forward_decls/;

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
# post proc
#
if (vpx_config("CONFIG_VP9_POSTPROC") eq "yes") {
add_proto qw/void vp9_filter_by_weight16x16/, "const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int src_weight";
specialize qw/vp9_filter_by_weight16x16 sse2 msa/;

add_proto qw/void vp9_filter_by_weight8x8/, "const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int src_weight";
specialize qw/vp9_filter_by_weight8x8 sse2 msa/;
}

#
# dct
#
# Force C versions if CONFIG_EMULATE_HARDWARE is 1
add_proto qw/void vp9_iht4x4_16_add/, "const tran_low_t *input, uint8_t *dest, int stride, int tx_type";

add_proto qw/void vp9_iht8x8_64_add/, "const tran_low_t *input, uint8_t *dest, int stride, int tx_type";

add_proto qw/void vp9_iht16x16_256_add/, "const tran_low_t *input, uint8_t *dest, int stride, int tx_type";

if (vpx_config("CONFIG_EMULATE_HARDWARE") ne "yes") {
  # Note that there are more specializations appended when
  # CONFIG_VP9_HIGHBITDEPTH is off.
  specialize qw/vp9_iht4x4_16_add neon sse2 vsx/;
  specialize qw/vp9_iht8x8_64_add neon sse2 vsx/;
  specialize qw/vp9_iht16x16_256_add neon sse2 vsx/;
  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") ne "yes") {
    # Note that these specializations are appended to the above ones.
    specialize qw/vp9_iht4x4_16_add dspr2 msa/;
    specialize qw/vp9_iht8x8_64_add dspr2 msa/;
    specialize qw/vp9_iht16x16_256_add dspr2 msa/;
  }
}

# High bitdepth functions
if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  #
  # post proc
  #
  if (vpx_config("CONFIG_VP9_POSTPROC") eq "yes") {
    add_proto qw/void vp9_highbd_mbpost_proc_down/, "uint16_t *dst, int pitch, int rows, int cols, int flimit";

    add_proto qw/void vp9_highbd_mbpost_proc_across_ip/, "uint16_t *src, int pitch, int rows, int cols, int flimit";

    add_proto qw/void vp9_highbd_post_proc_down_and_across/, "const uint16_t *src_ptr, uint16_t *dst_ptr, int src_pixels_per_line, int dst_pixels_per_line, int rows, int cols, int flimit";
  }

  #
  # dct
  #
  # Note as optimized versions of these functions are added we need to add a check to ensure
  # that when CONFIG_EMULATE_HARDWARE is on, it defaults to the C versions only.
  add_proto qw/void vp9_highbd_iht4x4_16_add/, "const tran_low_t *input, uint16_t *dest, int stride, int tx_type, int bd";

  add_proto qw/void vp9_highbd_iht8x8_64_add/, "const tran_low_t *input, uint16_t *dest, int stride, int tx_type, int bd";

  add_proto qw/void vp9_highbd_iht16x16_256_add/, "const tran_low_t *input, uint16_t *dest, int stride, int tx_type, int bd";

  if (vpx_config("CONFIG_EMULATE_HARDWARE") ne "yes") {
    specialize qw/vp9_highbd_iht4x4_16_add neon sse4_1/;
    specialize qw/vp9_highbd_iht8x8_64_add neon sse4_1/;
    specialize qw/vp9_highbd_iht16x16_256_add neon sse4_1/;
  }
}

#
# Encoder functions below this point.
#
if (vpx_config("CONFIG_VP9_ENCODER") eq "yes") {
# ENCODEMB INVOKE

#
# Denoiser
#
if (vpx_config("CONFIG_VP9_TEMPORAL_DENOISING") eq "yes") {
  add_proto qw/int vp9_denoiser_filter/, "const uint8_t *sig, int sig_stride, const uint8_t *mc_avg, int mc_avg_stride, uint8_t *avg, int avg_stride, int increase_denoising, BLOCK_SIZE bs, int motion_magnitude";
  specialize qw/vp9_denoiser_filter neon sse2/;
}

add_proto qw/int64_t vp9_block_error/, "const tran_low_t *coeff, const tran_low_t *dqcoeff, intptr_t block_size, int64_t *ssz";

add_proto qw/int64_t vp9_block_error_fp/, "const tran_low_t *coeff, const tran_low_t *dqcoeff, int block_size";
specialize qw/vp9_block_error_fp neon sve avx2 sse2/;

add_proto qw/void vp9_quantize_fp/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
specialize qw/vp9_quantize_fp neon sse2 ssse3 avx2 vsx/;

add_proto qw/void vp9_quantize_fp_32x32/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
specialize qw/vp9_quantize_fp_32x32 neon ssse3 avx2 vsx/;

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
  specialize qw/vp9_block_error neon sve avx2 sse2/;

  add_proto qw/int64_t vp9_highbd_block_error/, "const tran_low_t *coeff, const tran_low_t *dqcoeff, intptr_t block_size, int64_t *ssz, int bd";
  specialize qw/vp9_highbd_block_error neon sse2/;
} else {
  specialize qw/vp9_block_error neon sve avx2 msa sse2/;
}

# fdct functions

add_proto qw/void vp9_fht4x4/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";

add_proto qw/void vp9_fht8x8/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";

add_proto qw/void vp9_fht16x16/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";

add_proto qw/void vp9_fwht4x4/, "const int16_t *input, tran_low_t *output, int stride";

# Note that there are more specializations appended when CONFIG_VP9_HIGHBITDEPTH
# is off.
specialize qw/vp9_fht4x4 sse2 neon/;
specialize qw/vp9_fht8x8 sse2 neon/;
specialize qw/vp9_fht16x16 sse2 neon/;
specialize qw/vp9_fwht4x4 sse2/;
if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") ne "yes") {
  # Note that these specializations are appended to the above ones.
  specialize qw/vp9_fht4x4 msa/;
  specialize qw/vp9_fht8x8 msa/;
  specialize qw/vp9_fht16x16 msa/;
  specialize qw/vp9_fwht4x4 msa/;
}

#
# Motion search
#
add_proto qw/int vp9_diamond_search_sad/, "const struct macroblock *x, const struct search_site_config *cfg,  struct mv *ref_mv, uint32_t start_mv_sad, struct mv *best_mv, int search_param, int sad_per_bit, int *num00, const struct vp9_sad_table *sad_fn_ptr, const struct mv *center_mv";
specialize qw/vp9_diamond_search_sad neon/;

#
# Apply temporal filter
#
if (vpx_config("CONFIG_REALTIME_ONLY") ne "yes") {
add_proto qw/void vp9_apply_temporal_filter/, "const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre, int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *const blk_fw, int use_32x32, uint32_t *y_accumulator, uint16_t *y_count, uint32_t *u_accumulator, uint16_t *u_count, uint32_t *v_accumulator, uint16_t *v_count";
specialize qw/vp9_apply_temporal_filter sse4_1 neon/;

  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
    add_proto qw/void vp9_highbd_apply_temporal_filter/, "const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre, int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride, const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *const blk_fw, int use_32x32, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count";
    specialize qw/vp9_highbd_apply_temporal_filter sse4_1 neon/;
  }
}

#
# 12-tap filter used in prediction data generation during temporal filtering
#
if (vpx_config("CONFIG_REALTIME_ONLY") ne "yes") {
  add_proto qw/void vpx_convolve12_vert/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
  specialize qw/vpx_convolve12_vert ssse3 avx2 neon neon_dotprod neon_i8mm/;

  add_proto qw/void vpx_convolve12_horiz/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
  specialize qw/vpx_convolve12_horiz ssse3 avx2 neon neon_dotprod neon_i8mm/;

  add_proto qw/void vpx_convolve12/, "const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h";
  specialize qw/vpx_convolve12 ssse3 avx2 neon neon_dotprod neon_i8mm/;

  if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {
    add_proto qw/void vpx_highbd_convolve12_vert/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
    specialize qw/vpx_highbd_convolve12_vert ssse3 avx2 neon sve2/;

    add_proto qw/void vpx_highbd_convolve12_horiz/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
    specialize qw/vpx_highbd_convolve12_horiz ssse3 avx2 neon sve2/;

    add_proto qw/void vpx_highbd_convolve12/, "const uint16_t *src, ptrdiff_t src_stride, uint16_t *dst, ptrdiff_t dst_stride, const InterpKernel12 *filter, int x0_q4, int x_step_q4, int y0_q4, int y_step_q4, int w, int h, int bd";
    specialize qw/vpx_highbd_convolve12 ssse3 avx2 neon sve2/;
  }
}

if (vpx_config("CONFIG_VP9_HIGHBITDEPTH") eq "yes") {

  # ENCODEMB INVOKE

  add_proto qw/void vp9_highbd_quantize_fp/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
  specialize qw/vp9_highbd_quantize_fp avx2 neon/;

  add_proto qw/void vp9_highbd_quantize_fp_32x32/, "const tran_low_t *coeff_ptr, intptr_t n_coeffs, const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const struct ScanOrder *const scan_order";
  specialize qw/vp9_highbd_quantize_fp_32x32 avx2 neon/;

  # fdct functions
  add_proto qw/void vp9_highbd_fht4x4/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";
  specialize qw/vp9_highbd_fht4x4 neon/;

  add_proto qw/void vp9_highbd_fht8x8/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";
  specialize qw/vp9_highbd_fht8x8 neon/;

  add_proto qw/void vp9_highbd_fht16x16/, "const int16_t *input, tran_low_t *output, int stride, int tx_type";
  specialize qw/vp9_highbd_fht16x16 neon/;

  add_proto qw/void vp9_highbd_fwht4x4/, "const int16_t *input, tran_low_t *output, int stride";

  add_proto qw/void vp9_highbd_temporal_filter_apply/, "const uint8_t *frame1, unsigned int stride, const uint8_t *frame2, unsigned int block_width, unsigned int block_height, int strength, int *blk_fw, int use_32x32, uint32_t *accumulator, uint16_t *count";

}
# End vp9_high encoder functions

#
# frame based scale
#
add_proto qw/void vp9_scale_and_extend_frame/, "const struct yv12_buffer_config *src, struct yv12_buffer_config *dst, INTERP_FILTER filter_type, int phase_scaler";
specialize qw/vp9_scale_and_extend_frame neon ssse3/;

}
# end encoder functions
1;
