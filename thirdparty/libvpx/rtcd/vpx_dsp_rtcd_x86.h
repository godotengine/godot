#ifndef VPX_DSP_RTCD_H_
#define VPX_DSP_RTCD_H_

#ifdef RTCD_C
#define RTCD_EXTERN
#else
#define RTCD_EXTERN extern
#endif

/*
 * DSP
 */

#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"


#ifdef __cplusplus
extern "C" {
#endif

void vpx_convolve8_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve8_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8_avg)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve8_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8_avg_horiz)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve8_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_vert_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_avg_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8_avg_vert)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve8_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_horiz_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8_horiz)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve8_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_vert_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve8_vert_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve8_vert)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve_avg_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve_avg_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve_avg)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_convolve_copy_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_convolve_copy_sse2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_convolve_copy)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_d117_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d117_predictor_16x16 vpx_d117_predictor_16x16_c

void vpx_d117_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d117_predictor_32x32 vpx_d117_predictor_32x32_c

void vpx_d117_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d117_predictor_4x4 vpx_d117_predictor_4x4_c

void vpx_d117_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d117_predictor_8x8 vpx_d117_predictor_8x8_c

void vpx_d135_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d135_predictor_16x16 vpx_d135_predictor_16x16_c

void vpx_d135_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d135_predictor_32x32 vpx_d135_predictor_32x32_c

void vpx_d135_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d135_predictor_4x4 vpx_d135_predictor_4x4_c

void vpx_d135_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d135_predictor_8x8 vpx_d135_predictor_8x8_c

void vpx_d153_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d153_predictor_16x16_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d153_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d153_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d153_predictor_32x32_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d153_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d153_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d153_predictor_4x4_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d153_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d153_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d153_predictor_8x8_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d153_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d207_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d207_predictor_16x16_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d207_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d207_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d207_predictor_32x32_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d207_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d207_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d207_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d207_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d207_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d207_predictor_8x8_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d207_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d207e_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d207e_predictor_16x16 vpx_d207e_predictor_16x16_c

void vpx_d207e_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d207e_predictor_32x32 vpx_d207e_predictor_32x32_c

void vpx_d207e_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d207e_predictor_4x4 vpx_d207e_predictor_4x4_c

void vpx_d207e_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d207e_predictor_8x8 vpx_d207e_predictor_8x8_c

void vpx_d45_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d45_predictor_16x16_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d45_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d45_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d45_predictor_32x32_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d45_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d45_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d45_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d45_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d45_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d45_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d45_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d45e_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d45e_predictor_16x16 vpx_d45e_predictor_16x16_c

void vpx_d45e_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d45e_predictor_32x32 vpx_d45e_predictor_32x32_c

void vpx_d45e_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d45e_predictor_4x4 vpx_d45e_predictor_4x4_c

void vpx_d45e_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d45e_predictor_8x8 vpx_d45e_predictor_8x8_c

void vpx_d63_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d63_predictor_16x16_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d63_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d63_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d63_predictor_32x32_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d63_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d63_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d63_predictor_4x4_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d63_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d63_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_d63_predictor_8x8_ssse3(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_d63_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_d63e_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d63e_predictor_16x16 vpx_d63e_predictor_16x16_c

void vpx_d63e_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d63e_predictor_32x32 vpx_d63e_predictor_32x32_c

void vpx_d63e_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d63e_predictor_4x4 vpx_d63e_predictor_4x4_c

void vpx_d63e_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d63e_predictor_8x8 vpx_d63e_predictor_8x8_c

void vpx_d63f_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_d63f_predictor_4x4 vpx_d63f_predictor_4x4_c

void vpx_dc_128_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_128_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_128_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_128_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_128_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_128_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_128_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_128_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_128_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_128_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_128_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_128_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_left_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_left_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_left_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_left_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_left_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_left_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_left_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_left_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_left_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_left_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_left_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_left_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_top_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_top_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_top_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_top_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_top_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_top_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_top_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_top_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_top_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_dc_top_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_dc_top_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_dc_top_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_h_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_h_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_h_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_h_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_h_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_h_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_h_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_h_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_h_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_h_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_h_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_h_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_he_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_he_predictor_4x4 vpx_he_predictor_4x4_c

void vpx_idct16x16_10_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct16x16_10_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct16x16_10_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct16x16_1_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct16x16_1_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct16x16_1_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct16x16_256_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct16x16_256_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct16x16_256_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct32x32_1024_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct32x32_1024_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct32x32_1024_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct32x32_135_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct32x32_1024_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct32x32_135_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct32x32_1_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct32x32_1_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct32x32_1_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct32x32_34_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct32x32_34_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct32x32_34_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct4x4_16_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct4x4_1_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct4x4_1_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct4x4_1_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct8x8_12_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct8x8_12_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct8x8_12_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct8x8_1_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct8x8_1_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct8x8_1_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_idct8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_idct8x8_64_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_idct8x8_64_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_iwht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
void vpx_iwht4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest, int dest_stride);
RTCD_EXTERN void (*vpx_iwht4x4_16_add)(const tran_low_t *input, uint8_t *dest, int dest_stride);

void vpx_iwht4x4_1_add_c(const tran_low_t *input, uint8_t *dest, int dest_stride);
#define vpx_iwht4x4_1_add vpx_iwht4x4_1_add_c

void vpx_lpf_horizontal_4_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_horizontal_4_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_horizontal_4)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_horizontal_4_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
void vpx_lpf_horizontal_4_dual_sse2(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
RTCD_EXTERN void (*vpx_lpf_horizontal_4_dual)(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);

void vpx_lpf_horizontal_8_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_horizontal_8_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_horizontal_8)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_horizontal_8_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
void vpx_lpf_horizontal_8_dual_sse2(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
RTCD_EXTERN void (*vpx_lpf_horizontal_8_dual)(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);

void vpx_lpf_horizontal_edge_16_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_horizontal_edge_16_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_horizontal_edge_16)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_horizontal_edge_8_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_horizontal_edge_8_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_horizontal_edge_8)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_vertical_16_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_vertical_16_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_vertical_16)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_vertical_16_dual_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_vertical_16_dual_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_vertical_16_dual)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_vertical_4_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_vertical_4_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_vertical_4)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_vertical_4_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
void vpx_lpf_vertical_4_dual_sse2(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
RTCD_EXTERN void (*vpx_lpf_vertical_4_dual)(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);

void vpx_lpf_vertical_8_c(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void vpx_lpf_vertical_8_sse2(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
RTCD_EXTERN void (*vpx_lpf_vertical_8)(uint8_t *s, int pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);

void vpx_lpf_vertical_8_dual_c(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
void vpx_lpf_vertical_8_dual_sse2(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);
RTCD_EXTERN void (*vpx_lpf_vertical_8_dual)(uint8_t *s, int pitch, const uint8_t *blimit0, const uint8_t *limit0, const uint8_t *thresh0, const uint8_t *blimit1, const uint8_t *limit1, const uint8_t *thresh1);

void vpx_scaled_2d_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void vpx_scaled_2d_ssse3(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
RTCD_EXTERN void (*vpx_scaled_2d)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);

void vpx_scaled_avg_2d_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
#define vpx_scaled_avg_2d vpx_scaled_avg_2d_c

void vpx_scaled_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
#define vpx_scaled_avg_horiz vpx_scaled_avg_horiz_c

void vpx_scaled_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
#define vpx_scaled_avg_vert vpx_scaled_avg_vert_c

void vpx_scaled_horiz_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
#define vpx_scaled_horiz vpx_scaled_horiz_c

void vpx_scaled_vert_c(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
#define vpx_scaled_vert vpx_scaled_vert_c

void vpx_tm_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_tm_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_tm_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_tm_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_tm_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_tm_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_tm_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_tm_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_tm_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_tm_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_tm_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_tm_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_v_predictor_16x16_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_v_predictor_16x16_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_v_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_v_predictor_32x32_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_v_predictor_32x32_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_v_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_v_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_v_predictor_4x4_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_v_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_v_predictor_8x8_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void vpx_v_predictor_8x8_sse2(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
RTCD_EXTERN void (*vpx_v_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);

void vpx_ve_predictor_4x4_c(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
#define vpx_ve_predictor_4x4 vpx_ve_predictor_4x4_c

void vpx_dsp_rtcd(void);

#ifdef RTCD_C
#include "vpx_ports/x86.h"
static void setup_rtcd_internal(void)
{
    int flags = x86_simd_caps();

    vpx_convolve8 = vpx_convolve8_c;
    if (flags & HAS_SSE2) vpx_convolve8 = vpx_convolve8_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8 = vpx_convolve8_ssse3;
    vpx_convolve8_avg = vpx_convolve8_avg_c;
    if (flags & HAS_SSE2) vpx_convolve8_avg = vpx_convolve8_avg_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8_avg = vpx_convolve8_avg_ssse3;
    vpx_convolve8_avg_horiz = vpx_convolve8_avg_horiz_c;
    if (flags & HAS_SSE2) vpx_convolve8_avg_horiz = vpx_convolve8_avg_horiz_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8_avg_horiz = vpx_convolve8_avg_horiz_ssse3;
    vpx_convolve8_avg_vert = vpx_convolve8_avg_vert_c;
    if (flags & HAS_SSE2) vpx_convolve8_avg_vert = vpx_convolve8_avg_vert_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8_avg_vert = vpx_convolve8_avg_vert_ssse3;
    vpx_convolve8_horiz = vpx_convolve8_horiz_c;
    if (flags & HAS_SSE2) vpx_convolve8_horiz = vpx_convolve8_horiz_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8_horiz = vpx_convolve8_horiz_ssse3;
    vpx_convolve8_vert = vpx_convolve8_vert_c;
    if (flags & HAS_SSE2) vpx_convolve8_vert = vpx_convolve8_vert_sse2;
    if (flags & HAS_SSSE3) vpx_convolve8_vert = vpx_convolve8_vert_ssse3;
    vpx_convolve_avg = vpx_convolve_avg_c;
    if (flags & HAS_SSE2) vpx_convolve_avg = vpx_convolve_avg_sse2;
    vpx_convolve_copy = vpx_convolve_copy_c;
    if (flags & HAS_SSE2) vpx_convolve_copy = vpx_convolve_copy_sse2;
    vpx_d153_predictor_16x16 = vpx_d153_predictor_16x16_c;
    if (flags & HAS_SSSE3) vpx_d153_predictor_16x16 = vpx_d153_predictor_16x16_ssse3;
    vpx_d153_predictor_32x32 = vpx_d153_predictor_32x32_c;
    if (flags & HAS_SSSE3) vpx_d153_predictor_32x32 = vpx_d153_predictor_32x32_ssse3;
    vpx_d153_predictor_4x4 = vpx_d153_predictor_4x4_c;
    if (flags & HAS_SSSE3) vpx_d153_predictor_4x4 = vpx_d153_predictor_4x4_ssse3;
    vpx_d153_predictor_8x8 = vpx_d153_predictor_8x8_c;
    if (flags & HAS_SSSE3) vpx_d153_predictor_8x8 = vpx_d153_predictor_8x8_ssse3;
    vpx_d207_predictor_16x16 = vpx_d207_predictor_16x16_c;
    if (flags & HAS_SSSE3) vpx_d207_predictor_16x16 = vpx_d207_predictor_16x16_ssse3;
    vpx_d207_predictor_32x32 = vpx_d207_predictor_32x32_c;
    if (flags & HAS_SSSE3) vpx_d207_predictor_32x32 = vpx_d207_predictor_32x32_ssse3;
    vpx_d207_predictor_4x4 = vpx_d207_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_d207_predictor_4x4 = vpx_d207_predictor_4x4_sse2;
    vpx_d207_predictor_8x8 = vpx_d207_predictor_8x8_c;
    if (flags & HAS_SSSE3) vpx_d207_predictor_8x8 = vpx_d207_predictor_8x8_ssse3;
    vpx_d45_predictor_16x16 = vpx_d45_predictor_16x16_c;
    if (flags & HAS_SSSE3) vpx_d45_predictor_16x16 = vpx_d45_predictor_16x16_ssse3;
    vpx_d45_predictor_32x32 = vpx_d45_predictor_32x32_c;
    if (flags & HAS_SSSE3) vpx_d45_predictor_32x32 = vpx_d45_predictor_32x32_ssse3;
    vpx_d45_predictor_4x4 = vpx_d45_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_d45_predictor_4x4 = vpx_d45_predictor_4x4_sse2;
    vpx_d45_predictor_8x8 = vpx_d45_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_d45_predictor_8x8 = vpx_d45_predictor_8x8_sse2;
    vpx_d63_predictor_16x16 = vpx_d63_predictor_16x16_c;
    if (flags & HAS_SSSE3) vpx_d63_predictor_16x16 = vpx_d63_predictor_16x16_ssse3;
    vpx_d63_predictor_32x32 = vpx_d63_predictor_32x32_c;
    if (flags & HAS_SSSE3) vpx_d63_predictor_32x32 = vpx_d63_predictor_32x32_ssse3;
    vpx_d63_predictor_4x4 = vpx_d63_predictor_4x4_c;
    if (flags & HAS_SSSE3) vpx_d63_predictor_4x4 = vpx_d63_predictor_4x4_ssse3;
    vpx_d63_predictor_8x8 = vpx_d63_predictor_8x8_c;
    if (flags & HAS_SSSE3) vpx_d63_predictor_8x8 = vpx_d63_predictor_8x8_ssse3;
    vpx_dc_128_predictor_16x16 = vpx_dc_128_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_dc_128_predictor_16x16 = vpx_dc_128_predictor_16x16_sse2;
    vpx_dc_128_predictor_32x32 = vpx_dc_128_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_dc_128_predictor_32x32 = vpx_dc_128_predictor_32x32_sse2;
    vpx_dc_128_predictor_4x4 = vpx_dc_128_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_dc_128_predictor_4x4 = vpx_dc_128_predictor_4x4_sse2;
    vpx_dc_128_predictor_8x8 = vpx_dc_128_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_dc_128_predictor_8x8 = vpx_dc_128_predictor_8x8_sse2;
    vpx_dc_left_predictor_16x16 = vpx_dc_left_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_dc_left_predictor_16x16 = vpx_dc_left_predictor_16x16_sse2;
    vpx_dc_left_predictor_32x32 = vpx_dc_left_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_dc_left_predictor_32x32 = vpx_dc_left_predictor_32x32_sse2;
    vpx_dc_left_predictor_4x4 = vpx_dc_left_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_dc_left_predictor_4x4 = vpx_dc_left_predictor_4x4_sse2;
    vpx_dc_left_predictor_8x8 = vpx_dc_left_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_dc_left_predictor_8x8 = vpx_dc_left_predictor_8x8_sse2;
    vpx_dc_predictor_16x16 = vpx_dc_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_dc_predictor_16x16 = vpx_dc_predictor_16x16_sse2;
    vpx_dc_predictor_32x32 = vpx_dc_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_dc_predictor_32x32 = vpx_dc_predictor_32x32_sse2;
    vpx_dc_predictor_4x4 = vpx_dc_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_dc_predictor_4x4 = vpx_dc_predictor_4x4_sse2;
    vpx_dc_predictor_8x8 = vpx_dc_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_dc_predictor_8x8 = vpx_dc_predictor_8x8_sse2;
    vpx_dc_top_predictor_16x16 = vpx_dc_top_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_dc_top_predictor_16x16 = vpx_dc_top_predictor_16x16_sse2;
    vpx_dc_top_predictor_32x32 = vpx_dc_top_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_dc_top_predictor_32x32 = vpx_dc_top_predictor_32x32_sse2;
    vpx_dc_top_predictor_4x4 = vpx_dc_top_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_dc_top_predictor_4x4 = vpx_dc_top_predictor_4x4_sse2;
    vpx_dc_top_predictor_8x8 = vpx_dc_top_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_dc_top_predictor_8x8 = vpx_dc_top_predictor_8x8_sse2;
    vpx_h_predictor_16x16 = vpx_h_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_h_predictor_16x16 = vpx_h_predictor_16x16_sse2;
    vpx_h_predictor_32x32 = vpx_h_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_h_predictor_32x32 = vpx_h_predictor_32x32_sse2;
    vpx_h_predictor_4x4 = vpx_h_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_h_predictor_4x4 = vpx_h_predictor_4x4_sse2;
    vpx_h_predictor_8x8 = vpx_h_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_h_predictor_8x8 = vpx_h_predictor_8x8_sse2;
    vpx_idct16x16_10_add = vpx_idct16x16_10_add_c;
    if (flags & HAS_SSE2) vpx_idct16x16_10_add = vpx_idct16x16_10_add_sse2;
    vpx_idct16x16_1_add = vpx_idct16x16_1_add_c;
    if (flags & HAS_SSE2) vpx_idct16x16_1_add = vpx_idct16x16_1_add_sse2;
    vpx_idct16x16_256_add = vpx_idct16x16_256_add_c;
    if (flags & HAS_SSE2) vpx_idct16x16_256_add = vpx_idct16x16_256_add_sse2;
    vpx_idct32x32_1024_add = vpx_idct32x32_1024_add_c;
    if (flags & HAS_SSE2) vpx_idct32x32_1024_add = vpx_idct32x32_1024_add_sse2;
    vpx_idct32x32_135_add = vpx_idct32x32_135_add_c;
    if (flags & HAS_SSE2) vpx_idct32x32_135_add = vpx_idct32x32_1024_add_sse2;
    vpx_idct32x32_1_add = vpx_idct32x32_1_add_c;
    if (flags & HAS_SSE2) vpx_idct32x32_1_add = vpx_idct32x32_1_add_sse2;
    vpx_idct32x32_34_add = vpx_idct32x32_34_add_c;
    if (flags & HAS_SSE2) vpx_idct32x32_34_add = vpx_idct32x32_34_add_sse2;
    vpx_idct4x4_16_add = vpx_idct4x4_16_add_c;
    if (flags & HAS_SSE2) vpx_idct4x4_16_add = vpx_idct4x4_16_add_sse2;
    vpx_idct4x4_1_add = vpx_idct4x4_1_add_c;
    if (flags & HAS_SSE2) vpx_idct4x4_1_add = vpx_idct4x4_1_add_sse2;
    vpx_idct8x8_12_add = vpx_idct8x8_12_add_c;
    if (flags & HAS_SSE2) vpx_idct8x8_12_add = vpx_idct8x8_12_add_sse2;
    vpx_idct8x8_1_add = vpx_idct8x8_1_add_c;
    if (flags & HAS_SSE2) vpx_idct8x8_1_add = vpx_idct8x8_1_add_sse2;
    vpx_idct8x8_64_add = vpx_idct8x8_64_add_c;
    if (flags & HAS_SSE2) vpx_idct8x8_64_add = vpx_idct8x8_64_add_sse2;
    vpx_iwht4x4_16_add = vpx_iwht4x4_16_add_c;
    if (flags & HAS_SSE2) vpx_iwht4x4_16_add = vpx_iwht4x4_16_add_sse2;
    vpx_lpf_horizontal_4 = vpx_lpf_horizontal_4_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_4 = vpx_lpf_horizontal_4_sse2;
    vpx_lpf_horizontal_4_dual = vpx_lpf_horizontal_4_dual_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_4_dual = vpx_lpf_horizontal_4_dual_sse2;
    vpx_lpf_horizontal_8 = vpx_lpf_horizontal_8_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_8 = vpx_lpf_horizontal_8_sse2;
    vpx_lpf_horizontal_8_dual = vpx_lpf_horizontal_8_dual_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_8_dual = vpx_lpf_horizontal_8_dual_sse2;
    vpx_lpf_horizontal_edge_16 = vpx_lpf_horizontal_edge_16_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_edge_16 = vpx_lpf_horizontal_edge_16_sse2;
    vpx_lpf_horizontal_edge_8 = vpx_lpf_horizontal_edge_8_c;
    if (flags & HAS_SSE2) vpx_lpf_horizontal_edge_8 = vpx_lpf_horizontal_edge_8_sse2;
    vpx_lpf_vertical_16 = vpx_lpf_vertical_16_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_16 = vpx_lpf_vertical_16_sse2;
    vpx_lpf_vertical_16_dual = vpx_lpf_vertical_16_dual_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_16_dual = vpx_lpf_vertical_16_dual_sse2;
    vpx_lpf_vertical_4 = vpx_lpf_vertical_4_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_4 = vpx_lpf_vertical_4_sse2;
    vpx_lpf_vertical_4_dual = vpx_lpf_vertical_4_dual_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_4_dual = vpx_lpf_vertical_4_dual_sse2;
    vpx_lpf_vertical_8 = vpx_lpf_vertical_8_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_8 = vpx_lpf_vertical_8_sse2;
    vpx_lpf_vertical_8_dual = vpx_lpf_vertical_8_dual_c;
    if (flags & HAS_SSE2) vpx_lpf_vertical_8_dual = vpx_lpf_vertical_8_dual_sse2;
    vpx_scaled_2d = vpx_scaled_2d_c;
    if (flags & HAS_SSSE3) vpx_scaled_2d = vpx_scaled_2d_ssse3;
    vpx_tm_predictor_16x16 = vpx_tm_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_tm_predictor_16x16 = vpx_tm_predictor_16x16_sse2;
    vpx_tm_predictor_32x32 = vpx_tm_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_tm_predictor_32x32 = vpx_tm_predictor_32x32_sse2;
    vpx_tm_predictor_4x4 = vpx_tm_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_tm_predictor_4x4 = vpx_tm_predictor_4x4_sse2;
    vpx_tm_predictor_8x8 = vpx_tm_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_tm_predictor_8x8 = vpx_tm_predictor_8x8_sse2;
    vpx_v_predictor_16x16 = vpx_v_predictor_16x16_c;
    if (flags & HAS_SSE2) vpx_v_predictor_16x16 = vpx_v_predictor_16x16_sse2;
    vpx_v_predictor_32x32 = vpx_v_predictor_32x32_c;
    if (flags & HAS_SSE2) vpx_v_predictor_32x32 = vpx_v_predictor_32x32_sse2;
    vpx_v_predictor_4x4 = vpx_v_predictor_4x4_c;
    if (flags & HAS_SSE2) vpx_v_predictor_4x4 = vpx_v_predictor_4x4_sse2;
    vpx_v_predictor_8x8 = vpx_v_predictor_8x8_c;
    if (flags & HAS_SSE2) vpx_v_predictor_8x8 = vpx_v_predictor_8x8_sse2;
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
