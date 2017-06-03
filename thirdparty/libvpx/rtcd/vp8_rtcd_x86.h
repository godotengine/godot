#ifndef VP8_RTCD_H_
#define VP8_RTCD_H_

#ifdef RTCD_C
#define RTCD_EXTERN
#else
#define RTCD_EXTERN extern
#endif

/*
 * VP8
 */

struct blockd;
struct loop_filter_info;

#ifdef __cplusplus
extern "C" {
#endif

void vp8_bilinear_predict16x16_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict16x16_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict16x16_sse2(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict16x16_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_bilinear_predict16x16)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_bilinear_predict4x4_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict4x4_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_bilinear_predict4x4)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_bilinear_predict8x4_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict8x4_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_bilinear_predict8x4)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_bilinear_predict8x8_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict8x8_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict8x8_sse2(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_bilinear_predict8x8_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_bilinear_predict8x8)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_clear_system_state_c();
void vpx_reset_mmx_state();
RTCD_EXTERN void (*vp8_clear_system_state)();

void vp8_copy_mem16x16_c(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
void vp8_copy_mem16x16_mmx(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
void vp8_copy_mem16x16_sse2(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_copy_mem16x16)(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);

void vp8_copy_mem8x4_c(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
void vp8_copy_mem8x4_mmx(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_copy_mem8x4)(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);

void vp8_copy_mem8x8_c(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
void vp8_copy_mem8x8_mmx(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_copy_mem8x8)(unsigned char *src, int src_pitch, unsigned char *dst, int dst_pitch);

void vp8_dc_only_idct_add_c(short input, unsigned char *pred, int pred_stride, unsigned char *dst, int dst_stride);
void vp8_dc_only_idct_add_mmx(short input, unsigned char *pred, int pred_stride, unsigned char *dst, int dst_stride);
RTCD_EXTERN void (*vp8_dc_only_idct_add)(short input, unsigned char *pred, int pred_stride, unsigned char *dst, int dst_stride);

void vp8_dequant_idct_add_c(short *input, short *dq, unsigned char *output, int stride);
void vp8_dequant_idct_add_mmx(short *input, short *dq, unsigned char *output, int stride);
RTCD_EXTERN void (*vp8_dequant_idct_add)(short *input, short *dq, unsigned char *output, int stride);

void vp8_dequant_idct_add_uv_block_c(short *q, short *dq, unsigned char *dst_u, unsigned char *dst_v, int stride, char *eobs);
void vp8_dequant_idct_add_uv_block_mmx(short *q, short *dq, unsigned char *dst_u, unsigned char *dst_v, int stride, char *eobs);
void vp8_dequant_idct_add_uv_block_sse2(short *q, short *dq, unsigned char *dst_u, unsigned char *dst_v, int stride, char *eobs);
RTCD_EXTERN void (*vp8_dequant_idct_add_uv_block)(short *q, short *dq, unsigned char *dst_u, unsigned char *dst_v, int stride, char *eobs);

void vp8_dequant_idct_add_y_block_c(short *q, short *dq, unsigned char *dst, int stride, char *eobs);
void vp8_dequant_idct_add_y_block_mmx(short *q, short *dq, unsigned char *dst, int stride, char *eobs);
void vp8_dequant_idct_add_y_block_sse2(short *q, short *dq, unsigned char *dst, int stride, char *eobs);
RTCD_EXTERN void (*vp8_dequant_idct_add_y_block)(short *q, short *dq, unsigned char *dst, int stride, char *eobs);

void vp8_dequantize_b_c(struct blockd*, short *dqc);
void vp8_dequantize_b_mmx(struct blockd*, short *dqc);
RTCD_EXTERN void (*vp8_dequantize_b)(struct blockd*, short *dqc);

void vp8_loop_filter_bh_c(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_bh_mmx(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_bh_sse2(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
RTCD_EXTERN void (*vp8_loop_filter_bh)(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);

void vp8_loop_filter_bv_c(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_bv_mmx(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_bv_sse2(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
RTCD_EXTERN void (*vp8_loop_filter_bv)(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);

void vp8_loop_filter_mbh_c(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_mbh_mmx(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_mbh_sse2(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
RTCD_EXTERN void (*vp8_loop_filter_mbh)(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);

void vp8_loop_filter_mbv_c(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_mbv_mmx(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
void vp8_loop_filter_mbv_sse2(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);
RTCD_EXTERN void (*vp8_loop_filter_mbv)(unsigned char *y, unsigned char *u, unsigned char *v, int ystride, int uv_stride, struct loop_filter_info *lfi);

void vp8_loop_filter_bhs_c(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_bhs_mmx(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_bhs_sse2(unsigned char *y, int ystride, const unsigned char *blimit);
RTCD_EXTERN void (*vp8_loop_filter_simple_bh)(unsigned char *y, int ystride, const unsigned char *blimit);

void vp8_loop_filter_bvs_c(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_bvs_mmx(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_bvs_sse2(unsigned char *y, int ystride, const unsigned char *blimit);
RTCD_EXTERN void (*vp8_loop_filter_simple_bv)(unsigned char *y, int ystride, const unsigned char *blimit);

void vp8_loop_filter_simple_horizontal_edge_c(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_simple_horizontal_edge_mmx(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_simple_horizontal_edge_sse2(unsigned char *y, int ystride, const unsigned char *blimit);
RTCD_EXTERN void (*vp8_loop_filter_simple_mbh)(unsigned char *y, int ystride, const unsigned char *blimit);

void vp8_loop_filter_simple_vertical_edge_c(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_simple_vertical_edge_mmx(unsigned char *y, int ystride, const unsigned char *blimit);
void vp8_loop_filter_simple_vertical_edge_sse2(unsigned char *y, int ystride, const unsigned char *blimit);
RTCD_EXTERN void (*vp8_loop_filter_simple_mbv)(unsigned char *y, int ystride, const unsigned char *blimit);

void vp8_short_idct4x4llm_c(short *input, unsigned char *pred, int pitch, unsigned char *dst, int dst_stride);
void vp8_short_idct4x4llm_mmx(short *input, unsigned char *pred, int pitch, unsigned char *dst, int dst_stride);
RTCD_EXTERN void (*vp8_short_idct4x4llm)(short *input, unsigned char *pred, int pitch, unsigned char *dst, int dst_stride);

void vp8_short_inv_walsh4x4_c(short *input, short *output);
void vp8_short_inv_walsh4x4_mmx(short *input, short *output);
void vp8_short_inv_walsh4x4_sse2(short *input, short *output);
RTCD_EXTERN void (*vp8_short_inv_walsh4x4)(short *input, short *output);

void vp8_short_inv_walsh4x4_1_c(short *input, short *output);
#define vp8_short_inv_walsh4x4_1 vp8_short_inv_walsh4x4_1_c

void vp8_sixtap_predict16x16_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict16x16_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict16x16_sse2(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict16x16_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_sixtap_predict16x16)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_sixtap_predict4x4_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict4x4_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict4x4_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_sixtap_predict4x4)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_sixtap_predict8x4_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x4_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x4_sse2(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x4_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_sixtap_predict8x4)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_sixtap_predict8x8_c(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x8_mmx(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x8_sse2(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
void vp8_sixtap_predict8x8_ssse3(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);
RTCD_EXTERN void (*vp8_sixtap_predict8x8)(unsigned char *src, int src_pitch, int xofst, int yofst, unsigned char *dst, int dst_pitch);

void vp8_rtcd(void);

#ifdef RTCD_C
#include "vpx_ports/x86.h"
static void setup_rtcd_internal(void)
{
    int flags = x86_simd_caps();

    vp8_bilinear_predict16x16 = vp8_bilinear_predict16x16_c;
    if (flags & HAS_MMX) vp8_bilinear_predict16x16 = vp8_bilinear_predict16x16_mmx;
    if (flags & HAS_SSE2) vp8_bilinear_predict16x16 = vp8_bilinear_predict16x16_sse2;
    if (flags & HAS_SSSE3) vp8_bilinear_predict16x16 = vp8_bilinear_predict16x16_ssse3;
    vp8_bilinear_predict4x4 = vp8_bilinear_predict4x4_c;
    if (flags & HAS_MMX) vp8_bilinear_predict4x4 = vp8_bilinear_predict4x4_mmx;
    vp8_bilinear_predict8x4 = vp8_bilinear_predict8x4_c;
    if (flags & HAS_MMX) vp8_bilinear_predict8x4 = vp8_bilinear_predict8x4_mmx;
    vp8_bilinear_predict8x8 = vp8_bilinear_predict8x8_c;
    if (flags & HAS_MMX) vp8_bilinear_predict8x8 = vp8_bilinear_predict8x8_mmx;
    if (flags & HAS_SSE2) vp8_bilinear_predict8x8 = vp8_bilinear_predict8x8_sse2;
    if (flags & HAS_SSSE3) vp8_bilinear_predict8x8 = vp8_bilinear_predict8x8_ssse3;
    vp8_clear_system_state = vp8_clear_system_state_c;
    if (flags & HAS_MMX) vp8_clear_system_state = vpx_reset_mmx_state;
    vp8_copy_mem16x16 = vp8_copy_mem16x16_c;
    if (flags & HAS_MMX) vp8_copy_mem16x16 = vp8_copy_mem16x16_mmx;
    if (flags & HAS_SSE2) vp8_copy_mem16x16 = vp8_copy_mem16x16_sse2;
    vp8_copy_mem8x4 = vp8_copy_mem8x4_c;
    if (flags & HAS_MMX) vp8_copy_mem8x4 = vp8_copy_mem8x4_mmx;
    vp8_copy_mem8x8 = vp8_copy_mem8x8_c;
    if (flags & HAS_MMX) vp8_copy_mem8x8 = vp8_copy_mem8x8_mmx;
    vp8_dc_only_idct_add = vp8_dc_only_idct_add_c;
    if (flags & HAS_MMX) vp8_dc_only_idct_add = vp8_dc_only_idct_add_mmx;
    vp8_dequant_idct_add = vp8_dequant_idct_add_c;
    if (flags & HAS_MMX) vp8_dequant_idct_add = vp8_dequant_idct_add_mmx;
    vp8_dequant_idct_add_uv_block = vp8_dequant_idct_add_uv_block_c;
    if (flags & HAS_MMX) vp8_dequant_idct_add_uv_block = vp8_dequant_idct_add_uv_block_mmx;
    if (flags & HAS_SSE2) vp8_dequant_idct_add_uv_block = vp8_dequant_idct_add_uv_block_sse2;
    vp8_dequant_idct_add_y_block = vp8_dequant_idct_add_y_block_c;
    if (flags & HAS_MMX) vp8_dequant_idct_add_y_block = vp8_dequant_idct_add_y_block_mmx;
    if (flags & HAS_SSE2) vp8_dequant_idct_add_y_block = vp8_dequant_idct_add_y_block_sse2;
    vp8_dequantize_b = vp8_dequantize_b_c;
    if (flags & HAS_MMX) vp8_dequantize_b = vp8_dequantize_b_mmx;
    vp8_loop_filter_bh = vp8_loop_filter_bh_c;
    if (flags & HAS_MMX) vp8_loop_filter_bh = vp8_loop_filter_bh_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_bh = vp8_loop_filter_bh_sse2;
    vp8_loop_filter_bv = vp8_loop_filter_bv_c;
    if (flags & HAS_MMX) vp8_loop_filter_bv = vp8_loop_filter_bv_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_bv = vp8_loop_filter_bv_sse2;
    vp8_loop_filter_mbh = vp8_loop_filter_mbh_c;
    if (flags & HAS_MMX) vp8_loop_filter_mbh = vp8_loop_filter_mbh_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_mbh = vp8_loop_filter_mbh_sse2;
    vp8_loop_filter_mbv = vp8_loop_filter_mbv_c;
    if (flags & HAS_MMX) vp8_loop_filter_mbv = vp8_loop_filter_mbv_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_mbv = vp8_loop_filter_mbv_sse2;
    vp8_loop_filter_simple_bh = vp8_loop_filter_bhs_c;
    if (flags & HAS_MMX) vp8_loop_filter_simple_bh = vp8_loop_filter_bhs_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_simple_bh = vp8_loop_filter_bhs_sse2;
    vp8_loop_filter_simple_bv = vp8_loop_filter_bvs_c;
    if (flags & HAS_MMX) vp8_loop_filter_simple_bv = vp8_loop_filter_bvs_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_simple_bv = vp8_loop_filter_bvs_sse2;
    vp8_loop_filter_simple_mbh = vp8_loop_filter_simple_horizontal_edge_c;
    if (flags & HAS_MMX) vp8_loop_filter_simple_mbh = vp8_loop_filter_simple_horizontal_edge_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_simple_mbh = vp8_loop_filter_simple_horizontal_edge_sse2;
    vp8_loop_filter_simple_mbv = vp8_loop_filter_simple_vertical_edge_c;
    if (flags & HAS_MMX) vp8_loop_filter_simple_mbv = vp8_loop_filter_simple_vertical_edge_mmx;
    if (flags & HAS_SSE2) vp8_loop_filter_simple_mbv = vp8_loop_filter_simple_vertical_edge_sse2;
    vp8_short_idct4x4llm = vp8_short_idct4x4llm_c;
    if (flags & HAS_MMX) vp8_short_idct4x4llm = vp8_short_idct4x4llm_mmx;
    vp8_short_inv_walsh4x4 = vp8_short_inv_walsh4x4_c;
    if (flags & HAS_MMX) vp8_short_inv_walsh4x4 = vp8_short_inv_walsh4x4_mmx;
    if (flags & HAS_SSE2) vp8_short_inv_walsh4x4 = vp8_short_inv_walsh4x4_sse2;
    vp8_sixtap_predict16x16 = vp8_sixtap_predict16x16_c;
    if (flags & HAS_MMX) vp8_sixtap_predict16x16 = vp8_sixtap_predict16x16_mmx;
    if (flags & HAS_SSE2) vp8_sixtap_predict16x16 = vp8_sixtap_predict16x16_sse2;
    if (flags & HAS_SSSE3) vp8_sixtap_predict16x16 = vp8_sixtap_predict16x16_ssse3;
    vp8_sixtap_predict4x4 = vp8_sixtap_predict4x4_c;
    if (flags & HAS_MMX) vp8_sixtap_predict4x4 = vp8_sixtap_predict4x4_mmx;
    if (flags & HAS_SSSE3) vp8_sixtap_predict4x4 = vp8_sixtap_predict4x4_ssse3;
    vp8_sixtap_predict8x4 = vp8_sixtap_predict8x4_c;
    if (flags & HAS_MMX) vp8_sixtap_predict8x4 = vp8_sixtap_predict8x4_mmx;
    if (flags & HAS_SSE2) vp8_sixtap_predict8x4 = vp8_sixtap_predict8x4_sse2;
    if (flags & HAS_SSSE3) vp8_sixtap_predict8x4 = vp8_sixtap_predict8x4_ssse3;
    vp8_sixtap_predict8x8 = vp8_sixtap_predict8x8_c;
    if (flags & HAS_MMX) vp8_sixtap_predict8x8 = vp8_sixtap_predict8x8_mmx;
    if (flags & HAS_SSE2) vp8_sixtap_predict8x8 = vp8_sixtap_predict8x8_sse2;
    if (flags & HAS_SSSE3) vp8_sixtap_predict8x8 = vp8_sixtap_predict8x8_ssse3;
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
