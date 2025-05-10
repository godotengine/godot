// Copyright 2016 Adrien Descamps
// Distributed under BSD 3-Clause License
#include "SDL_internal.h"

#ifdef SDL_HAVE_YUV
#include "yuv_rgb_internal.h"

#ifdef SDL_SSE2_INTRINSICS

/* SDL doesn't use these atm and compiling them adds seconds onto the build.  --ryan.
#define SSE_FUNCTION_NAME	yuv420_rgb565_sse
#define STD_FUNCTION_NAME	yuv420_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB565
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_rgb24_sse
#define STD_FUNCTION_NAME	yuv420_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB24
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_rgba_sse
#define STD_FUNCTION_NAME	yuv420_rgba_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGBA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_bgra_sse
#define STD_FUNCTION_NAME	yuv420_bgra_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_BGRA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_argb_sse
#define STD_FUNCTION_NAME	yuv420_argb_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ARGB
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_abgr_sse
#define STD_FUNCTION_NAME	yuv420_abgr_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ABGR
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgb565_sse
#define STD_FUNCTION_NAME	yuv422_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB565
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgb24_sse
#define STD_FUNCTION_NAME	yuv422_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB24
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgba_sse
#define STD_FUNCTION_NAME	yuv422_rgba_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGBA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_bgra_sse
#define STD_FUNCTION_NAME	yuv422_bgra_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_BGRA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_argb_sse
#define STD_FUNCTION_NAME	yuv422_argb_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ARGB
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_abgr_sse
#define STD_FUNCTION_NAME	yuv422_abgr_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ABGR
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgb565_sse
#define STD_FUNCTION_NAME	yuvnv12_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB565
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgb24_sse
#define STD_FUNCTION_NAME	yuvnv12_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB24
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgba_sse
#define STD_FUNCTION_NAME	yuvnv12_rgba_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGBA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_bgra_sse
#define STD_FUNCTION_NAME	yuvnv12_bgra_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_BGRA
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_argb_sse
#define STD_FUNCTION_NAME	yuvnv12_argb_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ARGB
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_abgr_sse
#define STD_FUNCTION_NAME	yuvnv12_abgr_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ABGR
#define SSE_ALIGNED
#include "yuv_rgb_sse_func.h"
*/

#define SSE_FUNCTION_NAME	yuv420_rgb565_sseu
#define STD_FUNCTION_NAME	yuv420_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_rgb24_sseu
#define STD_FUNCTION_NAME	yuv420_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_rgba_sseu
#define STD_FUNCTION_NAME	yuv420_rgba_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_bgra_sseu
#define STD_FUNCTION_NAME	yuv420_bgra_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_argb_sseu
#define STD_FUNCTION_NAME	yuv420_argb_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv420_abgr_sseu
#define STD_FUNCTION_NAME	yuv420_abgr_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgb565_sseu
#define STD_FUNCTION_NAME	yuv422_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgb24_sseu
#define STD_FUNCTION_NAME	yuv422_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_rgba_sseu
#define STD_FUNCTION_NAME	yuv422_rgba_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_bgra_sseu
#define STD_FUNCTION_NAME	yuv422_bgra_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_argb_sseu
#define STD_FUNCTION_NAME	yuv422_argb_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuv422_abgr_sseu
#define STD_FUNCTION_NAME	yuv422_abgr_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgb565_sseu
#define STD_FUNCTION_NAME	yuvnv12_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgb24_sseu
#define STD_FUNCTION_NAME	yuvnv12_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_rgba_sseu
#define STD_FUNCTION_NAME	yuvnv12_rgba_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_bgra_sseu
#define STD_FUNCTION_NAME	yuvnv12_bgra_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_argb_sseu
#define STD_FUNCTION_NAME	yuvnv12_argb_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_sse_func.h"

#define SSE_FUNCTION_NAME	yuvnv12_abgr_sseu
#define STD_FUNCTION_NAME	yuvnv12_abgr_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_sse_func.h"


/* SDL doesn't use these atm and compiling them adds seconds onto the build.  --ryan.
#define UNPACK_RGB24_32_STEP1(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
R1 = _mm_unpacklo_epi8(RGB1, RGB4); \
R2 = _mm_unpackhi_epi8(RGB1, RGB4); \
G1 = _mm_unpacklo_epi8(RGB2, RGB5); \
G2 = _mm_unpackhi_epi8(RGB2, RGB5); \
B1 = _mm_unpacklo_epi8(RGB3, RGB6); \
B2 = _mm_unpackhi_epi8(RGB3, RGB6);

#define UNPACK_RGB24_32_STEP2(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
RGB1 = _mm_unpacklo_epi8(R1, G2); \
RGB2 = _mm_unpackhi_epi8(R1, G2); \
RGB3 = _mm_unpacklo_epi8(R2, B1); \
RGB4 = _mm_unpackhi_epi8(R2, B1); \
RGB5 = _mm_unpacklo_epi8(G1, B2); \
RGB6 = _mm_unpackhi_epi8(G1, B2); \

#define UNPACK_RGB24_32(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
UNPACK_RGB24_32_STEP1(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
UNPACK_RGB24_32_STEP2(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
UNPACK_RGB24_32_STEP1(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
UNPACK_RGB24_32_STEP2(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
UNPACK_RGB24_32_STEP1(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \

#define RGB2YUV_16(R, G, B, Y, U, V) \
Y = _mm_add_epi16(_mm_mullo_epi16(R, _mm_set1_epi16(param->matrix[0][0])), \
		_mm_mullo_epi16(G, _mm_set1_epi16(param->matrix[0][1]))); \
Y = _mm_add_epi16(Y, _mm_mullo_epi16(B, _mm_set1_epi16(param->matrix[0][2]))); \
Y = _mm_add_epi16(Y, _mm_set1_epi16((param->y_shift)<<PRECISION)); \
Y = _mm_srai_epi16(Y, PRECISION); \
U = _mm_add_epi16(_mm_mullo_epi16(R, _mm_set1_epi16(param->matrix[1][0])), \
		_mm_mullo_epi16(G, _mm_set1_epi16(param->matrix[1][1]))); \
U = _mm_add_epi16(U, _mm_mullo_epi16(B, _mm_set1_epi16(param->matrix[1][2]))); \
U = _mm_add_epi16(U, _mm_set1_epi16(128<<PRECISION)); \
U = _mm_srai_epi16(U, PRECISION); \
V = _mm_add_epi16(_mm_mullo_epi16(R, _mm_set1_epi16(param->matrix[2][0])), \
		_mm_mullo_epi16(G, _mm_set1_epi16(param->matrix[2][1]))); \
V = _mm_add_epi16(V, _mm_mullo_epi16(B, _mm_set1_epi16(param->matrix[2][2]))); \
V = _mm_add_epi16(V, _mm_set1_epi16(128<<PRECISION)); \
V = _mm_srai_epi16(V, PRECISION);
*/

#if 0  // SDL doesn't use these atm and compiling them adds seconds onto the build.  --ryan.
#define RGB2YUV_32 \
	__m128i r1, r2, b1, b2, g1, g2; \
	__m128i r_16, g_16, b_16; \
	__m128i y1_16, y2_16, u1_16, u2_16, v1_16, v2_16, y, u1, u2, v1, v2, u1_tmp, u2_tmp, v1_tmp, v2_tmp; \
	__m128i rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+16)), \
		rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+32)), \
		rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr2)), \
		rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2+16)), \
		rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+32)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	UNPACK_RGB24_32(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, r1, r2, g1, g2, b1, b2) \
	/* process pixels of first line */ \
	r_16 = _mm_unpacklo_epi8(r1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(g1, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(b1, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y1_16, u1_16, v1_16) \
	r_16 = _mm_unpackhi_epi8(r1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(g1, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(b1, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y2_16, u2_16, v2_16) \
	y = _mm_packus_epi16(y1_16, y2_16); \
	u1 = _mm_packus_epi16(u1_16, u2_16); \
	v1 = _mm_packus_epi16(v1_16, v2_16); \
	/* save Y values */ \
	SAVE_SI128((__m128i*)(y_ptr1), y); \
	/* process pixels of second line */ \
	r_16 = _mm_unpacklo_epi8(r2, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(g2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(b2, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y1_16, u1_16, v1_16) \
	r_16 = _mm_unpackhi_epi8(r2, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(g2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(b2, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y2_16, u2_16, v2_16) \
	y = _mm_packus_epi16(y1_16, y2_16); \
	u2 = _mm_packus_epi16(u1_16, u2_16); \
	v2 = _mm_packus_epi16(v1_16, v2_16); \
	/* save Y values */ \
	SAVE_SI128((__m128i*)(y_ptr2), y); \
	/* vertical subsampling of u/v values */ \
	u1_tmp = _mm_avg_epu8(u1, u2); \
	v1_tmp = _mm_avg_epu8(v1, v2); \
	/* do the same again with next data */ \
	rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1+48)); \
	rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+64)); \
	rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+80)); \
	rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr2+48)); \
	rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2+64)); \
	rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+80)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	UNPACK_RGB24_32(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, r1, r2, g1, g2, b1, b2) \
	/* process pixels of first line */ \
	r_16 = _mm_unpacklo_epi8(r1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(g1, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(b1, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y1_16, u1_16, v1_16) \
	r_16 = _mm_unpackhi_epi8(r1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(g1, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(b1, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y2_16, u2_16, v2_16) \
	y = _mm_packus_epi16(y1_16, y2_16); \
	u1 = _mm_packus_epi16(u1_16, u2_16); \
	v1 = _mm_packus_epi16(v1_16, v2_16); \
	/* save Y values */ \
	SAVE_SI128((__m128i*)(y_ptr1+16), y); \
	/* process pixels of second line */ \
	r_16 = _mm_unpacklo_epi8(r2, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(g2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(b2, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y1_16, u1_16, v1_16) \
	r_16 = _mm_unpackhi_epi8(r2, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(g2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(b2, _mm_setzero_si128()); \
	RGB2YUV_16(r_16, g_16, b_16, y2_16, u2_16, v2_16) \
	y = _mm_packus_epi16(y1_16, y2_16); \
	u2 = _mm_packus_epi16(u1_16, u2_16); \
	v2 = _mm_packus_epi16(v1_16, v2_16); \
	/* save Y values */ \
	SAVE_SI128((__m128i*)(y_ptr2+16), y); \
	/* vertical subsampling of u/v values */ \
	u2_tmp = _mm_avg_epu8(u1, u2); \
	v2_tmp = _mm_avg_epu8(v1, v2); \
	/* horizontal subsampling of u/v values */ \
	u1 = _mm_packus_epi16(_mm_srl_epi16(u1_tmp, _mm_cvtsi32_si128(8)), _mm_srl_epi16(u2_tmp, _mm_cvtsi32_si128(8))); \
	v1 = _mm_packus_epi16(_mm_srl_epi16(v1_tmp, _mm_cvtsi32_si128(8)), _mm_srl_epi16(v2_tmp, _mm_cvtsi32_si128(8))); \
	u2 = _mm_packus_epi16(_mm_and_si128(u1_tmp, _mm_set1_epi16(0xFF)), _mm_and_si128(u2_tmp, _mm_set1_epi16(0xFF))); \
	v2 = _mm_packus_epi16(_mm_and_si128(v1_tmp, _mm_set1_epi16(0xFF)), _mm_and_si128(v2_tmp, _mm_set1_epi16(0xFF))); \
	u1 = _mm_avg_epu8(u1, u2); \
	v1 = _mm_avg_epu8(v1, v2); \
	SAVE_SI128((__m128i*)(u_ptr), u1); \
	SAVE_SI128((__m128i*)(v_ptr), v1);
#endif

/* SDL doesn't use these atm and compiling them adds seconds onto the build.  --ryan.
void SDL_TARGETING("sse2") rgb24_yuv420_sse(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);

	uint32_t xpos, ypos;
	for(ypos=0; ypos<(height-1); ypos+=2)
	{
		const uint8_t *rgb_ptr1=RGB+ypos*RGB_stride,
			*rgb_ptr2=RGB+(ypos+1)*RGB_stride;

		uint8_t *y_ptr1=Y+ypos*Y_stride,
			*y_ptr2=Y+(ypos+1)*Y_stride,
			*u_ptr=U+(ypos/2)*UV_stride,
			*v_ptr=V+(ypos/2)*UV_stride;

		for(xpos=0; xpos<(width-31); xpos+=32)
		{
			RGB2YUV_32

			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void SDL_TARGETING("sse2") rgb24_yuv420_sseu(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);

	uint32_t xpos, ypos;
	for(ypos=0; ypos<(height-1); ypos+=2)
	{
		const uint8_t *rgb_ptr1=RGB+ypos*RGB_stride,
			*rgb_ptr2=RGB+(ypos+1)*RGB_stride;

		uint8_t *y_ptr1=Y+ypos*Y_stride,
			*y_ptr2=Y+(ypos+1)*Y_stride,
			*u_ptr=U+(ypos/2)*UV_stride,
			*v_ptr=V+(ypos/2)*UV_stride;

		for(xpos=0; xpos<(width-31); xpos+=32)
		{
			RGB2YUV_32

			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}
*/

#endif // SDL_SSE2_INTRINSICS

#endif // SDL_HAVE_YUV
