// Copyright 2016 Adrien Descamps
// Distributed under BSD 3-Clause License
#include "SDL_internal.h"

#ifdef SDL_HAVE_YUV

#include "yuv_rgb_internal.h"

// divide by PRECISION_FACTOR and clamp to [0:255] interval
// input must be in the [-128*PRECISION_FACTOR:384*PRECISION_FACTOR] range
static uint8_t clampU8(int32_t v)
{
    static const uint8_t lut[512] =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,
             47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
             91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,
             126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,
             159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
             192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,
             225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
             255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
             255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
             255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
             255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
            };
    return lut[((v+128*PRECISION_FACTOR)>>PRECISION)&511];
}

static uint16_t clamp10(int32_t v)
{
    v >>= PRECISION;
    if (v < 0) {
        return 0;
    } else if (v > 1023) {
        return 1023;
    } else {
        return (uint16_t)v;
    }
}

#define YUV_BITS    8

#define STD_FUNCTION_NAME	yuv420_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv420_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv420_rgba_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv420_bgra_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv420_argb_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv420_abgr_std
#define YUV_FORMAT			YUV_FORMAT_420
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_rgba_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_bgra_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_argb_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuv422_abgr_std
#define YUV_FORMAT			YUV_FORMAT_422
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_rgb565_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB565
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_rgb24_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGB24
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_rgba_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_RGBA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_bgra_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_BGRA
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_argb_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ARGB
#include "yuv_rgb_std_func.h"

#define STD_FUNCTION_NAME	yuvnv12_abgr_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_ABGR
#include "yuv_rgb_std_func.h"

#undef YUV_BITS
#define YUV_BITS    10

#define STD_FUNCTION_NAME	yuvp010_xbgr2101010_std
#define YUV_FORMAT			YUV_FORMAT_NV12
#define RGB_FORMAT			RGB_FORMAT_XBGR2101010
#include "yuv_rgb_std_func.h"

void rgb24_yuv420_std(
        uint32_t width, uint32_t height,
        const uint8_t *RGB, uint32_t RGB_stride,
        uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
        YCbCrType yuv_type)
{
    const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);

    uint32_t x, y;
    for(y=0; y<(height-1); y+=2)
    {
        const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
                *rgb_ptr2=RGB+(y+1)*RGB_stride;

        uint8_t *y_ptr1=Y+y*Y_stride,
                *y_ptr2=Y+(y+1)*Y_stride,
                *u_ptr=U+(y/2)*UV_stride,
                *v_ptr=V+(y/2)*UV_stride;

        for(x=0; x<(width-1); x+=2)
        {
            // compute yuv for the four pixels, u and v values are summed
            int32_t y_tmp, u_tmp, v_tmp;

            y_tmp = param->matrix[0][0]*rgb_ptr1[0] + param->matrix[0][1]*rgb_ptr1[1] + param->matrix[0][2]*rgb_ptr1[2];
            u_tmp = param->matrix[1][0]*rgb_ptr1[0] + param->matrix[1][1]*rgb_ptr1[1] + param->matrix[1][2]*rgb_ptr1[2];
            v_tmp = param->matrix[2][0]*rgb_ptr1[0] + param->matrix[2][1]*rgb_ptr1[1] + param->matrix[2][2]*rgb_ptr1[2];
            y_ptr1[0]=clampU8(y_tmp+((param->y_shift)<<PRECISION));

            y_tmp = param->matrix[0][0]*rgb_ptr1[3] + param->matrix[0][1]*rgb_ptr1[4] + param->matrix[0][2]*rgb_ptr1[5];
            u_tmp += param->matrix[1][0]*rgb_ptr1[3] + param->matrix[1][1]*rgb_ptr1[4] + param->matrix[1][2]*rgb_ptr1[5];
            v_tmp += param->matrix[2][0]*rgb_ptr1[3] + param->matrix[2][1]*rgb_ptr1[4] + param->matrix[2][2]*rgb_ptr1[5];
            y_ptr1[1]=clampU8(y_tmp+((param->y_shift)<<PRECISION));

            y_tmp = param->matrix[0][0]*rgb_ptr2[0] + param->matrix[0][1]*rgb_ptr2[1] + param->matrix[0][2]*rgb_ptr2[2];
            u_tmp += param->matrix[1][0]*rgb_ptr2[0] + param->matrix[1][1]*rgb_ptr2[1] + param->matrix[1][2]*rgb_ptr2[2];
            v_tmp += param->matrix[2][0]*rgb_ptr2[0] + param->matrix[2][1]*rgb_ptr2[1] + param->matrix[2][2]*rgb_ptr2[2];
            y_ptr2[0]=clampU8(y_tmp+((param->y_shift)<<PRECISION));

            y_tmp = param->matrix[0][0]*rgb_ptr2[3] + param->matrix[0][1]*rgb_ptr2[4] + param->matrix[0][2]*rgb_ptr2[5];
            u_tmp += param->matrix[1][0]*rgb_ptr2[3] + param->matrix[1][1]*rgb_ptr2[4] + param->matrix[1][2]*rgb_ptr2[5];
            v_tmp += param->matrix[2][0]*rgb_ptr2[3] + param->matrix[2][1]*rgb_ptr2[4] + param->matrix[2][2]*rgb_ptr2[5];
            y_ptr2[1]=clampU8(y_tmp+((param->y_shift)<<PRECISION));

            u_ptr[0] = clampU8(u_tmp/4+(128<<PRECISION));
            v_ptr[0] = clampU8(v_tmp/4+(128<<PRECISION));

            rgb_ptr1 += 6;
            rgb_ptr2 += 6;
            y_ptr1 += 2;
            y_ptr2 += 2;
            u_ptr += 1;
            v_ptr += 1;
        }
    }
}

#endif /* SDL_HAVE_YUV */
