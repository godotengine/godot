/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifdef _YUV_C
#include "yuv_util.h"

int YTable [256];
int BUTable[256];
int GUTable[256];
int GVTable[256];
int RVTable[256];

#define CLIP_RGB_COLOR(dst, x) \
	tmp = (x) >> 13;\
	if ((tmp & ~0xFF) == 0) dst = tmp;\
	else                    dst = (-tmp) >> 31;

#define _decodeRGB(t, stride, nBytes, maxWidth, i1, i2, i3, j1, j2, j3)\
	register int tmp;\
	int nBytes2 = nBytes * 2, cv, cu, rgbY1, rgbY2, rgbY3, rgbY4, rV, gUV, bU, width = maxWidth == 0 ? t->w : maxWidth;\
	unsigned int y;\
	unsigned char *ySrcEven, *ySrcOdd, *yLineEnd, *uSrc, *vSrc, *out1, *out2;\
	\
	for (y = 0; y < t->h; y += 2)\
	{\
		ySrcEven = t->y + y * t->yStride;\
		ySrcOdd  = t->y + (y + 1) * t->yStride;\
		uSrc = t->u + y * t->uStride / 2;\
		vSrc = t->v + y * t->vStride / 2;\
		out1 = t->out + y * stride;\
		out2 = t->out + (y + 1) * stride;\
		\
		for (yLineEnd = ySrcEven + width; ySrcEven != yLineEnd;)\
		{\
			cu = *uSrc; ++uSrc;\
			cv = *vSrc; ++vSrc;\
			rV   = RVTable[cv];\
			gUV  = GUTable[cu] + GVTable[cv];\
			bU   = BUTable[cu];\
			\
			rgbY1 = YTable[*ySrcEven]; ++ySrcEven;\
			rgbY2 = YTable[*ySrcOdd];  ++ySrcOdd;\
			rgbY3 = YTable[*ySrcEven]; ++ySrcEven;\
			rgbY4 = YTable[*ySrcOdd];  ++ySrcOdd;\
			\
			CLIP_RGB_COLOR(out1[i1], rgbY1 + rV );\
			CLIP_RGB_COLOR(out1[i2], rgbY1 - gUV);\
			CLIP_RGB_COLOR(out1[i3], rgbY1 + bU );\
			\
			CLIP_RGB_COLOR(out2[i1], rgbY2 + rV );\
			CLIP_RGB_COLOR(out2[i2], rgbY2 - gUV);\
			CLIP_RGB_COLOR(out2[i3], rgbY2 + bU );\
			\
			CLIP_RGB_COLOR(out1[j1], rgbY3 + rV );\
			CLIP_RGB_COLOR(out1[j2], rgbY3 - gUV);\
			CLIP_RGB_COLOR(out1[j3], rgbY3 + bU );\
			\
			CLIP_RGB_COLOR(out2[j1], rgbY4 + rV );\
			CLIP_RGB_COLOR(out2[j2], rgbY4 - gUV);\
			CLIP_RGB_COLOR(out2[j3], rgbY4 + bU );\
			\
			out1 += nBytes2;  out2 += nBytes2;\
		}\
	}

// The 'trick' with this function is that it skips decoding YUV pixels if the alpha value is 0, thus improving the decoding speed of a frame
#define _decodeRGBA(t, stride, nBytes, maxWidth, i1, i2, i3, j1, j2, j3, aindex1, aindex2)\
\
	register int tmp;\
	int nBytes2 = nBytes * 2, cv, cu, rgbY1, rgbY2, rgbY3, rgbY4, a1, a2, a3, a4, rV, gUV, bU, width = maxWidth == 0 ? t->w : maxWidth;\
	int alphaStride = t->w;\
	unsigned int y;\
	unsigned char *ySrcEven, *ySrcOdd, *yLineEnd, *uSrc, *vSrc, *out1, *out2;\
	\
	for (y = 0; y < t->h; y += 2)\
	{\
		ySrcEven = t->y + y * t->yStride;\
		ySrcOdd  = t->y + (y + 1) * t->yStride;\
		uSrc = t->u + y * t->uStride / 2;\
		vSrc = t->v + y * t->vStride / 2;\
		out1 = t->out + y * stride;\
		out2 = t->out + (y + 1) * stride;\
		\
		for (yLineEnd = ySrcEven + width; ySrcEven != yLineEnd;)\
		{\
			cu = *uSrc; ++uSrc;\
			cv = *vSrc; ++vSrc;\
			rV   = RVTable[cv];\
			gUV  = GUTable[cu] + GVTable[cv];\
			bU   = BUTable[cu];\
			\
			rgbY1 = YTable[*ySrcEven]; a1 = ySrcEven[alphaStride]; ++ySrcEven;\
			rgbY2 = YTable[*ySrcOdd];  a2 = ySrcOdd [alphaStride];  ++ySrcOdd;\
			rgbY3 = YTable[*ySrcEven]; a3 = ySrcEven[alphaStride]; ++ySrcEven;\
			rgbY4 = YTable[*ySrcOdd];  a4 = ySrcOdd [alphaStride];  ++ySrcOdd;\
			\
			if (a1 > 16)\
			{\
				CLIP_RGB_COLOR(out1[i1], rgbY1 + rV );\
				CLIP_RGB_COLOR(out1[i2], rgbY1 - gUV);\
				CLIP_RGB_COLOR(out1[i3], rgbY1 + bU );\
				out1[aindex1] = a1 >= 235 ? 255 : (unsigned char) (((a1 - 16) * 255) / 219);\
			}\
			else *((unsigned int*) out1) = 0;\
			\
			if (a2 > 16)\
			{\
				CLIP_RGB_COLOR(out2[i1], rgbY2 + rV );\
				CLIP_RGB_COLOR(out2[i2], rgbY2 - gUV);\
				CLIP_RGB_COLOR(out2[i3], rgbY2 + bU );\
				out2[aindex1] = a2 >= 235 ? 255 : (unsigned char) (((a2 - 16) * 255) / 219);\
			}\
			else *((unsigned int*) out2) = 0;\
			\
			if (a3 > 16)\
			{\
				CLIP_RGB_COLOR(out1[j1], rgbY3 + rV );\
				CLIP_RGB_COLOR(out1[j2], rgbY3 - gUV);\
				CLIP_RGB_COLOR(out1[j3], rgbY3 + bU );\
				out1[aindex2] = a3 >= 235 ? 255 : (unsigned char) (((a3 - 16) * 255) / 219);\
			}\
			else *((unsigned int*) &out1[4]) = 0;\
			\
			if (a4 > 16)\
			{\
				CLIP_RGB_COLOR(out2[j1], rgbY4 + rV );\
				CLIP_RGB_COLOR(out2[j2], rgbY4 - gUV);\
				CLIP_RGB_COLOR(out2[j3], rgbY4 + bU );\
				out2[aindex2] = a4 >= 235 ? 255 : (unsigned char) (((a4 - 16) * 255) / 219);\
			}\
			else *((unsigned int*) &out2[4]) = 0;\
			\
			out1 += nBytes2;  out2 += nBytes2;\
		}\
	}\

void decodeRGB(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 3, 3, 0, 0, 1, 2, 3, 4, 5);
}

void decodeRGBA(struct TheoraPixelTransform* t)
{
	_decodeRGBA(t, t->w * 4, 4, 0, 0, 1, 2, 4, 5, 6, 3, 7);
// This is the old 2-phase version, leaving it here in case more debugging is needed
//	_decodeRGB(t, t->w * 4, 4, 0, 0, 1, 2, 4, 5, 6);
//	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeRGBX(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 4, 4, 0, 0, 1, 2, 4, 5, 6);
}

void decodeARGB(struct TheoraPixelTransform* t)
{
	_decodeRGBA(t, t->w * 4, 4, 0, 1, 2, 3, 5, 6, 7, 0, 4);
// This is the old 2-phase version, leaving it here in case more debugging is needed
//	_decodeRGB(t, t->w * 4, 4, 0, 1, 2, 3, 5, 6, 7);
//	_decodeAlpha(t, t->w * 4);
}

void decodeXRGB(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 4, 4, 0, 1, 2, 3, 5, 6, 7);
}

void decodeBGR(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 3, 3, 0, 2, 1, 0, 5, 4, 3);
}

void decodeBGRA(struct TheoraPixelTransform* t)
{
	_decodeRGBA(t, t->w * 4, 4, 0, 2, 1, 0, 6, 5, 4, 3, 7);
// This is the old 2-phase version, leaving it here in case more debugging is needed
//	_decodeRGB(t, t->w * 4, 4, 0, 2, 1, 0, 6, 5, 4);
//	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeBGRX(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 4, 4, 0, 2, 1, 0, 6, 5, 4);
}

void decodeABGR(struct TheoraPixelTransform* t)
{
	_decodeRGBA(t, t->w * 4, 4, 0, 3, 2, 1, 7, 6, 5, 0, 4);
// This is the old 2-phase version, leaving it here in case more debugging is needed
//	_decodeRGB(t, t->w * 4, 4, 0, 3, 2, 1, 7, 6, 5);
//	_decodeAlpha(t, t->w * 4);
}

void decodeXBGR(struct TheoraPixelTransform* t)
{
	_decodeRGB(t, t->w * 4, 4, 0, 3, 2, 1, 7, 6, 5);
}

void initYUVConversionModule()
{
	//used to bring the table into the high side (scale up) so we
	//can maintain high precision and not use floats (FIXED POINT)
	
	// this is the pseudocode for yuv->rgb conversion
	//        r = 1.164*(*ySrc - 16) + 1.596*(cv - 128);
	//        b = 1.164*(*ySrc - 16)                   + 2.018*(cu - 128);
	//        g = 1.164*(*ySrc - 16) - 0.813*(cv - 128) - 0.391*(cu - 128);
	
    double scale = 1L << 13, temp;
	
	int i;
	for (i = 0; i < 256; ++i)
	{
		temp = i - 128;
		
		YTable[i]  = (int)((1.164 * scale + 0.5) * (i - 16));	//Calc Y component
		RVTable[i] = (int)((1.596 * scale + 0.5) * temp);		//Calc R component
		GUTable[i] = (int)((0.391 * scale + 0.5) * temp);		//Calc G u & v components
		GVTable[i] = (int)((0.813 * scale + 0.5) * temp);
		BUTable[i] = (int)((2.018 * scale + 0.5) * temp);		//Calc B component
	}
}

/*
 * Below are the function versions of the above macros, use those for debugging, but leave the macros for maximum CPU execution speed
 *
 *
 *
 *

void _decodeRGB(struct TheoraPixelTransform* t, int stride, int nBytes, int maxWidth, int i1, int i2, int i3, int j1, int j2, int j3)
{
	register int tmp;
	int nBytes2 = nBytes * 2, cv, cu, rgbY1, rgbY2, rgbY3, rgbY4, rV, gUV, bU, width = maxWidth == 0 ? t->w : maxWidth;
	unsigned int y;
	unsigned char *ySrcEven, *ySrcOdd, *yLineEnd, *uSrc, *vSrc, *out1, *out2;
	
	for (y = 0; y < t->h; y += 2)
	{
		ySrcEven = t->y + y * t->yStride;
		ySrcOdd  = t->y + (y + 1) * t->yStride;
		uSrc = t->u + y * t->uStride / 2;
		vSrc = t->v + y * t->vStride / 2;
		out1 = t->out + y * stride;
		out2 = t->out + (y + 1) * stride;
		
		for (yLineEnd = ySrcEven + width; ySrcEven != yLineEnd;)
		{
			cu = *uSrc; ++uSrc;
			cv = *vSrc; ++vSrc;
			rV   = RVTable[cv];
			gUV  = GUTable[cu] + GVTable[cv];
			bU   = BUTable[cu];
			
			rgbY1 = YTable[*ySrcEven]; ++ySrcEven;
			rgbY2 = YTable[*ySrcOdd];  ++ySrcOdd;
			rgbY3 = YTable[*ySrcEven]; ++ySrcEven;
			rgbY4 = YTable[*ySrcOdd];  ++ySrcOdd;
			
			CLIP_RGB_COLOR(out1[i1], rgbY1 + rV );
			CLIP_RGB_COLOR(out1[i2], rgbY1 - gUV);
			CLIP_RGB_COLOR(out1[i3], rgbY1 + bU );
			
			CLIP_RGB_COLOR(out2[i1], rgbY2 + rV );
			CLIP_RGB_COLOR(out2[i2], rgbY2 - gUV);
			CLIP_RGB_COLOR(out2[i3], rgbY2 + bU );
			
			CLIP_RGB_COLOR(out1[j1], rgbY3 + rV );
			CLIP_RGB_COLOR(out1[j2], rgbY3 - gUV);
			CLIP_RGB_COLOR(out1[j3], rgbY3 + bU );
			
			CLIP_RGB_COLOR(out2[j1], rgbY4 + rV );
			CLIP_RGB_COLOR(out2[j2], rgbY4 - gUV);
			CLIP_RGB_COLOR(out2[j3], rgbY4 + bU );
			
			out1 += nBytes2;  out2 += nBytes2;
		}
	}
}
 
void _decodeRGBA(struct TheoraPixelTransform* t, int stride, int nBytes, int maxWidth, int i1, int i2, int i3, int j1, int j2, int j3, int aindex1, int aindex2)
{
	register int tmp;
	int nBytes2 = nBytes * 2, cv, cu, rgbY1, rgbY2, rgbY3, rgbY4, a1, a2, a3, a4, rV, gUV, bU, width = maxWidth == 0 ? t->w : maxWidth;
	int alphaStride = t->w;
	unsigned int y;
	unsigned char *ySrcEven, *ySrcOdd, *yLineEnd, *uSrc, *vSrc, *out1, *out2;
	
	for (y = 0; y < t->h; y += 2)
	{
		ySrcEven = t->y + y * t->yStride;
		ySrcOdd  = t->y + (y + 1) * t->yStride;
		uSrc = t->u + y * t->uStride / 2;
		vSrc = t->v + y * t->vStride / 2;
		out1 = t->out + y * stride;
		out2 = t->out + (y + 1) * stride;
		
		for (yLineEnd = ySrcEven + width; ySrcEven != yLineEnd;)
		{
			cu = *uSrc; ++uSrc;
			cv = *vSrc; ++vSrc;
			rV   = RVTable[cv];
			gUV  = GUTable[cu] + GVTable[cv];
			bU   = BUTable[cu];
			
			rgbY1 = YTable[*ySrcEven]; a1 = ySrcEven[alphaStride]; ++ySrcEven;
			rgbY2 = YTable[*ySrcOdd];  a2 = ySrcOdd [alphaStride];  ++ySrcOdd;
			rgbY3 = YTable[*ySrcEven]; a3 = ySrcEven[alphaStride]; ++ySrcEven;
			rgbY4 = YTable[*ySrcOdd];  a4 = ySrcOdd [alphaStride];  ++ySrcOdd;
			
			if (a1 >= 32)
			{
				CLIP_RGB_COLOR(out1[i1], rgbY1 + rV );
				CLIP_RGB_COLOR(out1[i2], rgbY1 - gUV);
				CLIP_RGB_COLOR(out1[i3], rgbY1 + bU );
				out1[aindex1] = a1 > 224 ? 255 : a1;
			}
			else *((unsigned int*) out1) = 0;
			
			if (a2 >= 32)
			{
				CLIP_RGB_COLOR(out2[i1], rgbY2 + rV );
				CLIP_RGB_COLOR(out2[i2], rgbY2 - gUV);
				CLIP_RGB_COLOR(out2[i3], rgbY2 + bU );
				out2[aindex1] = a2 > 224 ? 255 : a2;
			}
			else *((unsigned int*) out2) = 0;

			
			if (a3 >= 32)
			{
				CLIP_RGB_COLOR(out1[j1], rgbY3 + rV );
				CLIP_RGB_COLOR(out1[j2], rgbY3 - gUV);
				CLIP_RGB_COLOR(out1[j3], rgbY3 + bU );
				out1[aindex2] = a3 > 224 ? 255 : a3;
			}
			else *((unsigned int*) &out1[4]) = 0;

			if (a4 >= 32)
			{
				CLIP_RGB_COLOR(out2[j1], rgbY4 + rV );
				CLIP_RGB_COLOR(out2[j2], rgbY4 - gUV);
				CLIP_RGB_COLOR(out2[j3], rgbY4 + bU );
				out2[aindex2] = a4 > 224 ? 255 : a4;
			}
			else *((unsigned int*) &out2[4]) = 0;

			out1 += nBytes2;  out2 += nBytes2;
		}
	}
}
*/
#endif
