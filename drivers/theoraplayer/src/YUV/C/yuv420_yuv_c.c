/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include "yuv_util.h"

static void _decodeYUV(struct TheoraPixelTransform* t, int stride, int nBytes, int maxWidth)
{
	int cv, cu, y1, y2, y3, y4, width = maxWidth == 0 ? t->w : maxWidth;
	unsigned char *ySrcEven, *ySrcOdd, *yLineEnd, *uSrc, *vSrc, *out1, *out2;
	unsigned int y;

	for (y=0; y < t->h; y += 2)
	{
		ySrcEven = t->y + y * t->yStride;
		ySrcOdd  = t->y + (y + 1) * t->yStride;
		uSrc = t->u + y * t->uStride / 2;
		vSrc = t->v + y * t->vStride / 2;
		out1 = t->out + y * stride;
		out2 = t->out + (y + 1) * stride;
		
		for (yLineEnd = ySrcEven + width; ySrcEven != yLineEnd;)
		{
			// EVEN columns
			cu = *uSrc; ++uSrc;
			cv = *vSrc; ++vSrc;
			
			y1 = *ySrcEven; ++ySrcEven;
			y2 = *ySrcOdd;  ++ySrcOdd;
			y3 = *ySrcEven; ++ySrcEven;
			y4 = *ySrcOdd;  ++ySrcOdd;
			
			// EVEN columns
			out1[0] = y1;
			out1[1] = cu;
			out1[2] = cv;
			
			out2[0] = y2;
			out2[1] = cu;
			out2[2] = cv;
			
			out1 += nBytes;  out2 += nBytes;
			// ODD columns
			out1[0] = y3;
			out1[1] = cu;
			out1[2] = cv;
			
			out2[0] = y4;
			out2[1] = cu;
			out2[2] = cv;
			out1 += nBytes;  out2 += nBytes;
		}
	}
}

void decodeYUV(struct TheoraPixelTransform* t)
{
	_decodeYUV(t, t->w * 3, 3, 0);
}

void decodeYUVA(struct TheoraPixelTransform* t)
{
	_decodeYUV(t, t->w * 4, 4, 0);
	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeYUVX(struct TheoraPixelTransform* t)
{
	_decodeYUV(t, t->w * 4, 4, 0);
}

void decodeAYUV(struct TheoraPixelTransform* t)
{
	_decodeYUV(incOut(t, 1), t->w * 4, 4, 0);
	_decodeAlpha(t, t->w * 4);
}

void decodeXYUV(struct TheoraPixelTransform* t)
{
	_decodeYUV(incOut(t, 1), t->w * 4, 4, 0);
}

