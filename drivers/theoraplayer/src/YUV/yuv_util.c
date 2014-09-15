/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include "yuv_util.h"

struct TheoraPixelTransform* incOut(struct TheoraPixelTransform* t, int n)
{
	// used for XRGB, XBGR and similar
	t->out += n;
	return t;
}

void _decodeAlpha(struct TheoraPixelTransform* t, int stride)
{
	int width = t->w;
	unsigned char *ySrc, *yLineEnd, *out;
	int luma;
	unsigned int y;
	for (y = 0; y < t->h; y++)
	{
		ySrc = t->y + y * t->yStride + width;
		out = t->out + y * stride;
		
		for (yLineEnd = ySrc + width; ySrc != yLineEnd; ++ySrc, out += 4)
		{
			luma = (*ySrc);
            // because in YCbCr specification, luma values are in the range of [16, 235]
            // account for 'footroom' and 'headroom' ranges while using luma values as alpha channel
            if (luma <= 16)       *out = 0;
            else if (luma >= 235) *out = 255;
            else                  *out = (unsigned char) (((luma - 16) * 255) / 219);
		}
	}
}
