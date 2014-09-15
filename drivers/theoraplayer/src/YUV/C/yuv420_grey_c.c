/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include "yuv_util.h"

static void _decodeGrey3(struct TheoraPixelTransform* t, int stride, int nBytes)
{
	unsigned char *ySrc = t->y, *yLineEnd, *out = t->out;
	unsigned int y;
	for (y = 0; y < t->h; ++y, ySrc += t->yStride - t->w, out += stride-t->w * nBytes)
		for (yLineEnd = ySrc + t->w; ySrc != yLineEnd; ++ySrc, out += nBytes)
			out[0] = out[1] = out[2] = *ySrc;
}

void decodeGrey(struct TheoraPixelTransform* t)
{
	unsigned char *ySrc = t->y, *yLineEnd, *out = t->out;
	unsigned int y;
	for (y = 0; y < t->h; ++y, ySrc += t->yStride - t->w)
		for (yLineEnd = ySrc + t->w; ySrc != yLineEnd; ++ySrc, ++out)
			*out = *ySrc;

}

void decodeGrey3(struct TheoraPixelTransform* t)
{
	_decodeGrey3(t, t->w * 3, 3);
}

void decodeGreyA(struct TheoraPixelTransform* t)
{
	_decodeGrey3(t, t->w * 4, 4);
	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeGreyX(struct TheoraPixelTransform* t)
{
	_decodeGrey3(t, t->w * 4, 4);
}

void decodeAGrey(struct TheoraPixelTransform* t)
{
	_decodeGrey3(incOut(t, 1), t->w * 4, 4);
	_decodeAlpha(t, t->w * 4);
}

void decodeXGrey(struct TheoraPixelTransform* t)
{
	_decodeGrey3(incOut(t, 1), t->w * 4, 4);
}

