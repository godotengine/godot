/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifdef _YUV_LIBYUV
#include <libyuv.h>
#include "yuv_util.h"
#include "yuv_libyuv.h"

void decodeRGB(struct TheoraPixelTransform* t)
{
	I420ToRAW(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 3, t->w, t->h);
}

void decodeRGBA(struct TheoraPixelTransform* t)
{
	I420ToABGR(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeRGBX(struct TheoraPixelTransform* t)
{
    I420ToABGR(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
}

void decodeARGB(struct TheoraPixelTransform* t)
{
    I420ToBGRA(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
	_decodeAlpha(t, t->w * 4);
}

void decodeXRGB(struct TheoraPixelTransform* t)
{
    I420ToBGRA(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
}

void decodeBGR(struct TheoraPixelTransform* t)
{
	I420ToRGB24(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 3, t->w, t->h);
}

void decodeBGRA(struct TheoraPixelTransform* t)
{
	I420ToARGB(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
	_decodeAlpha(incOut(t, 3), t->w * 4);
}

void decodeBGRX(struct TheoraPixelTransform* t)
{
	I420ToARGB(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
}

void decodeABGR(struct TheoraPixelTransform* t)
{
    I420ToRGBA(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
	_decodeAlpha(t, t->w * 4);
}

void decodeXBGR(struct TheoraPixelTransform* t)
{
	I420ToRGBA(t->y, t->yStride, t->u, t->uStride, t->v, t->vStride, t->out, t->w * 4, t->w, t->h);
}

void initYUVConversionModule()
{

}
#endif
