/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifndef _TheoraPixelTransform_h
#define _TheoraPixelTransform_h

struct TheoraPixelTransform
{
	unsigned char *raw, *y, *u, *v, *out;
	unsigned int w, h, rawStride, yStride, uStride, vStride;
};

#endif
