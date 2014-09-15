/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include <memory.h>
#include "TheoraPixelTransform.h"
#include "TheoraVideoClip.h"
#include "TheoraVideoFrame.h"
#include "TheoraVideoManager.h"

//#define YUV_TEST // uncomment this if you want to benchmark YUV decoding functions

extern "C"
{
void decodeRGB  (struct TheoraPixelTransform* t);
void decodeRGBA (struct TheoraPixelTransform* t);
void decodeRGBX (struct TheoraPixelTransform* t);
void decodeARGB (struct TheoraPixelTransform* t);
void decodeXRGB (struct TheoraPixelTransform* t);
void decodeBGR  (struct TheoraPixelTransform* t);
void decodeBGRA (struct TheoraPixelTransform* t);
void decodeBGRX (struct TheoraPixelTransform* t);
void decodeABGR (struct TheoraPixelTransform* t);
void decodeXBGR (struct TheoraPixelTransform* t);
void decodeGrey (struct TheoraPixelTransform* t);
void decodeGrey3(struct TheoraPixelTransform* t);
void decodeGreyA(struct TheoraPixelTransform* t);
void decodeGreyX(struct TheoraPixelTransform* t);
void decodeAGrey(struct TheoraPixelTransform* t);
void decodeXGrey(struct TheoraPixelTransform* t);
void decodeYUV  (struct TheoraPixelTransform* t);
void decodeYUVA (struct TheoraPixelTransform* t);
void decodeYUVX (struct TheoraPixelTransform* t);
void decodeAYUV (struct TheoraPixelTransform* t);
void decodeXYUV (struct TheoraPixelTransform* t);
}

static void (*conversion_functions[])(struct TheoraPixelTransform*) = {0,
	decodeRGB,
	decodeRGBA,
	decodeRGBX,
	decodeARGB,
	decodeXRGB,
	decodeBGR,
	decodeBGRA,
	decodeBGRX,
	decodeABGR,
	decodeXBGR,
	decodeGrey,
	decodeGrey3,
	decodeGreyA,
	decodeGreyX,
	decodeAGrey,
	decodeXGrey,
	decodeYUV,
	decodeYUVA,
	decodeYUVX,
	decodeAYUV,
	decodeXYUV
};

TheoraVideoFrame::TheoraVideoFrame(TheoraVideoClip* parent)
{
	mReady = mInUse = false;
	mParent = parent;
	mIteration = 0;
	// number of bytes based on output mode
	int bytemap[]={0, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 1, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4};
	mBpp = bytemap[mParent->getOutputMode()];
	unsigned int size = mParent->getStride() * mParent->mHeight * mBpp;
	try
	{
		mBuffer = new unsigned char[size];
	}
	catch (std::bad_alloc)
	{
		mBuffer = NULL;
		return;
	}
	memset(mBuffer, 255, size);
}

TheoraVideoFrame::~TheoraVideoFrame()
{
	if (mBuffer) delete [] mBuffer;
}

int TheoraVideoFrame::getWidth()
{
	return mParent->getWidth();
}

int TheoraVideoFrame::getStride()
{
	return mParent->mStride;
}

int TheoraVideoFrame::getHeight()
{
	return mParent->getHeight();
}

unsigned char* TheoraVideoFrame::getBuffer()
{
	return mBuffer;
}

void TheoraVideoFrame::decode(struct TheoraPixelTransform* t)
{
	if (t->raw != NULL)
	{
		int bufferStride = mParent->getWidth() * mBpp;
		if (bufferStride == t->rawStride)
		{
			memcpy(mBuffer, t->raw, t->rawStride * mParent->getHeight());
		}
		else
		{
			unsigned char *buff = mBuffer, *src = t->raw;
			int i, h = mParent->getHeight();
			for (i = 0; i < h; ++i, buff += bufferStride, src += t->rawStride)
			{
				memcpy(buff, src, bufferStride);
			}
		}
	}
	else
	{
		t->out = mBuffer;
		t->w = mParent->getWidth();
		t->h = mParent->getHeight();
        
#ifdef YUV_TEST // when benchmarking yuv conversion functions during development, do a timed average
        #define N 1000
        clock_t time = clock();
        for (int i = 0; i < N; ++i)
        {
            conversion_functions[mParent->getOutputMode()](t);
        }
        float diff = (clock() - time) * 1000.0f / CLOCKS_PER_SEC;
        
		char s[128];
		sprintf(s, "%.2f", diff / N);
        TheoraVideoManager::getSingleton().logMessage("YUV Decoding time: " + std::string(s) + " ms\n");
#else
		conversion_functions[mParent->getOutputMode()](t);
#endif
	}
	mReady = true;
}

void TheoraVideoFrame::clear()
{
	mInUse = mReady = false;
}
