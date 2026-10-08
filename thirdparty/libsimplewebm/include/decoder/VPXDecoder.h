/*
 *    MIT License
 *
 *    Copyright (c) 2016 Błażej Szczygieł
 * 	  Copyright (c) 2026-present The Simplicity Group & contributors
 * 	  See AUTHORS.md for more information
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a copy
 *    of this software and associated documentation files (the "Software"), to deal
 *    in the Software without restriction, including without limitation the rights
 *    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in all
 *    copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#ifndef VPX_DECODER_H
#define VPX_DECODER_H

#include "include/demuxer/WebMDemuxer.h"

struct vpx_codec_ctx;
struct vpx_codec_iface;

class VPXDecoder
{
	VPXDecoder(const VPXDecoder &);
	void operator =(const VPXDecoder &);

public:
	class Image
	{
	public:
		int getWidth(int plane) const;
		int getHeight(int plane) const;

		int w, h;
		int cs;

		int chromaShiftW, chromaShiftH;

		unsigned char *planes[3];
		int linesize[3];

		unsigned char *alpha;
		int alphaLinesize;
	};

	enum IMAGE_ERROR
	{
		UNSUPPORTED_FRAME = -1,
		NO_ERROR,
		NO_FRAME
	};

	VPXDecoder(const WebMDemuxer &demuxer, unsigned threads = 1);
	~VPXDecoder();

	inline bool isOpen() const
	{
		return (bool)m_ctx;
	}

	inline bool hasAlpha() const
	{
		return (bool)m_alphaCtx;
	}

	inline int getFramesDelay() const
	{
		return m_delay;
	}

	bool decode(const WebMFrame &frame);

	// The image data is not copied. Only 8-bit planar images are supported.
	IMAGE_ERROR getImage(Image &image);

private:
	vpx_codec_ctx *m_ctx;
	vpx_codec_ctx *m_alphaCtx;
	const vpx_codec_iface *m_codecIface;

	const void *m_iter;
	const void *m_alphaIter;

	unsigned m_threads;

	int m_delay;
	int m_last_space;
};

#endif // VPXDECODER_H