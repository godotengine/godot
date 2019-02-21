/*
	MIT License

	Copyright (c) 2016 Błażej Szczygieł

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
*/

#ifndef VPXDECODER_HPP
#define VPXDECODER_HPP

#include "WebMDemuxer.hpp"

struct vpx_codec_ctx;

class VPXDecoder
{
	VPXDecoder(const VPXDecoder &);
	void operator =(const VPXDecoder &);
public:
	class Image
	{
	public:
// -- GODOT begin --
#if 0
// -- GODOT end --
		int getWidth(int plane) const;
		int getHeight(int plane) const;
// -- GODOT begin --
#endif
// -- GODOT end --

		int w, h;
		int cs;
		int chromaShiftW, chromaShiftH;
		unsigned char *planes[3];
		int linesize[3];
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

	inline int getFramesDelay() const
	{
		return m_delay;
	}

	bool decode(const WebMFrame &frame);
	IMAGE_ERROR getImage(Image &image); //The data is NOT copied! Only 3-plane, 8-bit images are supported.

private:
	vpx_codec_ctx *m_ctx;
	const void *m_iter;
	int m_delay;
	int m_last_space;
};

#endif // VPXDECODER_HPP
