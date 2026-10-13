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

#ifndef OPUS_DECODER_H
#define OPUS_DECODER_H

#include "include/demuxer/WebMDemuxer.h"

struct OpusDecoderState;

class OpusDecoder
{
	OpusDecoder(const OpusDecoder &);
	void operator=(const OpusDecoder &);

public:
	explicit OpusDecoder(const WebMDemuxer &demuxer);
	~OpusDecoder();

	bool isOpen() const;

	inline int getBufferSamples() const
	{
		return m_numSamples;
	}

	bool getPCMS16(const WebMFrame &frame, short *buffer, int &numOutSamples);
	bool getPCMF(const WebMFrame &frame, float *buffer, int &numOutSamples);

private:
	bool open(const WebMDemuxer &demuxer);
	void close();

	OpusDecoderState *m_decoder;
	int m_numSamples;
	int m_channels;
};

#endif // OPUSDECODER_H