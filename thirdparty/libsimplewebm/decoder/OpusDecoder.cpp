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

#include "include/decoder/OpusDecoder.h"

#include <opus/opus.h>

struct OpusDecoderState
{
	::OpusDecoder *decoder;
};

OpusDecoder::OpusDecoder(const WebMDemuxer &demuxer) : m_decoder(NULL), m_numSamples(0), m_channels(demuxer.getChannels())
{
	if (!open(demuxer))
		close();
}

OpusDecoder::~OpusDecoder()
{
	close();
}

bool OpusDecoder::isOpen() const
{
	return m_decoder != NULL;
}

bool OpusDecoder::getPCMS16(const WebMFrame &frame, short *buffer, int &numOutSamples)
{
	numOutSamples = 0;

	if (!m_decoder || !m_decoder->decoder || !buffer || frame.bufferSize <= 0)
		return false;

	const int samples = opus_decode(m_decoder->decoder, frame.buffer, frame.bufferSize, buffer, m_numSamples, 0);

	if (samples < 0)
		return false;

	numOutSamples = samples;
	return true;
}

// -- GODOT begin --
bool OpusDecoder::getPCMF(const WebMFrame &frame, float *buffer, int &numOutSamples)
{
	numOutSamples = 0;

	if (!m_decoder || !m_decoder->decoder || !buffer || frame.bufferSize <= 0)
		return false;

	const int samples = opus_decode_float(m_decoder->decoder, frame.buffer, frame.bufferSize, buffer, m_numSamples, 0);

	if (samples < 0)
		return false;

	numOutSamples = samples;
	return true;
}
// -- GODOT end --

bool OpusDecoder::open(const WebMDemuxer &demuxer)
{
	int opusError = OPUS_OK;

	::OpusDecoder *decoder = opus_decoder_create(static_cast<opus_int32>(demuxer.getSampleRate()), m_channels, &opusError);

	if (!decoder || opusError != OPUS_OK)
	{
		if (decoder)
			opus_decoder_destroy(decoder);

		return false;
	}

	m_decoder = new OpusDecoderState;
	m_decoder->decoder = decoder;

	// Maximum Opus frame duration is 60 ms.
	m_numSamples = static_cast<int>(demuxer.getSampleRate() * 0.06 + 0.5);

	return true;
}

void OpusDecoder::close()
{
	if (!m_decoder)
		return;

	if (m_decoder->decoder)
	{
		opus_decoder_destroy(m_decoder->decoder);
		m_decoder->decoder = NULL;
	}

	delete m_decoder;
	m_decoder = NULL;
}