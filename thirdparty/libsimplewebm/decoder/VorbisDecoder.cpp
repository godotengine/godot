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

#include "include/decoder/VorbisDecoder.h"

#include <vorbis/codec.h>

#include <cstring>

struct VorbisDecoderState
{
	vorbis_info info;
	vorbis_dsp_state dspState;
	vorbis_block block;
	ogg_packet packet;

	bool hasDSPState;
	bool hasBlock;
};

VorbisDecoder::VorbisDecoder(const WebMDemuxer &demuxer) : m_decoder(NULL), m_numSamples(0), m_channels(demuxer.getChannels())
{
	if (!open(demuxer))
		close();
}

VorbisDecoder::~VorbisDecoder()
{
	close();
}

bool VorbisDecoder::isOpen() const
{
	return m_decoder != NULL;
}

bool VorbisDecoder::getPCMS16(const WebMFrame &frame, short *buffer, int &numOutSamples)
{
	numOutSamples = 0;

	if (!m_decoder || !buffer || frame.bufferSize <= 0)
		return false;

	m_decoder->packet.packet = frame.buffer;
	m_decoder->packet.bytes = frame.bufferSize;

	if (vorbis_synthesis(&m_decoder->block, &m_decoder->packet))
		return false;

	if (vorbis_synthesis_blockin(&m_decoder->dspState, &m_decoder->block))
		return false;

	const int maxSamples = getBufferSamples();

	int samplesCount;
	int count = 0;
	float **pcm;

	while ((samplesCount = vorbis_synthesis_pcmout(&m_decoder->dspState, &pcm)) > 0)
	{
		const int toConvert = samplesCount <= maxSamples ? samplesCount : maxSamples;

		for (int c = 0; c < m_channels; ++c)
		{
			float *samples = pcm[c];

			for (int i = 0, j = c; i < toConvert; ++i, j += m_channels)
			{
				int sample = static_cast<int>(samples[i] * 32767.0f);

				if (sample > 32767)
					sample = 32767;
				else if (sample < -32768)
					sample = -32768;

				buffer[count + j] = static_cast<short>(sample);
			}
		}

		vorbis_synthesis_read(&m_decoder->dspState, toConvert);
		count += toConvert;
	}

	numOutSamples = count;
	return true;
}

// -- GODOT begin --
bool VorbisDecoder::getPCMF(const WebMFrame &frame, float *buffer, int &numOutSamples)
{
	numOutSamples = 0;

	if (!m_decoder || !buffer || frame.bufferSize <= 0)
		return false;

	m_decoder->packet.packet = frame.buffer;
	m_decoder->packet.bytes = frame.bufferSize;

	if (vorbis_synthesis(&m_decoder->block, &m_decoder->packet))
		return false;

	if (vorbis_synthesis_blockin(&m_decoder->dspState, &m_decoder->block))
		return false;

	const int maxSamples = getBufferSamples();

	int samplesCount;
	int count = 0;
	float **pcm;

	while ((samplesCount = vorbis_synthesis_pcmout(&m_decoder->dspState, &pcm)) > 0)
	{
		const int toConvert = samplesCount <= maxSamples ? samplesCount : maxSamples;

		for (int c = 0; c < m_channels; ++c)
		{
			float *samples = pcm[c];
			for (int i = 0, j = c; i < toConvert; ++i, j += m_channels)
			{
				buffer[count + j] = samples[i];
			}
		}

		vorbis_synthesis_read(&m_decoder->dspState, toConvert);
		count += toConvert;
	}

	numOutSamples = count;
	return true;
}
// -- GODOT end --

bool VorbisDecoder::open(const WebMDemuxer &demuxer)
{
	size_t extradataSize = 0;
	const unsigned char *extradata = demuxer.getAudioExtradata(extradataSize);

	if (!extradata || extradataSize < 3 || extradata[0] != 2)
		return false;

	size_t headerSize[3] = { 0 };
	size_t offset = 1;

	// Calculate the sizes of the first two Vorbis headers.
	for (int i = 0; i < 2; ++i)
	{
		for (;;)
		{
			if (offset >= extradataSize)
				return false;

			headerSize[i] += extradata[offset];

			if (extradata[offset++] < 0xFF)
				break;
		}
	}

	headerSize[2] = extradataSize - (headerSize[0] + headerSize[1] + offset);

	if (headerSize[0] + headerSize[1] + headerSize[2] + offset != extradataSize)
		return false;

	ogg_packet packets[3];
	std::memset(packets, 0, sizeof(packets));

	packets[0].packet = const_cast<unsigned char *>(extradata) + offset;
	packets[0].bytes = headerSize[0];
	packets[0].b_o_s = 1;

	packets[1].packet = const_cast<unsigned char *>(extradata) + offset + headerSize[0];
	packets[1].bytes = headerSize[1];

	packets[2].packet = const_cast<unsigned char *>(extradata) + offset + headerSize[0] + headerSize[1];
	packets[2].bytes = headerSize[2];

	m_decoder = new VorbisDecoderState;
	m_decoder->hasDSPState = false;
	m_decoder->hasBlock = false;

	vorbis_info_init(&m_decoder->info);

	vorbis_comment comment;
	vorbis_comment_init(&comment);

	for (int i = 0; i < 3; ++i)
	{
		if (vorbis_synthesis_headerin(&m_decoder->info, &comment, &packets[i]))
		{
			vorbis_comment_clear(&comment);
			return false;
		}
	}

	vorbis_comment_clear(&comment);

	if (vorbis_synthesis_init(&m_decoder->dspState, &m_decoder->info))
		return false;

	m_decoder->hasDSPState = true;

	if (m_decoder->info.channels != m_channels || m_decoder->info.rate != static_cast<long>(demuxer.getSampleRate()))
	{
		return false;
	}

	if (vorbis_block_init(&m_decoder->dspState, &m_decoder->block))
		return false;

	m_decoder->hasBlock = true;

	std::memset(&m_decoder->packet, 0, sizeof(m_decoder->packet));

	m_numSamples = 4096 / m_channels;

	return true;
}

void VorbisDecoder::close()
{
	if (!m_decoder)
		return;

	if (m_decoder->hasBlock)
		vorbis_block_clear(&m_decoder->block);

	if (m_decoder->hasDSPState)
		vorbis_dsp_clear(&m_decoder->dspState);

	vorbis_info_clear(&m_decoder->info);

	delete m_decoder;
	m_decoder = NULL;
}
