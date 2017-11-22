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

#include "OpusVorbisDecoder.hpp"

#include <vorbis/codec.h>
#include <opus/opus.h>

#include <string.h>

struct VorbisDecoder
{
	vorbis_info info;
	vorbis_dsp_state dspState;
	vorbis_block block;
	ogg_packet op;

	bool hasDSPState, hasBlock;
};

/**/

OpusVorbisDecoder::OpusVorbisDecoder(const WebMDemuxer &demuxer) :
	m_vorbis(NULL), m_opus(NULL),
	m_numSamples(0)
{
	switch (demuxer.getAudioCodec())
	{
		case WebMDemuxer::AUDIO_VORBIS:
			m_channels = demuxer.getChannels();
			if (openVorbis(demuxer))
				return;
			break;
		case WebMDemuxer::AUDIO_OPUS:
			m_channels = demuxer.getChannels();
			if (openOpus(demuxer))
				return;
			break;
		default:
			return;
	}
	close();
}
OpusVorbisDecoder::~OpusVorbisDecoder()
{
	close();
}

bool OpusVorbisDecoder::isOpen() const
{
	return (m_vorbis || m_opus);
}

bool OpusVorbisDecoder::getPCMS16(WebMFrame &frame, short *buffer, int &numOutSamples)
{
	if (m_vorbis)
	{
		m_vorbis->op.packet = frame.buffer;
		m_vorbis->op.bytes = frame.bufferSize;

		if (vorbis_synthesis(&m_vorbis->block, &m_vorbis->op))
			return false;
		if (vorbis_synthesis_blockin(&m_vorbis->dspState, &m_vorbis->block))
			return false;

		const int maxSamples = getBufferSamples();
		int samplesCount, count = 0;
		float **pcm;
		while ((samplesCount = vorbis_synthesis_pcmout(&m_vorbis->dspState, &pcm)))
		{
			const int toConvert = samplesCount <= maxSamples ? samplesCount : maxSamples;
			for (int c = 0; c < m_channels; ++c)
			{
				float *samples = pcm[c];
				for (int i = 0, j = c; i < toConvert; ++i, j += m_channels)
				{
					int sample = samples[i] * 32767.0f;
					if (sample > 32767)
						sample = 32767;
					else if (sample < -32768)
						sample = -32768;
					buffer[count + j] = sample;
				}
			}
			vorbis_synthesis_read(&m_vorbis->dspState, toConvert);
			count += toConvert;
		}

		numOutSamples = count;
		return true;
	}
	else if (m_opus)
	{
		const int samples = opus_decode(m_opus, frame.buffer, frame.bufferSize, buffer, m_numSamples, 0);
		if (samples >= 0)
		{
			numOutSamples = samples;
			return true;
		}
	}
	return false;
}

bool OpusVorbisDecoder::getPCMF(WebMFrame &frame, float *buffer, int &numOutSamples) {
	if (m_vorbis) {
		m_vorbis->op.packet = frame.buffer;
		m_vorbis->op.bytes = frame.bufferSize;

		if (vorbis_synthesis(&m_vorbis->block, &m_vorbis->op))
			return false;
		if (vorbis_synthesis_blockin(&m_vorbis->dspState, &m_vorbis->block))
			return false;

		const int maxSamples = getBufferSamples();
		int samplesCount, count = 0;
		float **pcm;
		while ((samplesCount = vorbis_synthesis_pcmout(&m_vorbis->dspState, &pcm))) {
			const int toConvert = samplesCount <= maxSamples ? samplesCount : maxSamples;
			for (int c = 0; c < m_channels; ++c) {
				float *samples = pcm[c];
				for (int i = 0, j = c; i < toConvert; ++i, j += m_channels) {
					buffer[count + j] = samples[i];
				}
			}
			vorbis_synthesis_read(&m_vorbis->dspState, toConvert);
			count += toConvert;
		}

		numOutSamples = count;
		return true;
	} else if (m_opus) {
		const int samples = opus_decode_float(m_opus, frame.buffer, frame.bufferSize, buffer, m_numSamples, 0);
		if (samples >= 0) {
			numOutSamples = samples;
			return true;
		}
	}
	return false;
}

bool OpusVorbisDecoder::openVorbis(const WebMDemuxer &demuxer)
{
	size_t extradataSize = 0;
	const unsigned char *extradata = demuxer.getAudioExtradata(extradataSize);

	if (extradataSize < 3 || !extradata || extradata[0] != 2)
		return false;

	size_t headerSize[3] = {0};
	size_t offset = 1;

	/* Calculate three headers sizes */
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

	ogg_packet op[3];
	memset(op, 0, sizeof op);

	op[0].packet = (unsigned char *)extradata + offset;
	op[0].bytes = headerSize[0];
	op[0].b_o_s = 1;

	op[1].packet = (unsigned char *)extradata + offset + headerSize[0];
	op[1].bytes = headerSize[1];

	op[2].packet = (unsigned char *)extradata + offset + headerSize[0] + headerSize[1];
	op[2].bytes = headerSize[2];

	m_vorbis = new VorbisDecoder;
	m_vorbis->hasDSPState = m_vorbis->hasBlock = false;
	vorbis_info_init(&m_vorbis->info);

	/* Upload three Vorbis headers into libvorbis */
	vorbis_comment vc;
	vorbis_comment_init(&vc);
	for (int i = 0; i < 3; ++i)
	{
		if (vorbis_synthesis_headerin(&m_vorbis->info, &vc, &op[i]))
		{
			vorbis_comment_clear(&vc);
			return false;
		}
	}
	vorbis_comment_clear(&vc);

	if (vorbis_synthesis_init(&m_vorbis->dspState, &m_vorbis->info))
		return false;
	m_vorbis->hasDSPState = true;

	if (m_vorbis->info.channels != m_channels || m_vorbis->info.rate != demuxer.getSampleRate())
		return false;

	if (vorbis_block_init(&m_vorbis->dspState, &m_vorbis->block))
		return false;
	m_vorbis->hasBlock = true;

	memset(&m_vorbis->op, 0, sizeof m_vorbis->op);

	m_numSamples = 4096 / m_channels;

	return true;
}
bool OpusVorbisDecoder::openOpus(const WebMDemuxer &demuxer)
{
	int opusErr = 0;
	m_opus = opus_decoder_create(demuxer.getSampleRate(), m_channels, &opusErr);
	if (!opusErr)
	{
		m_numSamples = demuxer.getSampleRate() * 0.06 + 0.5; //Maximum frame size (for 60 ms frame)
		return true;
	}
	return false;
}

void OpusVorbisDecoder::close()
{
	if (m_vorbis)
	{
		if (m_vorbis->hasBlock)
			vorbis_block_clear(&m_vorbis->block);
		if (m_vorbis->hasDSPState)
			vorbis_dsp_clear(&m_vorbis->dspState);
		vorbis_info_clear(&m_vorbis->info);
		delete m_vorbis;
	}
	if (m_opus)
		opus_decoder_destroy(m_opus);
}
