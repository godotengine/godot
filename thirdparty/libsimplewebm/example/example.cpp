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
#include "include/decoder/VorbisDecoder.h"
#include "include/decoder/VPXDecoder.h"

#include <webm/file_reader.h>

#include <cstdio>
#include <memory>

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		std::fprintf(stderr, "Usage: %s <file.webm>\n", argv[0]);
		return 1;
	}

	FILE *file = std::fopen(argv[1], "rb");

	if (!file)
	{
		std::fprintf(stderr, "Failed to open file: %s\n", argv[1]);
		return 1;
	}

	webm::FileReader reader(file);
	WebMDemuxer demuxer(&reader);

	if (!demuxer.isOpen())
	{
		std::fprintf(stderr, "Failed to open WebM file: %s\n", argv[1]);
		return 1;
	}

	VPXDecoder videoDec(demuxer, 8);

	const WebMDemuxer::AUDIO_CODEC audioCodec = demuxer.getAudioCodec();

	std::unique_ptr<VorbisDecoder> vorbisDec;
	std::unique_ptr<OpusDecoder> opusDec;

	switch (audioCodec)
	{
		case WebMDemuxer::AUDIO_VORBIS:
			vorbisDec.reset(new VorbisDecoder(demuxer));
			break;

		case WebMDemuxer::AUDIO_OPUS:
			opusDec.reset(new OpusDecoder(demuxer));
			break;

		default:
			break;
	}

	std::fprintf(stderr, "File: %s\n", argv[1]);
	std::fprintf(stderr, "Length: %.3f seconds\n", demuxer.getLength());

	std::fprintf(stderr, "Video decoder: %s\n", videoDec.isOpen() ? "opened" : "unavailable");

	if (vorbisDec)
	{
		std::fprintf(stderr, "Audio decoder: Vorbis (%s)\n", vorbisDec->isOpen() ? "opened" : "unavailable");
	}
	else if (opusDec)
	{
		std::fprintf(stderr, "Audio decoder: Opus (%s)\n", opusDec->isOpen() ? "opened" : "unavailable");
	}
	else
	{
		std::fprintf(stderr, "Audio decoder: unavailable\n");
	}

	WebMFrame videoFrame;
	WebMFrame audioFrame;
	VPXDecoder::Image image;

	const bool audioDecoderOpen = (vorbisDec && vorbisDec->isOpen()) || (opusDec && opusDec->isOpen());

	std::unique_ptr<short[]> pcm;

	if (audioDecoderOpen)
	{
		const int channels = demuxer.getChannels();

		int bufferSamples = 0;

		if (vorbisDec && vorbisDec->isOpen())
		{
			bufferSamples = vorbisDec->getBufferSamples();
		}
		else if (opusDec && opusDec->isOpen())
		{
			bufferSamples = opusDec->getBufferSamples();
		}

		if (channels > 0 && bufferSamples > 0)
		{
			pcm.reset(new short[bufferSamples * channels]);
		}
	}

	unsigned int videoFrames = 0;
	unsigned int audioFrames = 0;
	unsigned int decodedImages = 0;

	bool hasAlpha = false;
	bool success = true;

	while (demuxer.readFrame(&videoFrame, &audioFrame))
	{
		if (videoDec.isOpen() && videoFrame.isValid())
		{
			if (!videoDec.decode(videoFrame))
			{
				std::fprintf(stderr, "Video decode error\n");
				success = false;
				break;
			}

			++videoFrames;

			while (videoDec.getImage(image) == VPXDecoder::NO_ERROR)
			{
				++decodedImages;

				hasAlpha |= image.alpha != nullptr;
			}
		}

		if (audioFrame.isValid() && audioDecoderOpen)
		{
			int numOutSamples = 0;
			bool decoded = false;

			if (vorbisDec && vorbisDec->isOpen())
			{
				decoded = vorbisDec->getPCMS16(audioFrame, pcm.get(), numOutSamples);
			}
			else if (opusDec && opusDec->isOpen())
			{
				decoded = opusDec->getPCMS16(audioFrame, pcm.get(), numOutSamples);
			}

			if (!decoded)
			{
				std::fprintf(stderr, "Audio decode error\n");
				success = false;
				break;
			}

			++audioFrames;
		}
	}

	return success ? 0 : 1;
}
