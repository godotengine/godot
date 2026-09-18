/*
 *    MIT License
 *
 *    Copyright (c) 2016 Błażej Szczygieł
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

#include "OpusVorbisDecoder.hpp"
#include "VPXDecoder.hpp"

#include <mkvparser/mkvparser.h>

#include <stdio.h>

class MkvReader : public mkvparser::IMkvReader
{
public:
	MkvReader(const char *filePath) :
		m_file(fopen(filePath, "rb"))
	{}
	~MkvReader()
	{
		if (m_file)
			fclose(m_file);
	}

	int Read(long long pos, long len, unsigned char *buf)
	{
		if (!m_file)
			return -1;
		fseek(m_file, pos, SEEK_SET);
		const size_t size = fread(buf, 1, len, m_file);
		if (size < size_t(len))
			return -1;
		return 0;
	}
	int Length(long long *total, long long *available)
	{
		if (!m_file)
			return -1;
		const off_t pos = ftell(m_file);
		fseek(m_file, 0, SEEK_END);
		if (total)
			*total = ftell(m_file);
		if (available)
			*available = ftell(m_file);
		fseek(m_file, pos, SEEK_SET);
		return 0;
	}

private:
	FILE *m_file;
};

int main(int argc, char *argv[])
{
	if (argc != 2)
		return -1;

	WebMDemuxer demuxer(new MkvReader(argv[1]));
	if (demuxer.isOpen())
	{
		VPXDecoder videoDec(demuxer, 8);
		OpusVorbisDecoder audioDec(demuxer);

		WebMFrame videoFrame, audioFrame;

		VPXDecoder::Image image;
		short *pcm = audioDec.isOpen() ? new short[audioDec.getBufferSamples() * demuxer.getChannels()] : NULL;

		fprintf(stderr, "Length: %f\n", demuxer.getLength());

		while (demuxer.readFrame(&videoFrame, &audioFrame))
		{
			if (videoDec.isOpen() && videoFrame.isValid())
			{
				if (!videoDec.decode(videoFrame))
				{
					fprintf(stderr, "Video decode error\n");
					break;
				}
				while (videoDec.getImage(image) == VPXDecoder::NO_ERROR)
				{
// 					for (int p = 0; p < 3; ++p)
// 					{
// 						const int w = image.getWidth(p);
// 						const int h = image.getHeight(p);
// 						int offset = 0;
// 						for (int y = 0; y < h; ++y)
// 						{
// 							fwrite(image.planes[p] + offset, 1, w, stdout);
// 							offset += image.linesize[p];
// 						}
// 					}
				}
			}
			if (audioDec.isOpen() && audioFrame.isValid())
			{
				int numOutSamples;
				if (!audioDec.getPCMS16(audioFrame, pcm, numOutSamples))
				{
					fprintf(stderr, "Audio decode error\n");
					break;
				}
// 				fwrite(pcm, 1, numOutSamples * demuxer.getChannels() * sizeof(short), stdout);
			}
		}

		delete[] pcm;
	}
	return 0;
}
