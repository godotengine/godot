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

#ifndef WEBM_DEMUXER_H
#define WEBM_DEMUXER_H

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include <webm/callback.h>
#include <webm/reader.h>
#include <webm/webm_parser.h>

class WebMFrame
{
	WebMFrame(const WebMFrame &);
	void operator =(const WebMFrame &);

public:
	WebMFrame();
	~WebMFrame();

	inline bool isValid() const
	{
		return bufferSize > 0;
	}

	inline bool hasAlpha() const
	{
		return alphaBufferSize > 0;
	}

	long bufferSize, bufferCapacity;
	unsigned char *buffer;

	long alphaBufferSize, alphaBufferCapacity;
	unsigned char *alphaBuffer;

	double time;
	bool key;
};

class WebMDemuxer
{
	WebMDemuxer(const WebMDemuxer &);
	void operator =(const WebMDemuxer &);

	struct ParsedFrame
	{
		std::vector<unsigned char> buffer;
		std::vector<unsigned char> alphaBuffer;
		double time;
		bool key;
		bool video;

		ParsedFrame() : time(0), key(false), video(false) {}
	};

	class ParserCallback;

public:
	enum VIDEO_CODEC
	{
		NO_VIDEO,
		VIDEO_VP8,
		VIDEO_VP9
	};

	enum AUDIO_CODEC
	{
		NO_AUDIO,
		AUDIO_VORBIS,
		AUDIO_OPUS
	};

	// videoTrack and audioTrack are zero-based indexes into the video/audio
	// tracks in the WebM file, not Matroska track numbers.
	WebMDemuxer(webm::Reader *reader, int videoTrack = 0, int audioTrack = 0);
	~WebMDemuxer();

	inline bool isOpen() const
	{
		return m_isOpen;
	}

	inline bool isEOS() const
	{
		return m_eos;
	}

	double getLength() const;

	VIDEO_CODEC getVideoCodec() const;
	int getWidth() const;
	int getHeight() const;

	AUDIO_CODEC getAudioCodec() const;
	const unsigned char *getAudioExtradata(size_t &size) const;
	double getSampleRate() const;
	int getChannels() const;
	int getAudioDepth() const;

	bool readFrame(WebMFrame *videoFrame, WebMFrame *audioFrame);

private:
	webm::Reader *m_reader;
	webm::WebmParser m_parser;

	ParserCallback *m_callback;

	std::vector<ParsedFrame> m_frames;
	size_t m_frameIndex;

	double m_length;

	// These are zero-based indexes supplied by the caller.
	int m_videoTrack;
	int m_audioTrack;

	VIDEO_CODEC m_vCodec;
	AUDIO_CODEC m_aCodec;

	int m_width;
	int m_height;

	double m_sampleRate;
	int m_channels;
	int m_audioDepth;

	std::vector<unsigned char> m_audioExtradata;

	bool m_isOpen;
	bool m_eos;
	bool m_parseError;
};

#endif // WEBMDEMUXER_H