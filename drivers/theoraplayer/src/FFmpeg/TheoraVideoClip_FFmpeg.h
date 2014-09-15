/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#if defined(__FFMPEG) && !defined(_TheoraVideoClip_FFmpeg_h)
#define _TheoraVideoClip_FFmpeg_h

#include "TheoraAudioPacketQueue.h"
#include "TheoraVideoClip.h"

struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVFrame;
struct AVIOContext;

class TheoraVideoClip_FFmpeg : public TheoraVideoClip, public TheoraAudioPacketQueue
{
protected:
	bool mLoaded;
	
	AVFormatContext* mFormatContext;
	AVCodecContext* mCodecContext;
	AVIOContext* mAvioContext;
	AVCodec* mCodec;
	AVFrame* mFrame;
	unsigned char* mInputBuffer;
	int mVideoStreamIndex;
	int mFrameNumber;
	
	void unload();
	void doSeek();
public:
	TheoraVideoClip_FFmpeg(TheoraDataSource* data_source,
								 TheoraOutputMode output_mode,
								 int nPrecachedFrames,
								 bool usePower2Stride);
	~TheoraVideoClip_FFmpeg();
	
	bool _readData();
	bool decodeNextFrame();
	void _restart();
	void load(TheoraDataSource* source);
	float decodeAudio();
	void decodedAudioCheck();
	std::string getDecoderName() { return "FFmpeg"; }
};

#endif
