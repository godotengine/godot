/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#if defined(__THEORA) && !defined(_TheoraVideoClip_Theora_h)
#define _TheoraVideoClip_Theora_h

#include <ogg/ogg.h>
#include <vorbis/vorbisfile.h>
#include <theora/theoradec.h>
#include "TheoraAudioPacketQueue.h"
#include "TheoraVideoClip.h"

struct TheoraInfoStruct
{
	// ogg/vorbis/theora variables
	ogg_sync_state   OggSyncState;
	ogg_page         OggPage;
	ogg_stream_state VorbisStreamState;
	ogg_stream_state TheoraStreamState;
	//Theora State
	th_info        TheoraInfo;
	th_comment     TheoraComment;
	th_setup_info* TheoraSetup;
	th_dec_ctx*    TheoraDecoder;
	//Vorbis State
	vorbis_info      VorbisInfo;
	vorbis_dsp_state VorbisDSPState;
	vorbis_block     VorbisBlock;
	vorbis_comment   VorbisComment;
};

class TheoraVideoClip_Theora : public TheoraVideoClip, public TheoraAudioPacketQueue
{
protected:
	TheoraInfoStruct mInfo; // a pointer is used to avoid having to include theora & vorbis headers
	int mTheoraStreams, mVorbisStreams;	// Keeps track of Theora and Vorbis Streams

	long seekPage(long targetFrame, bool return_keyframe);
	void doSeek();
	void readTheoraVorbisHeaders();
	unsigned int mReadAudioSamples;
	unsigned long mLastDecodedFrameNumber;
public:
	TheoraVideoClip_Theora(TheoraDataSource* data_source,
						   TheoraOutputMode output_mode,
						   int nPrecachedFrames,
						   bool usePower2Stride);
	~TheoraVideoClip_Theora();

	bool _readData();
	bool decodeNextFrame();
	void _restart();
	void load(TheoraDataSource* source);
	float decodeAudio();
	void decodedAudioCheck();
	std::string getDecoderName() { return "Theora"; }
};

#endif
