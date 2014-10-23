/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include <stdlib.h>
#include "TheoraAudioPacketQueue.h"
#include "TheoraAudioInterface.h"

TheoraAudioPacketQueue::TheoraAudioPacketQueue()
{
	mTheoraAudioPacketQueue = NULL;
}

TheoraAudioPacketQueue::~TheoraAudioPacketQueue()
{
	destroyAllAudioPackets();
}

float TheoraAudioPacketQueue::getAudioPacketQueueLength()
{
	float len = 0;
	for (TheoraAudioPacket* p = mTheoraAudioPacketQueue; p != NULL; p = p->next)
		len += p->numSamples;
	
	return len / (mAudioFrequency * mNumAudioChannels);
}

void TheoraAudioPacketQueue::_addAudioPacket(float* data, int numSamples)
{
	TheoraAudioPacket* packet = new TheoraAudioPacket;
	packet->pcm = data;
	packet->numSamples = numSamples;
	packet->next = NULL;


	if (mTheoraAudioPacketQueue == NULL) mTheoraAudioPacketQueue = packet;
	else
	{
		TheoraAudioPacket* last = mTheoraAudioPacketQueue;
		for (TheoraAudioPacket* p = last; p != NULL; p = p->next)
			last = p;
		last->next = packet;
	}
}

void TheoraAudioPacketQueue::addAudioPacket(float** buffer, int numSamples, float gain)
{
	float* data = new float[numSamples * mNumAudioChannels];
	float* dataptr = data;
	int i;
	unsigned int j;
	
	if (gain < 1.0f)
	{
		// apply gain, let's attenuate the samples
		for (i = 0; i < numSamples; ++i)
			for (j = 0; j < mNumAudioChannels; j++, ++dataptr)
				*dataptr = buffer[i][j] * gain;
	}
	else
	{
		// do a simple copy, faster then the above method, when gain is 1.0f
		for (i = 0; i < numSamples; ++i)
			for (j = 0; j < mNumAudioChannels; j++, ++dataptr)
				*dataptr = buffer[j][i];
	}
		
	_addAudioPacket(data, numSamples * mNumAudioChannels);
}

void TheoraAudioPacketQueue::addAudioPacket(float* buffer, int numSamples, float gain)
{
	float* data = new float[numSamples * mNumAudioChannels];
	float* dataptr = data;
	int i, numFloats = numSamples * mNumAudioChannels;
	
	if (gain < 1.0f)
	{
		// apply gain, let's attenuate the samples
		for (i = 0; i < numFloats; ++i, dataptr++)
			*dataptr = buffer[i] * gain;
	}
	else
	{
		// do a simple copy, faster then the above method, when gain is 1.0f
		for (i = 0; i < numFloats; ++i, dataptr++)
			*dataptr = buffer[i];
	}
	
	_addAudioPacket(data, numFloats);
}

TheoraAudioPacket* TheoraAudioPacketQueue::popAudioPacket()
{
	if (mTheoraAudioPacketQueue == NULL) return NULL;
	TheoraAudioPacket* p = mTheoraAudioPacketQueue;
	mTheoraAudioPacketQueue = mTheoraAudioPacketQueue->next;
	return p;
}

void TheoraAudioPacketQueue::destroyAudioPacket(TheoraAudioPacket* p)
{
	if (p == NULL) return;
	delete [] p->pcm;
	delete p;
}

void TheoraAudioPacketQueue::destroyAllAudioPackets()
{
	for (TheoraAudioPacket* p = popAudioPacket(); p != NULL; p = popAudioPacket())
		destroyAudioPacket(p);
}

void TheoraAudioPacketQueue::flushAudioPackets(TheoraAudioInterface* audioInterface)
{
	
	for (TheoraAudioPacket* p = popAudioPacket(); p != NULL; p = popAudioPacket())
	{
		audioInterface->insertData(p->pcm, p->numSamples);
		destroyAudioPacket(p);
	}
}