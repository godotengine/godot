/*
 * MPEGFrame.cpp
 * -------------
 * Purpose: Basic MPEG frame parsing functionality
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "MPEGFrame.h"
#include "../common/FileReader.h"

OPENMPT_NAMESPACE_BEGIN

// Samples per frame - for each MPEG version and all three layers
static const uint16 samplesPerFrame[2][3] =
{
	{ 384, 1152, 1152 },	// MPEG 1
	{ 384, 1152,  576 }		// MPEG 2 / 2.5
};
// Bit rates for each MPEG version and all three layers
static const uint16 bitRates[2][3][15] =
{
	// MPEG 1
	{
		{ 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448 },	// Layer 1
		{ 0, 32, 48, 56,  64,  80,  96, 112, 128, 160, 192, 224, 256, 320, 384 },	// Layer 2
		{ 0, 32, 40, 48,  56,  64,  80,  96, 112, 128, 160, 192, 224, 256, 320 }	// Layer 3
	},
	// MPEG 2 / 2.5
	{
		{ 0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256 },	// Layer 1
		{ 0,  8, 16, 24, 32, 40, 48,  56,  64,  80,  96, 112, 128, 144, 160 },	// Layer 2
		{ 0,  8, 16, 24, 32, 40, 48,  56,  64,  80,  96, 112, 128, 144, 160 }	// Layer 3
	}
};
// Sampling rates for each MPEG version and all three layers
static const uint16 samplingRates[4][3] =
{ 
	{ 11025, 12000,  8000 },	// MPEG 2.5
	{     0,     0,     0 },	// Invalid
	{ 22050, 24000, 16000 },	// MPEG 2
	{ 44100, 48000, 32000 }		// MPEG 1
};
// Samples per Frame / 8
static const uint8 mpegCoefficients[2][3] =
{
	{ 12, 144, 144 },	// MPEG 1
	{ 12, 144,  72 }	// MPEG 2 / 2.5
};
// Side info size = Offset in frame where Xing/Info magic starts
static const uint8 sideInfoSize[2][2] =
{
	{ 17, 32 },	// MPEG 1
	{  9, 17 }	// MPEG 2 / 2.5
};


bool MPEGFrame::IsMPEGHeader(const uint8 (&header)[3])
{
	return header[0] == 0xFF && (header[1] & 0xE0) == 0xE0	// Sync
		&& (header[1] & 0x18) != 0x08	// Invalid MPEG version
		&& (header[1] & 0x06) != 0x00	// Invalid MPEG layer
		&& (header[2] & 0x0C) != 0x0C	// Invalid frequency
		&& (header[2] & 0xF0) != 0xF0;	// Invalid bitrate
}


MPEGFrame::MPEGFrame(FileReader &file)
	: frameSize(0)
	, numSamples(0)
	, isValid(false)
	, isLAME(false)
{
	uint8 header[4];
	file.ReadArray(header);
	
	if(!IsMPEGHeader(reinterpret_cast<const uint8(&)[3]>(header)))
		return;
		
	uint8 version = (header[1] & 0x18) >> 3;
	uint8 mpeg1 = (version == 3) ? 0 : 1;
	uint8 layer = 3 - ((header[1] & 0x06) >> 1);
	uint8 bitRate = (header[2] & 0xF0) >> 4;
	uint8 sampleRate = (header[2] & 0x0C) >> 2;
	uint8 padding = (header[2] & 0x02) >> 1;
	bool stereo = ((header[3] & 0xC0) >> 6) != 3;

	isValid = true;
	frameSize = (((mpegCoefficients[mpeg1][layer] * (bitRates[mpeg1][layer][bitRate] * 1000) / samplingRates[version][sampleRate]) + padding)) * (layer == 0 ? 4 : 1);
	numSamples = samplesPerFrame[mpeg1][layer];
	if(stereo) numSamples *= 2u;

	uint32 lameOffset = sideInfoSize[mpeg1][stereo ? 1 : 0];
	if(frameSize < lameOffset + 8)
		return;

	uint8 frame[36];
	file.ReadStructPartial(frame, lameOffset + 4);
	// Don't check first two bytes, might be CRC
	for(uint32 i = 2; i < lameOffset; i++)
	{
		if(frame[i] != 0)
			return;
	}

	// This is all we really need to know for our purposes in the MO3 decoder.
	isLAME = !memcmp(frame + lameOffset, "Info", 4) || !memcmp(frame + lameOffset, "Xing", 4);
}

OPENMPT_NAMESPACE_END
