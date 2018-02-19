/*
 * MPEGFrame.h
 * -----------
 * Purpose: Basic MPEG frame parsing functionality
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../common/FileReaderFwd.h"

OPENMPT_NAMESPACE_BEGIN

class MPEGFrame
{
public:
	uint16 frameSize;	// Complete frame size in bytes
	uint16 numSamples;	// Number of samples in this frame (multiplied by number of channels)
	bool isValid;		// Is a valid frame at all
	bool isLAME;		// Has Xing/LAME header

	MPEGFrame(FileReader &file);
	static bool IsMPEGHeader(const uint8 (&header)[3]);
};

OPENMPT_NAMESPACE_END
