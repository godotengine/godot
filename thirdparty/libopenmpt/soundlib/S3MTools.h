/*
 * S3MTools.h
 * ----------
 * Purpose: Definition of S3M file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../soundlib/ModSample.h"
#include "../soundlib/SampleIO.h"


OPENMPT_NAMESPACE_BEGIN

// S3M File Header
struct S3MFileHeader
{
	// Magic Bytes
	enum S3MMagic
	{
		idEOF				= 0x1A,
		idS3MType			= 0x10,
		idPanning			= 0xFC,
	};

	// Tracker Versions in the cwtv field
	enum S3MTrackerVersions
	{
		trackerMask			= 0xF000,
		versionMask			= 0x0FFF,

		trkScreamTracker	= 0x1000,
		trkImagoOrpheus		= 0x2000,
		trkImpulseTracker	= 0x3000,
		trkSchismTracker	= 0x4000,
		trkOpenMPT			= 0x5000,
		trkBeRoTracker		= 0x6000,
		trkCreamTracker		= 0x7000,

		trkST3_20			= 0x1320,
		trkIT2_14			= 0x3214,
		trkBeRoTrackerOld	= 0x4100,	// Used from 2004 to 2012
	};

	// Flags
	enum S3MHeaderFlags
	{
		st2Vibrato			= 0x01,	// Vibrato is twice as deep. Cannot be enabled from UI.
		zeroVolOptim		= 0x08,	// Volume 0 optimisations
		amigaLimits			= 0x10,	// Enforce Amiga limits
		fastVolumeSlides	= 0x40,	// Fast volume slides (like in ST3.00)
	};

	// S3M Format Versions
	enum S3MFormatVersion
	{
		oldVersion			= 0x01,	// Old Version, signed samples
		newVersion			= 0x02,	// New Version, unsigned samples
	};

	char     name[28];			// Song Title
	uint8le  dosEof;			// Supposed to be 0x1A, but even ST3 seems to ignore this sometimes (see STRSHINE.S3M by Purple Motion)
	uint8le  fileType;			// File Type, 0x10 = ST3 module
	char     reserved1[2];		// Reserved
	uint16le ordNum;			// Number of order items
	uint16le smpNum;			// Number of sample parapointers
	uint16le patNum;			// Number of pattern parapointers
	uint16le flags;				// Flags, see S3MHeaderFlags
	uint16le cwtv;				// "Made With" Tracker ID, see S3MTrackerVersions
	uint16le formatVersion;		// Format Version, see S3MFormatVersion
	char     magic[4];			// "SCRM" magic bytes
	uint8le  globalVol;			// Default Global Volume (0...64)
	uint8le  speed;				// Default Speed (1...254)
	uint8le  tempo;				// Default Tempo (33...255)
	uint8le  masterVolume;		// Sample Volume (0...127, stereo if high bit is set)
	uint8le  ultraClicks;		// Number of channels used for ultra click removal
	uint8le  usePanningTable;	// 0xFC => read extended panning table
	char     reserved2[8];		// More reserved bytes
	uint16le special;			// Pointer to special custom data (unused)
	uint8le  channels[32];		// Channel setup
};

MPT_BINARY_STRUCT(S3MFileHeader, 96)


// S3M Sample Header
struct S3MSampleHeader
{
	enum SampleType
	{
		typeNone	= 0,
		typePCM		= 1,
		typeAdMel	= 2,
	};

	enum SampleFlags
	{
		smpLoop		= 0x01,
		smpStereo	= 0x02,
		smp16Bit	= 0x04,
	};

	enum SamplePacking
	{
		pUnpacked	= 0x00,	// PCM
		pDP30ADPCM	= 0x01,	// Unused packing type
		pADPCM		= 0x04,	// MODPlugin ADPCM :(
	};

	uint8le  sampleType;		// Sample type, see SampleType
	char     filename[12];		// Sample filename
	uint8le  dataPointer[3];	// Pointer to sample data (divided by 16)
	uint32le length;			// Sample length, in samples
	uint32le loopStart;			// Loop start, in samples
	uint32le loopEnd;			// Loop end, in samples
	uint8le  defaultVolume;		// Default volume (0...64)
	char     reserved1;			// Reserved
	uint8le  pack;				// Packing algorithm, SamplePacking
	uint8le  flags;				// Sample flags
	uint32le c5speed;			// Middle-C frequency
	char     reserved2[12];		// Reserved + Internal ST3 stuff
	char     name[28];			// Sample name
	char     magic[4];			// "SCRS" magic bytes ("SCRI" for Adlib instruments)

	// Convert an S3M sample header to OpenMPT's internal sample header.
	void ConvertToMPT(ModSample &mptSmp) const;
	// Convert OpenMPT's internal sample header to an S3M sample header.
	SmpLength ConvertToS3M(const ModSample &mptSmp);
	// Retrieve the internal sample format flags for this sample.
	SampleIO GetSampleFormat(bool signedSamples) const;
};

MPT_BINARY_STRUCT(S3MSampleHeader, 80)


OPENMPT_NAMESPACE_END
