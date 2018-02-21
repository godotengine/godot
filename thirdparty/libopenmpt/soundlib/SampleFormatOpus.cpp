/*
 * SampleFormatOpus.cpp
 * --------------------
 * Purpose: Opus sample import.
 * Notes  :
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif
#include "../common/misc_util.h"
#include "Tagging.h"
#include "Loaders.h"
#include "ChunkReader.h"
#include "modsmp_ctrl.h"
#include "../soundbase/SampleFormatConverters.h"
#include "../soundbase/SampleFormatCopy.h"
#include "../soundlib/ModSampleCopy.h"
//#include "../common/mptCRC.h"
#include "OggStream.h"
#ifdef MPT_WITH_OGG
#include <ogg/ogg.h>
#endif // MPT_WITH_OGG
#if defined(MPT_WITH_OPUSFILE)
#include <opusfile.h>
#endif


OPENMPT_NAMESPACE_BEGIN


////////////////////////////////////////////////////////////////////////////////
// Opus

#if defined(MPT_WITH_OPUSFILE)

static mpt::ustring UStringFromOpus(const char *str)
{
	return str ? mpt::ToUnicode(mpt::CharsetUTF8, str) : mpt::ustring();
}

static FileTags GetOpusFileTags(OggOpusFile *of)
{
	FileTags tags;
	const OpusTags *ot = op_tags(of, -1);
	if(!ot)
	{
		return tags;
	}
	tags.encoder = UStringFromOpus(opus_tags_query(ot, "ENCODER", 0));
	tags.title = UStringFromOpus(opus_tags_query(ot, "TITLE", 0));
	tags.comments = UStringFromOpus(opus_tags_query(ot, "DESCRIPTION", 0));
	tags.bpm = UStringFromOpus(opus_tags_query(ot, "BPM", 0)); // non-standard
	tags.artist = UStringFromOpus(opus_tags_query(ot, "ARTIST", 0));
	tags.album = UStringFromOpus(opus_tags_query(ot, "ALBUM", 0));
	tags.trackno = UStringFromOpus(opus_tags_query(ot, "TRACKNUMBER", 0));
	tags.year = UStringFromOpus(opus_tags_query(ot, "DATE", 0));
	tags.url = UStringFromOpus(opus_tags_query(ot, "CONTACT", 0));
	tags.genre = UStringFromOpus(opus_tags_query(ot, "GENRE", 0));
	return tags;
}

#endif // MPT_WITH_OPUSFILE

bool CSoundFile::ReadOpusSample(SAMPLEINDEX sample, FileReader &file)
{
	file.Rewind();

#if defined(MPT_WITH_OPUSFILE)

	int rate = 0;
	int channels = 0;
	std::vector<int16> raw_sample_data;

	FileReader initial = file.GetChunk(65536); // 512 is recommended by libopusfile
	if(op_test(NULL, initial.GetRawData<unsigned char>(), initial.GetLength()) != 0)
	{
		return false;
	}

	OggOpusFile *of = op_open_memory(file.GetRawData<unsigned char>(), file.GetLength(), NULL);
	if(!of)
	{
		return false;
	}

	rate = 48000;
	channels = op_channel_count(of, -1);
	if(rate <= 0 || channels <= 0)
	{
		op_free(of);
		of = NULL;
		return false;
	}
	if(channels > 2 || op_link_count(of) != 1)
	{
		// We downmix multichannel to stereo as recommended by Opus specification in
		// case we are not able to handle > 2 channels.
		// We also decode chained files as stereo even if they start with a mono
		// stream, which simplifies handling of link boundaries for us.
		channels = 2;
	}

	std::vector<int16> decodeBuf(120 * 48000 / 1000); // 120ms (max Opus packet), 48kHz
	bool eof = false;
	while(!eof)
	{
		int framesRead = 0;
		if(channels == 2)
		{
			framesRead = op_read_stereo(of, &(decodeBuf[0]), static_cast<int>(decodeBuf.size()));
		} else if(channels == 1)
		{
			framesRead = op_read(of, &(decodeBuf[0]), static_cast<int>(decodeBuf.size()), NULL);
		}
		if(framesRead > 0)
		{
			raw_sample_data.insert(raw_sample_data.end(), decodeBuf.begin(), decodeBuf.begin() + (framesRead * channels));
		} else if(framesRead == 0)
		{
			eof = true;
		} else if(framesRead == OP_HOLE)
		{
			// continue
		} else
		{
			// other errors are fatal, stop decoding
			eof = true;
		}
	}

	op_free(of);
	of = NULL;

	if(raw_sample_data.empty())
	{
		return false;
	}

	if((raw_sample_data.size() / channels) > MAX_SAMPLE_LENGTH)
	{
		return false;
	}

	DestroySampleThreadsafe(sample);
	strcpy(m_szNames[sample], "");
	Samples[sample].Initialize();
	Samples[sample].nC5Speed = rate;
	Samples[sample].nLength = mpt::saturate_cast<SmpLength>(raw_sample_data.size() / channels);

	Samples[sample].uFlags.set(CHN_16BIT);
	Samples[sample].uFlags.set(CHN_STEREO, channels == 2);

	if(!Samples[sample].AllocateSample())
	{
		return false;
	}

	std::copy(raw_sample_data.begin(), raw_sample_data.end(), Samples[sample].pSample16);

	Samples[sample].Convert(MOD_TYPE_IT, GetType());
	Samples[sample].PrecomputeLoops(*this, false);

	return true;

#else // !MPT_WITH_OPUSFILE

	MPT_UNREFERENCED_PARAMETER(sample);
	MPT_UNREFERENCED_PARAMETER(file);

	return false;

#endif // MPT_WITH_OPUSFILE

}


OPENMPT_NAMESPACE_END
