/*
 * SampleFormatVorbis.cpp
 * ----------------------
 * Purpose: Vorbis sample import
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
#if defined(MPT_WITH_VORBIS)
#include <vorbis/codec.h>
#endif
#if defined(MPT_WITH_VORBISFILE)
#include <vorbis/vorbisfile.h>
#endif
#ifdef MPT_WITH_STBVORBIS
#include <stb_vorbis/stb_vorbis.c>
#endif // MPT_WITH_STBVORBIS


OPENMPT_NAMESPACE_BEGIN


////////////////////////////////////////////////////////////////////////////////
// Vorbis

#if defined(MPT_WITH_VORBISFILE)

static size_t VorbisfileFilereaderRead(void *ptr, size_t size, size_t nmemb, void *datasource)
{
	FileReader &file = *reinterpret_cast<FileReader*>(datasource);
	return file.ReadRaw(mpt::void_cast<mpt::byte*>(ptr), size * nmemb) / size;
}

static int VorbisfileFilereaderSeek(void *datasource, ogg_int64_t offset, int whence)
{
	FileReader &file = *reinterpret_cast<FileReader*>(datasource);
	switch(whence)
	{
	case SEEK_SET:
		{
			if(!Util::TypeCanHoldValue<FileReader::off_t>(offset))
			{
				return -1;
			}
			return file.Seek(mpt::saturate_cast<FileReader::off_t>(offset)) ? 0 : -1;
		}
		break;
	case SEEK_CUR:
		{
			if(offset < 0)
			{
				if(offset == std::numeric_limits<ogg_int64_t>::min())
				{
					return -1;
				}
				if(!Util::TypeCanHoldValue<FileReader::off_t>(0-offset))
				{
					return -1;
				}
				return file.SkipBack(mpt::saturate_cast<FileReader::off_t>(0 - offset)) ? 0 : -1;
			} else
			{
				if(!Util::TypeCanHoldValue<FileReader::off_t>(offset))
				{
					return -1;
				}
				return file.Skip(mpt::saturate_cast<FileReader::off_t>(offset)) ? 0 : -1;
			}
		}
		break;
	case SEEK_END:
		{
			if(!Util::TypeCanHoldValue<FileReader::off_t>(offset))
			{
				return -1;
			}
			if(!Util::TypeCanHoldValue<FileReader::off_t>(file.GetLength() + offset))
			{
				return -1;
			}
			return file.Seek(mpt::saturate_cast<FileReader::off_t>(file.GetLength() + offset)) ? 0 : -1;
		}
		break;
	default:
		return -1;
	}
}

static long VorbisfileFilereaderTell(void *datasource)
{
	FileReader &file = *reinterpret_cast<FileReader*>(datasource);
	MPT_MAYBE_CONSTANT_IF(!Util::TypeCanHoldValue<long>(file.GetPosition()))
	{
		return -1;
	}
	return static_cast<long>(file.GetPosition());
}

#if defined(MPT_WITH_VORBIS)
static mpt::ustring UStringFromVorbis(const char *str)
{
	return str ? mpt::ToUnicode(mpt::CharsetUTF8, str) : mpt::ustring();
}
#endif // MPT_WITH_VORBIS

static FileTags GetVorbisFileTags(OggVorbis_File &vf)
{
	FileTags tags;
	#if defined(MPT_WITH_VORBIS)
		vorbis_comment *vc = ov_comment(&vf, -1);
		if(!vc)
		{
			return tags;
		}
		tags.encoder = UStringFromVorbis(vorbis_comment_query(vc, "ENCODER", 0));
		tags.title = UStringFromVorbis(vorbis_comment_query(vc, "TITLE", 0));
		tags.comments = UStringFromVorbis(vorbis_comment_query(vc, "DESCRIPTION", 0));
		tags.bpm = UStringFromVorbis(vorbis_comment_query(vc, "BPM", 0)); // non-standard
		tags.artist = UStringFromVorbis(vorbis_comment_query(vc, "ARTIST", 0));
		tags.album = UStringFromVorbis(vorbis_comment_query(vc, "ALBUM", 0));
		tags.trackno = UStringFromVorbis(vorbis_comment_query(vc, "TRACKNUMBER", 0));
		tags.year = UStringFromVorbis(vorbis_comment_query(vc, "DATE", 0));
		tags.url = UStringFromVorbis(vorbis_comment_query(vc, "CONTACT", 0));
		tags.genre = UStringFromVorbis(vorbis_comment_query(vc, "GENRE", 0));
	#else // !MPT_WITH_VORBIS
		MPT_UNREFERENCED_PARAMETER(vf);
	#endif // MPT_WITH_VORBIS
	return tags;
}

#endif // MPT_WITH_VORBISFILE

bool CSoundFile::ReadVorbisSample(SAMPLEINDEX sample, FileReader &file)
{

#if defined(MPT_WITH_VORBISFILE) || defined(MPT_WITH_STBVORBIS)

	file.Rewind();

	int rate = 0;
	int channels = 0;
	std::vector<int16> raw_sample_data;

	std::string sampleName;

#endif // VORBIS

#if defined(MPT_WITH_VORBISFILE)

	bool unsupportedSample = false;

	ov_callbacks callbacks = {
		&VorbisfileFilereaderRead,
		&VorbisfileFilereaderSeek,
		NULL,
		&VorbisfileFilereaderTell
	};
	OggVorbis_File vf;
	MemsetZero(vf);
	if(ov_open_callbacks(&file, &vf, NULL, 0, callbacks) == 0)
	{
		if(ov_streams(&vf) == 1)
		{ // we do not support chained vorbis samples
			vorbis_info *vi = ov_info(&vf, -1);
			if(vi && vi->rate > 0 && vi->channels > 0)
			{
				sampleName = mpt::ToCharset(GetCharsetInternal(), GetSampleNameFromTags(GetVorbisFileTags(vf)));
				rate = vi->rate;
				channels = vi->channels;
				std::size_t offset = 0;
				int current_section = 0;
				long decodedSamples = 0;
				bool eof = false;
				while(!eof)
				{
					float **output = nullptr;
					long ret = ov_read_float(&vf, &output, 1024, &current_section);
					if(ret == 0)
					{
						eof = true;
					} else if(ret < 0)
					{
						// stream error, just try to continue
					} else
					{
						decodedSamples = ret;
						if(decodedSamples > 0 && (channels == 1 || channels == 2))
						{
							raw_sample_data.resize(raw_sample_data.size() + (channels * decodedSamples));
							for(int chn = 0; chn < channels; chn++)
							{
								CopyChannelToInterleaved<SC::Convert<int16, float> >(&(raw_sample_data[0]) + offset * channels, output[chn], channels, decodedSamples, chn);
							}
							offset += decodedSamples;
						}
					}
				}
			} else
			{
				unsupportedSample = true;
			}
		} else
		{
			unsupportedSample = true;
		}
		ov_clear(&vf);
	} else
	{
		unsupportedSample = true;
	}

	if(unsupportedSample)
	{
		return false;
	}

#elif defined(MPT_WITH_STBVORBIS)

	// NOTE/TODO: stb_vorbis does not handle inferred negative PCM sample position
	// at stream start. (See
	// <https://www.xiph.org/vorbis/doc/Vorbis_I_spec.html#x1-132000A.2>). This
	// means that, for remuxed and re-aligned/cutted (at stream start) Vorbis
	// files, stb_vorbis will include superfluous samples at the beginning.

	FileReader::PinnedRawDataView fileView = file.GetPinnedRawDataView();
	const mpt::byte* data = fileView.data();
	std::size_t dataLeft = fileView.size();

	std::size_t offset = 0;
	int consumed = 0;
	int error = 0;
	stb_vorbis *vorb = stb_vorbis_open_pushdata(data, mpt::saturate_cast<int>(dataLeft), &consumed, &error, nullptr);
	file.Skip(consumed);
	data += consumed;
	dataLeft -= consumed;
	if(!vorb)
	{
		return false;
	}
	rate = stb_vorbis_get_info(vorb).sample_rate;
	channels = stb_vorbis_get_info(vorb).channels;
	if(rate <= 0 || channels <= 0)
	{
		return false;
	}
	while((error == VORBIS__no_error || (error == VORBIS_need_more_data && dataLeft > 0)))
	{
		int frame_channels = 0;
		int decodedSamples = 0;
		float **output = nullptr;
		consumed = stb_vorbis_decode_frame_pushdata(vorb, data, mpt::saturate_cast<int>(dataLeft), &frame_channels, &output, &decodedSamples);
		file.Skip(consumed);
		data += consumed;
		dataLeft -= consumed;
		LimitMax(frame_channels, channels);
		if(decodedSamples > 0 && (frame_channels == 1 || frame_channels == 2))
		{
			raw_sample_data.resize(raw_sample_data.size() + (channels * decodedSamples));
			for(int chn = 0; chn < frame_channels; chn++)
			{
				CopyChannelToInterleaved<SC::Convert<int16, float> >(&(raw_sample_data[0]) + offset * channels, output[chn], channels, decodedSamples, chn);
			}
			offset += decodedSamples;
		}
		error = stb_vorbis_get_error(vorb);
	}
	stb_vorbis_close(vorb);

#endif // VORBIS

#if defined(MPT_WITH_VORBISFILE) || defined(MPT_WITH_STBVORBIS)

	if(rate <= 0 || channels <= 0 || raw_sample_data.empty())
	{
		return false;
	}

	if((raw_sample_data.size() / channels) > MAX_SAMPLE_LENGTH)
	{
		return false;
	}

	DestroySampleThreadsafe(sample);
	mpt::String::Copy(m_szNames[sample], sampleName);
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

#else // !VORBIS

	MPT_UNREFERENCED_PARAMETER(sample);
	MPT_UNREFERENCED_PARAMETER(file);

	return false;

#endif // VORBIS

}


bool CSoundFile::CanReadVorbis()
{
	bool result = false;
	#if defined(MPT_WITH_OGG) && defined(MPT_WITH_VORBIS) && defined(MPT_WITH_VORBISFILE)
		if(!result)
		{
			result = true;
		}
	#endif
	#if defined(MPT_WITH_STBVORBIS)
		if(!result)
		{
			result = true;
		}
	#endif
	return result;
}


OPENMPT_NAMESPACE_END
