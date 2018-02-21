/*
 * Load_it.cpp
 * -----------
 * Purpose: IT (Impulse Tracker) module loader / saver
 * Notes  : Also handles MPTM loading / saving, as the formats are almost identical.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "tuningcollection.h"
#include "mod_specifications.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/Moddoc.h"
#include "../mptrack/TrackerSettings.h"
#endif // MODPLUG_TRACKER
#ifdef MPT_EXTERNAL_SAMPLES
#include "../common/mptPathString.h"
#endif // MPT_EXTERNAL_SAMPLES
#include "../common/mptIO.h"
#include "../common/serialization_utils.h"
#ifndef MODPLUG_NO_FILESAVE
#include "../common/mptFileIO.h"
#endif // MODPLUG_NO_FILESAVE
#include "plugins/PlugInterface.h"
#include "../common/mptBufferIO.h"
#include "../common/version.h"
#include "ITTools.h"


OPENMPT_NAMESPACE_BEGIN

#define str_tooMuchPatternData	("Warning: File format limit was reached. Some pattern data may not get written to file.")
#define str_pattern				("pattern")
#define str_PatternSetTruncationNote ("The module contains %1 patterns but only %2 patterns can be loaded in this OpenMPT version.")
#define str_LoadingIncompatibleVersion	"The file informed that it is incompatible with this version of OpenMPT. Loading was terminated."
#define str_LoadingMoreRecentVersion	"The loaded file was made with a more recent OpenMPT version and this version may not be able to load all the features or play the file correctly."

const uint16 verMptFileVer = 0x891;
const uint16 verMptFileVerLoadLimit = 0x1000; // If cwtv-field is greater or equal to this value,
											  // the MPTM file will not be loaded.

/*
MPTM version history for cwtv-field in "IT" header (only for MPTM files!):
0x890(1.18.02.00) -> 0x891(1.19.00.00): Pattern-specific time signatures
										Fixed behaviour of Pattern Loop command for rows > 255 (r617)
0x88F(1.18.01.00) -> 0x890(1.18.02.00): Removed volume command velocity :xy, added delay-cut command :xy.
0x88E(1.17.02.50) -> 0x88F(1.18.01.00): Numerous changes
0x88D(1.17.02.49) -> 0x88E(1.17.02.50): Changed ID to that of IT and undone the orderlist change done in
				       0x88A->0x88B. Now extended orderlist is saved as extension.
0x88C(1.17.02.48) -> 0x88D(1.17.02.49): Some tuning related changes - that part fails to read on older versions.
0x88B -> 0x88C: Changed type in which tuning number is printed to file: size_t -> uint16.
0x88A -> 0x88B: Changed order-to-pattern-index table type from uint8-array to vector<uint32>.
*/


#ifndef MODPLUG_NO_FILESAVE

static bool AreNonDefaultTuningsUsed(CSoundFile& sf)
{
	const INSTRUMENTINDEX iCount = sf.GetNumInstruments();
	for(INSTRUMENTINDEX i = 1; i <= iCount; i++)
	{
		if(sf.Instruments[i] != nullptr && sf.Instruments[i]->pTuning != 0)
			return true;
	}
	return false;
}

static void WriteTuningCollection(std::ostream& oStrm, const CTuningCollection& tc)
{
	tc.Serialize(oStrm, "Tune specific tunings");
}

static void WriteTuningMap(std::ostream& oStrm, const CSoundFile& sf)
{
	if(sf.GetNumInstruments() > 0)
	{
		//Writing instrument tuning data: first creating
		//tuning name <-> tuning id number map,
		//and then writing the tuning id for every instrument.
		//For example if there are 6 instruments and
		//first half use tuning 'T1', and the other half
		//tuning 'T2', the output would be something like
		//T1 1 T2 2 1 1 1 2 2 2

		//Creating the tuning address <-> tuning id number map.
		std::map<CTuning*, uint16> tNameToShort_Map;

		unsigned short figMap = 0;
		for(INSTRUMENTINDEX i = 1; i <= sf.GetNumInstruments(); i++)
		{
			CTuning *pTuning = nullptr;
			if(sf.Instruments[i] != nullptr)
			{
				pTuning = sf.Instruments[i]->pTuning;
			}
			auto iter = tNameToShort_Map.find(pTuning);
			if(iter != tNameToShort_Map.end())
				continue; //Tuning already mapped.

			tNameToShort_Map[pTuning] = figMap;
			figMap++;
		}

		//...and write the map with tuning names replacing
		//the addresses.
		const uint16 tuningMapSize = static_cast<uint16>(tNameToShort_Map.size());
		mpt::IO::WriteIntLE<uint16>(oStrm, tuningMapSize);
		for(auto &iter : tNameToShort_Map)
		{
			if(iter.first)
				mpt::IO::WriteSizedStringLE<uint8>(oStrm, iter.first->GetName());
			else //Case: Using original IT tuning.
				mpt::IO::WriteSizedStringLE<uint8>(oStrm, "->MPT_ORIGINAL_IT<-");

			mpt::IO::WriteIntLE<uint16>(oStrm, iter.second);
		}

		//Writing tuning data for instruments.
		for(INSTRUMENTINDEX i = 1; i <= sf.GetNumInstruments(); i++)
		{
			CTuning *pTuning = nullptr;
			if(sf.Instruments[i] != nullptr)
			{
				pTuning = sf.Instruments[i]->pTuning;
			}
			auto iter = tNameToShort_Map.find(pTuning);
			if(iter == tNameToShort_Map.end()) //Should never happen
			{
				sf.AddToLog("Error: 210807_1");
				return;
			}
			mpt::IO::WriteIntLE<uint16>(oStrm, iter->second);
		}
	}
}

#endif // MODPLUG_NO_FILESAVE


static void ReadTuningCollection(std::istream& iStrm, CTuningCollection& tc, const size_t)
{
	std::string name;
	tc.Deserialize(iStrm, name);
}


template<class TUNNUMTYPE, class STRSIZETYPE>
static bool ReadTuningMapTemplate(std::istream& iStrm, std::map<uint16, std::string>& shortToTNameMap, const size_t maxNum = 500)
{
	TUNNUMTYPE numTuning = 0;
	mpt::IO::ReadIntLE<TUNNUMTYPE>(iStrm, numTuning);
	if(numTuning > maxNum)
		return true;

	for(size_t i = 0; i < numTuning; i++)
	{
		std::string temp;
		uint16 ui = 0;
		if(!mpt::IO::ReadSizedStringLE<STRSIZETYPE>(iStrm, temp, 255))
			return true;

		mpt::IO::ReadIntLE<uint16>(iStrm, ui);
		shortToTNameMap[ui] = temp;
	}
	if(iStrm.good())
		return false;
	else
		return true;
}


static void ReadTuningMapImpl(std::istream& iStrm, CSoundFile& csf, const size_t = 0, bool old = false)
{
	std::map<uint16, std::string> shortToTNameMap;
	if(old)
	{
		ReadTuningMapTemplate<uint32, uint32>(iStrm, shortToTNameMap);
	} else
	{
		ReadTuningMapTemplate<uint16, uint8>(iStrm, shortToTNameMap);
	}

	// Read & set tunings for instruments
	std::vector<std::string> notFoundTunings;
	for(INSTRUMENTINDEX i = 1; i<=csf.GetNumInstruments(); i++)
	{
		uint16 ui = 0;
		mpt::IO::ReadIntLE<uint16>(iStrm, ui);
		auto iter = shortToTNameMap.find(ui);
		if(csf.Instruments[i] && iter != shortToTNameMap.end())
		{
			const std::string str = iter->second;

			if(str == "->MPT_ORIGINAL_IT<-")
			{
				csf.Instruments[i]->pTuning = nullptr;
				continue;
			}

			csf.Instruments[i]->pTuning = csf.GetTuneSpecificTunings().GetTuning(str);
			if(csf.Instruments[i]->pTuning)
				continue;

#ifdef MODPLUG_TRACKER
			CTuning *localTuning = TrackerSettings::Instance().oldLocalTunings->GetTuning(str);
			if(localTuning)
			{
				CTuning* pNewTuning = new CTuning(*localTuning);
				if(!csf.GetTuneSpecificTunings().AddTuning(pNewTuning))
				{
					csf.AddToLog("Local tunings are deprecated and no longer supported. Tuning '" + str + "' found in Local tunings has been copied to Tune-specific tunings and will be saved in the module file.");
					csf.Instruments[i]->pTuning = pNewTuning;
					if(csf.GetpModDoc() != nullptr)
					{
						csf.GetpModDoc()->SetModified();
					}
					continue;
				} else
				{
					delete pNewTuning;
					csf.AddToLog("Copying Local tuning '" + str + "' to Tune-specific tunings failed.");
				}
			}
#endif

			if(str == "12TET [[fs15 1.17.02.49]]" || str == "12TET")
			{
				CTuning* pNewTuning = csf.CreateTuning12TET(str);
				if(!csf.GetTuneSpecificTunings().AddTuning(pNewTuning))
				{
					#ifdef MODPLUG_TRACKER
						csf.AddToLog("Built-in tunings will no longer be used. Tuning '" + str + "' has been copied to Tune-specific tunings and will be saved in the module file.");
						csf.Instruments[i]->pTuning = pNewTuning;
						if(csf.GetpModDoc() != nullptr)
						{
							csf.GetpModDoc()->SetModified();
						}
					#endif
					continue;
				} else
				{
					delete pNewTuning;
					#ifdef MODPLUG_TRACKER
						csf.AddToLog("Copying Built-in tuning '" + str + "' to Tune-specific tunings failed.");
					#endif
				}
			}

			// Checking if not found tuning already noticed.
			if(std::find(notFoundTunings.begin(), notFoundTunings.end(), str) == notFoundTunings.end())
			{
				notFoundTunings.push_back(str);
				csf.AddToLog("Tuning '" + str + "' used by the module was not found.");
#ifdef MODPLUG_TRACKER
				if(csf.GetpModDoc() != nullptr)
				{
					csf.GetpModDoc()->SetModified(); // The tuning is changed so the modified flag is set.
				}
#endif // MODPLUG_TRACKER

			}
			csf.Instruments[i]->pTuning = csf.GetDefaultTuning();

		} else
		{
			//This 'else' happens probably only in case of corrupted file.
			if(csf.Instruments[i])
				csf.Instruments[i]->pTuning = csf.GetDefaultTuning();
		}

	}
	//End read&set instrument tunings
}


static void ReadTuningMap(std::istream& iStrm, CSoundFile& csf, const size_t dummy = 0)
{
	ReadTuningMapImpl(iStrm, csf, dummy, false);
}


//////////////////////////////////////////////////////////
// Impulse Tracker IT file support


size_t CSoundFile::ITInstrToMPT(FileReader &file, ModInstrument &ins, uint16 trkvers)
{
	if(trkvers < 0x0200)
	{
		// Load old format (IT 1.xx) instrument (early IT 2.xx modules may have cmwt set to 1.00 for backwards compatibility)
		ITOldInstrument instrumentHeader;
		if(!file.ReadStruct(instrumentHeader))
		{
			return 0;
		} else
		{
			instrumentHeader.ConvertToMPT(ins);
			return sizeof(ITOldInstrument);
		}
	} else
	{
		const FileReader::off_t offset = file.GetPosition();

		// Try loading extended instrument... instSize will differ between normal and extended instruments.
		ITInstrumentEx instrumentHeader;
		file.ReadStructPartial(instrumentHeader);
		size_t instSize = instrumentHeader.ConvertToMPT(ins, GetType());
		file.Seek(offset + instSize);

		// Try reading modular instrument data.
		// Yes, it is completely idiotic that we have both this and LoadExtendedInstrumentProperties.
		// This is only required for files saved with *really* old OpenMPT versions (pre-1.17-RC1).
		// This chunk was also written in later versions (probably to maintain compatibility with
		// those ancient versions), but this also means that redundant information is stored in the file.
		// Starting from OpenMPT 1.25.02.07, this chunk is no longer written.
		if(file.ReadMagic("MSNI"))
		{
			//...the next piece of data must be the total size of the modular data
			FileReader modularData = file.ReadChunk(file.ReadUint32LE());
			instSize += 8 + modularData.GetLength();
			if(modularData.ReadMagic("GULP"))
			{
				ins.nMixPlug = modularData.ReadUint8();
				if(ins.nMixPlug > MAX_MIXPLUGINS) ins.nMixPlug = 0;
			}
		}

		return instSize;
	}
}


static void CopyPatternName(CPattern &pattern, FileReader &file)
{
	char name[MAX_PATTERNNAME] = "";
	file.ReadString<mpt::String::maybeNullTerminated>(name, MAX_PATTERNNAME);
	pattern.SetName(name);
}


// Date calculation derived from https://alcor.concordia.ca/~gpkatch/gdate-algorithm.html
template<int32 y, int32 m, int32 d>
struct SchismVersionFromDate
{
	static const int32 mm = (m + 9) % 12;
	static const int32 yy = y - mm / 10;
	static const int32 date = yy * 365 + yy / 4 - yy / 100 + yy / 400 + (mm * 306 + 5) / 10 + (d - 1);

	static constexpr int32 Version()
	{
		return 0x1050 + date - SchismVersionFromDate<2009, 10, 31>::date;
	}
};


// Get version of Schism Tracker that was used to create an IT/S3M file.
mpt::ustring CSoundFile::GetSchismTrackerVersion(uint16 cwtv)
{
	// Schism Tracker version information in a nutshell:
	// < 0x020: a proper version (files saved by such versions are likely very rare)
	// = 0x020: any version between the 0.2a release (2005-04-29?) and 2007-04-17
	// = 0x050: anywhere from 2007-04-17 to 2009-10-31
	// > 0x050: the number of days since 2009-10-31

	cwtv &= 0xFFF;
	mpt::ustring version;
	if(cwtv > 0x050)
	{
		int32 date = SchismVersionFromDate<2009, 10, 31>::date + cwtv - 0x050;
		int32 y = static_cast<int32>((Util::mul32to64(10000, date) + 14780) / 3652425);
		int32 ddd = date - (365 * y + y / 4 - y / 100 + y / 400);
		if(ddd < 0)
		{
			y--;
			ddd = date - (365 * y + y / 4 - y / 100 + y / 400);
		}
		int32 mi = (100 * ddd + 52) / 3060;
		version = mpt::format(MPT_USTRING("Schism Tracker %1-%2-%3"))(
			mpt::ufmt::dec0<4>(y + (mi + 2) / 12),
			mpt::ufmt::dec0<2>((mi + 2) % 12 + 1),
			mpt::ufmt::dec0<2>(ddd - (mi * 306 + 5) / 10 + 1));
	} else
	{
		version = mpt::format(MPT_USTRING("Schism Tracker 0.%1"))(mpt::ufmt::hex(cwtv));
	}
	return version;
}


static bool ValidateHeader(const ITFileHeader &fileHeader)
{
	if((std::memcmp(fileHeader.id, "IMPM", 4) && std::memcmp(fileHeader.id, "tpm.", 4))
		|| fileHeader.insnum > 0xFF
		|| fileHeader.smpnum >= MAX_SAMPLES
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const ITFileHeader &fileHeader)
{
	return fileHeader.ordnum + (fileHeader.insnum + fileHeader.smpnum + fileHeader.patnum) * 4;
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderIT(MemoryFileReader file, const uint64 *pfilesize)
{
	ITFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(fileHeader))
	{
		return ProbeFailure;
	}
	return ProbeAdditionalSize(file, pfilesize, GetHeaderMinimumAdditionalSize(fileHeader));
}


bool CSoundFile::ReadIT(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();

	ITFileHeader fileHeader;
	if(!file.ReadStruct(fileHeader))
	{
		return false;
	}
	if(!ValidateHeader(fileHeader))
	{
		return false;
	}
	if(!file.CanRead(mpt::saturate_cast<FileReader::off_t>(GetHeaderMinimumAdditionalSize(fileHeader))))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	InitializeGlobals(MOD_TYPE_IT);

	bool interpretModPlugMade = false;

	// OpenMPT crap at the end of file
	size_t mptStartPos = 0;

	if(!memcmp(fileHeader.id, "tpm.", 4))
	{
		// Legacy MPTM files (old 1.17.02.4x releases)
		SetType(MOD_TYPE_MPT);
		file.Seek(file.GetLength() - 4);
		mptStartPos = file.ReadUint32LE();
	} else
	{
		if(fileHeader.cwtv > 0x888 && fileHeader.cwtv <= 0xFFF)
		{
			file.Seek(file.GetLength() - 4);
			mptStartPos = file.ReadUint32LE();
			if(mptStartPos >= 0x100 && mptStartPos < file.GetLength())
			{
				if(file.Seek(mptStartPos) && file.ReadMagic("228"))
				{
					SetType(MOD_TYPE_MPT);

					if(fileHeader.cwtv >= verMptFileVerLoadLimit)
					{
						AddToLog(str_LoadingIncompatibleVersion);
						return false;
					} else if(fileHeader.cwtv > verMptFileVer)
					{
						AddToLog(str_LoadingMoreRecentVersion);
					}
				}
			}
		}

		if(GetType() == MOD_TYPE_IT)
		{
			// Which tracker was used to make this?
			if((fileHeader.cwtv & 0xF000) == 0x5000)
			{
				// OpenMPT Version number (Major.Minor)
				// This will only be interpreted as "made with ModPlug" (i.e. disable compatible playback etc) if the "reserved" field is set to "OMPT" - else, compatibility was used.
				m_dwLastSavedWithVersion = (fileHeader.cwtv & 0x0FFF) << 16;
				if(!memcmp(&fileHeader.reserved, "OMPT", 4))
					interpretModPlugMade = true;
			} else if(fileHeader.cmwt == 0x888 || fileHeader.cwtv == 0x888)
			{
				// OpenMPT 1.17.02.26 (r122) to 1.18 (raped IT format)
				// Exact version number will be determined later.
				interpretModPlugMade = true;
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 17, 00, 00);
			} else if(fileHeader.cwtv == 0x0217 && fileHeader.cmwt == 0x0200 && fileHeader.reserved == 0)
			{
				if(memchr(fileHeader.chnpan, 0xFF, sizeof(fileHeader.chnpan)) != nullptr)
				{
					// ModPlug Tracker 1.16 (semi-raped IT format) or BeRoTracker (will be determined later)
					m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 16, 00, 00);
					m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.09 - 1.16");
				} else
				{
					// OpenMPT 1.17 disguised as this in compatible mode,
					// but never writes 0xFF in the pan map for unused channels (which is an invalid value).
					m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 17, 00, 00);
					m_madeWithTracker = MPT_USTRING("OpenMPT 1.17 (compatibility export)");
				}
				interpretModPlugMade = true;
			} else if(fileHeader.cwtv == 0x0214 && fileHeader.cmwt == 0x0202 && fileHeader.reserved == 0)
			{
				// ModPlug Tracker b3.3 - 1.09, instruments 557 bytes apart
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 09, 00, 00);
				m_madeWithTracker = MPT_USTRING("ModPlug Tracker b3.3 - 1.09");
				interpretModPlugMade = true;
			} else if(fileHeader.cwtv == 0x0300 && fileHeader.cmwt == 0x0300 && fileHeader.reserved == 0 && fileHeader.ordnum == 256 && fileHeader.sep == 128 && fileHeader.pwd == 0)
			{
				// A rare variant used from OpenMPT 1.17.02.20 (r113) to 1.17.02.25 (r121), found e.g. in xTr1m-SD.it
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 17, 02, 20);
				interpretModPlugMade = true;
			}
		}
	}

	m_SongFlags.set(SONG_LINEARSLIDES, (fileHeader.flags & ITFileHeader::linearSlides) != 0);
	m_SongFlags.set(SONG_ITOLDEFFECTS, (fileHeader.flags & ITFileHeader::itOldEffects) != 0);
	m_SongFlags.set(SONG_ITCOMPATGXX, (fileHeader.flags & ITFileHeader::itCompatGxx) != 0);
	m_SongFlags.set(SONG_EXFILTERRANGE, (fileHeader.flags & ITFileHeader::extendedFilterRange) != 0);

	mpt::String::Read<mpt::String::spacePadded>(m_songName, fileHeader.songname);

	// Read row highlights
	if((fileHeader.special & ITFileHeader::embedPatternHighlights))
	{
		// MPT 1.09 and older (and maybe also newer) versions leave this blank (0/0), but have the "special" flag set.
		// Newer versions of MPT and OpenMPT 1.17 *always* write 4/16 here.
		// Thus, we will just ignore those old versions.
		// Note: OpenMPT 1.17.03.02 was the first version to properly make use of the time signature in the IT header.
		// This poses a small unsolvable problem:
		// - In compatible mode, we cannot distinguish this version from earlier 1.17 releases.
		//   Thus we cannot know when to read this field or not (m_dwLastSavedWithVersion will always be 1.17.00.00).
		//   Luckily OpenMPT 1.17.03.02 should not be very wide-spread.
		// - In normal mode the time signature is always present in the song extensions anyway. So it's okay if we read
		//   the signature here and maybe overwrite it later when parsing the song extensions.
		if(m_dwLastSavedWithVersion == 0 || m_dwLastSavedWithVersion >= MAKE_VERSION_NUMERIC(1, 17, 03, 02))
		{
			m_nDefaultRowsPerBeat = fileHeader.highlight_minor;
			m_nDefaultRowsPerMeasure = fileHeader.highlight_major;
		}
	}

	// Global Volume
	m_nDefaultGlobalVolume = fileHeader.globalvol << 1;
	if(m_nDefaultGlobalVolume > MAX_GLOBAL_VOLUME) m_nDefaultGlobalVolume = MAX_GLOBAL_VOLUME;
	if(fileHeader.speed) m_nDefaultSpeed = fileHeader.speed;
	m_nDefaultTempo.Set(std::max<uint8>(31, fileHeader.tempo));
	m_nSamplePreAmp = std::min<uint8>(fileHeader.mv, 128);

	// Reading Channels Pan Positions
	for(CHANNELINDEX i = 0; i < 64; i++) if(fileHeader.chnpan[i] != 0xFF)
	{
		ChnSettings[i].Reset();
		ChnSettings[i].nVolume = Clamp<uint8, uint8>(fileHeader.chnvol[i], 0, 64);
		if(fileHeader.chnpan[i] & 0x80) ChnSettings[i].dwFlags.set(CHN_MUTE);
		uint8 n = fileHeader.chnpan[i] & 0x7F;
		if(n <= 64) ChnSettings[i].nPan = n * 4;
		if(n == 100) ChnSettings[i].dwFlags.set(CHN_SURROUND);
	}

	// Reading orders
	file.Seek(sizeof(ITFileHeader));
	if(GetType() == MOD_TYPE_MPT && fileHeader.cwtv > 0x88A && fileHeader.cwtv <= 0x88D)
	{
		// Deprecated format used for MPTm files created with OpenMPT 1.17.02.46 - 1.17.02.48.
		uint16 version = file.ReadUint16LE();
		if(version != 0)
			return false;
		uint32 numOrd = file.ReadUint32LE();
		if(numOrd > ModSpecs::mptm.ordersMax || !ReadOrderFromFile<uint32le>(Order(), file, numOrd))
			return false;
	} else
	{
		ReadOrderFromFile<uint8>(Order(), file, fileHeader.ordnum, 0xFF, 0xFE);
	}

	// Reading instrument, sample and pattern offsets
	std::vector<uint32le> insPos, smpPos, patPos;
	if(!file.ReadVector(insPos, fileHeader.insnum)
		|| !file.ReadVector(smpPos, fileHeader.smpnum)
		|| !file.ReadVector(patPos, fileHeader.patnum))
	{
		return false;
	}

	// Find the first parapointer.
	// This is used for finding out whether the edit history is actually stored in the file or not,
	// as some early versions of Schism Tracker set the history flag, but didn't save anything.
	// We will consider the history invalid if it ends after the first parapointer.
	uint32 minPtr = Util::MaxValueOfType(minPtr);
	for(uint32 pos : insPos)
	{
		if(pos > 0) minPtr = std::min(minPtr, pos);
	}
	for(uint32 pos : smpPos)
	{
		if(pos > 0) minPtr = std::min(minPtr, pos);
	}
	for(uint32 pos : patPos)
	{
		if(pos > 0) minPtr = std::min(minPtr, pos);
	}
	if(fileHeader.special & ITFileHeader::embedSongMessage)
	{
		minPtr = std::min<uint32>(minPtr, fileHeader.msgoffset);
	}

	const bool possiblyUNMO3 = fileHeader.cmwt == 0x0214 && fileHeader.cwtv == 0x0214 && fileHeader.highlight_major == 0 && fileHeader.highlight_minor == 0
		&& fileHeader.pwd == 0 && fileHeader.reserved == 0 && (fileHeader.flags & (ITFileHeader::useMIDIPitchController | ITFileHeader::reqEmbeddedMIDIConfig)) == 0;

	if(possiblyUNMO3 && fileHeader.insnum == 0 && fileHeader.smpnum > 0 && file.GetPosition() + 4 * smpPos.size() + 2 <= minPtr)
	{
		// UNMO3 < v2.4.0.1 reserves some space for instrument parapointers even in sample mode.
		// This makes reading MIDI macros and plugin information impossible.
		// Note: While UNMO3 and CheeseTracker header fingerprints are almost identical, we cannot mis-detect CheeseTracker here,
		// as it always sets the instrument mode flag and writes non-zero row highlights.
		bool oldUNMO3 = true;
		for(uint16 i = 0; i < fileHeader.smpnum; i++)
		{
			if(file.ReadUint32LE() != 0)
			{
				oldUNMO3 = false;
				file.SkipBack(4 + i * 4);
				break;
			}
		}
		if(oldUNMO3)
		{
			m_madeWithTracker = MPT_USTRING("UNMO3 <= 2.4");
		}
	}

	// Reading IT Edit History Info
	// This is only supposed to be present if bit 1 of the special flags is set.
	// However, old versions of Schism and probably other trackers always set this bit
	// even if they don't write the edit history count. So we have to filter this out...
	// This is done by looking at the parapointers. If the history data ends after
	// the first parapointer, we assume that it's actually no history data.
	if(fileHeader.special & ITFileHeader::embedEditHistory)
	{
		const uint16 nflt = file.ReadUint16LE();

		if(file.CanRead(nflt * sizeof(ITHistoryStruct)) && file.GetPosition() + nflt * sizeof(ITHistoryStruct) <= minPtr)
		{
			m_FileHistory.resize(nflt);
			for(auto &mptHistory : m_FileHistory)
			{
				ITHistoryStruct itHistory;
				file.ReadStruct(itHistory);
				itHistory.ConvertToMPT(mptHistory);
			}

			if(possiblyUNMO3 && nflt == 0)
			{
				if(fileHeader.special & ITFileHeader::embedPatternHighlights)
					m_madeWithTracker = MPT_USTRING("UNMO3 <= 2.4.0.1");	// Set together with MIDI macro embed flag
				else
					m_madeWithTracker = MPT_USTRING("UNMO3");	// Either 2.4.0.2+ or no MIDI macros embedded
			}
		} else
		{
			// Oops, we were not supposed to read this.
			file.SkipBack(2);
		}
	} else if(possiblyUNMO3 && fileHeader.special <= 1)
	{
		// UNMO3 < v2.4.0.1 will set the edit history special bit iff the MIDI macro embed bit is also set,
		// but it always writes the two extra bytes for the edit history length (zeroes).
		// If MIDI macros are embedded, we are fine and end up in the first case of the if statement (read edit history).
		// Otherwise we end up here and might have to read the edit history length.
		if(file.ReadUint16LE() == 0)
		{
			m_madeWithTracker = MPT_USTRING("UNMO3 <= 2.4");
		} else
		{
			// These were not zero bytes, but potentially belong to the upcoming MIDI config - need to skip back.
			// I think the only application that could end up here is CheeseTracker, if it allows to write 0 for both row highlight values.
			// IT 2.14 itself will always write the edit history.
			file.SkipBack(2);
		}
	}

	// Reading MIDI Output & Macros
	bool hasMidiConfig = (fileHeader.flags & ITFileHeader::reqEmbeddedMIDIConfig) || (fileHeader.special & ITFileHeader::embedMIDIConfiguration);
	if(hasMidiConfig && file.ReadStruct<MIDIMacroConfigData>(m_MidiCfg))
	{
		m_MidiCfg.Sanitize();
	}

	// Ignore MIDI data. Fixes some files like denonde.it that were made with old versions of Impulse Tracker (which didn't support Zxx filters) and have Zxx effects in the patterns.
	if(fileHeader.cwtv < 0x0214)
	{
		m_MidiCfg.ClearZxxMacros();
	}

	// Read pattern names: "PNAM"
	FileReader patNames;
	if(file.ReadMagic("PNAM"))
	{
		patNames = file.ReadChunk(file.ReadUint32LE());
	}

	m_nChannels = 1;
	// Read channel names: "CNAM"
	if(file.ReadMagic("CNAM"))
	{
		FileReader chnNames = file.ReadChunk(file.ReadUint32LE());
		const CHANNELINDEX readChns = std::min(MAX_BASECHANNELS, static_cast<CHANNELINDEX>(chnNames.GetLength() / MAX_CHANNELNAME));
		m_nChannels = readChns;

		for(CHANNELINDEX i = 0; i < readChns; i++)
		{
			chnNames.ReadString<mpt::String::maybeNullTerminated>(ChnSettings[i].szName, MAX_CHANNELNAME);
		}
	}

	// Read mix plugins information
	FileReader pluginChunk = file.ReadChunk((minPtr >= file.GetPosition()) ? minPtr - file.GetPosition() : file.BytesLeft());
	LoadMixPlugins(pluginChunk);

	// Read Song Message
	if((fileHeader.special & ITFileHeader::embedSongMessage) && fileHeader.msglength > 0 && file.Seek(fileHeader.msgoffset))
	{
		// Generally, IT files should use CR for line endings. However, ChibiTracker uses LF. One could do...
		// if(itHeader.cwtv == 0x0214 && itHeader.cmwt == 0x0214 && itHeader.reserved == ITFileHeader::chibiMagic) --> Chibi detected.
		// But we'll just use autodetection here:
		m_songMessage.Read(file, fileHeader.msglength, SongMessage::leAutodetect);
	}

	// Reading Instruments
	m_nInstruments = 0;
	if(fileHeader.flags & ITFileHeader::instrumentMode)
	{
		m_nInstruments = std::min<INSTRUMENTINDEX>(fileHeader.insnum, MAX_INSTRUMENTS - 1);
	}
	for(INSTRUMENTINDEX i = 0; i < GetNumInstruments(); i++)
	{
		if(insPos[i] > 0 && file.Seek(insPos[i]) && file.CanRead(fileHeader.cmwt < 0x200 ? sizeof(ITOldInstrument) : sizeof(ITInstrument)))
		{
			ModInstrument *instrument = AllocateInstrument(i + 1);
			if(instrument != nullptr)
			{
				ITInstrToMPT(file, *instrument, fileHeader.cmwt);
				// MIDI Pitch Wheel Depth is a global setting in IT. Apply it to all instruments.
				instrument->midiPWD = fileHeader.pwd;
			}
		}
	}

	// In order to properly compute the position, in file, of eventual extended settings
	// such as "attack" we need to keep the "real" size of the last sample as those extra
	// setting will follow this sample in the file
	FileReader::off_t lastSampleOffset = 0;
	if(fileHeader.smpnum > 0)
	{
		lastSampleOffset = smpPos[fileHeader.smpnum - 1] + sizeof(ITSample);
	}

	bool possibleXMconversion = false;

	// Reading Samples
	m_nSamples = std::min<SAMPLEINDEX>(fileHeader.smpnum, MAX_SAMPLES - 1);
	bool lastSampleCompressed = false;
	for(SAMPLEINDEX i = 0; i < GetNumSamples(); i++)
	{
		ITSample sampleHeader;
		if(smpPos[i] > 0 && file.Seek(smpPos[i]) && file.ReadStruct(sampleHeader))
		{
			// IT does not check for the IMPS magic, and some bad XM->IT converter out there doesn't write the magic bytes for empty sample slots.
			ModSample &sample = Samples[i + 1];
			size_t sampleOffset = sampleHeader.ConvertToMPT(sample);

			mpt::String::Read<mpt::String::spacePadded>(m_szNames[i + 1], sampleHeader.name);

			if(!file.Seek(sampleOffset))
				continue;

			lastSampleCompressed = false;
			if(!sample.uFlags[SMP_KEEPONDISK])
			{
				SampleIO sampleIO = sampleHeader.GetSampleFormat(fileHeader.cwtv);
				if(loadFlags & loadSampleData)
				{
					sampleIO.ReadSample(sample, file);
				} else
				{
					if(sampleIO.IsVariableLengthEncoded())
						lastSampleCompressed = true;
					else
						file.Skip(sampleIO.CalculateEncodedSize(sample.nLength));
				}
				if(sampleIO.GetEncoding() == SampleIO::unsignedPCM && sample.nLength != 0)
				{
					// There is some XM to IT converter (don't know which one) and it identifies as IT 2.04.
					// The only safe way to distinguish it from an IT-saved file are the unsigned samples.
					possibleXMconversion = true;
				}
			} else
			{
				// External sample in MPTM file
				size_t strLen;
				file.ReadVarInt(strLen);
				if(loadFlags & loadSampleData)
				{
					std::string filenameU8;
					file.ReadString<mpt::String::maybeNullTerminated>(filenameU8, strLen);
#if defined(MPT_EXTERNAL_SAMPLES)
					SetSamplePath(i + 1, mpt::PathString::FromUTF8(filenameU8));
#elif !defined(LIBOPENMPT_BUILD_TEST)
					AddToLog(LogWarning, mpt::format(MPT_USTRING("Loading external sample %1 ('%2') failed: External samples are not supported."))(i, mpt::ToUnicode(mpt::CharsetUTF8, filenameU8)));
#endif // MPT_EXTERNAL_SAMPLES
				} else
				{
					file.Skip(strLen);
				}
			}
			lastSampleOffset = std::max(lastSampleOffset, file.GetPosition());
		}
	}
	m_nSamples = std::max(SAMPLEINDEX(1), GetNumSamples());

	if(possibleXMconversion && fileHeader.cwtv == 0x0204 && fileHeader.cmwt == 0x0200 && fileHeader.special == 0 && fileHeader.reserved == 0
		&& (fileHeader.flags & ~ITFileHeader::linearSlides) == (ITFileHeader::useStereoPlayback | ITFileHeader::instrumentMode | ITFileHeader::itOldEffects)
		&& fileHeader.globalvol == 128 && fileHeader.mv == 48 && fileHeader.sep == 128 && fileHeader.pwd == 0 && fileHeader.msglength == 0)
	{
		for(uint8 pan : fileHeader.chnpan)
		{
			if(pan != 0x20 && pan != 0xA0)
				possibleXMconversion = false;
		}
		for(uint8 vol : fileHeader.chnvol)
		{
			if(vol != 0x40)
				possibleXMconversion = false;
		}
		for(size_t i = 20; i < mpt::size(fileHeader.songname); i++)
		{
			if(fileHeader.songname[i] != 0)
				possibleXMconversion = false;
		}
		if(possibleXMconversion)
			m_madeWithTracker = MPT_USTRING("XM Conversion");
	}

	m_nMinPeriod = 0;
	m_nMaxPeriod = int32_max;

	PATTERNINDEX numPats = std::min(static_cast<PATTERNINDEX>(patPos.size()), GetModSpecifications().patternsMax);

	if(numPats != patPos.size())
	{
		// Hack: Notify user here if file contains more patterns than what can be read.
		AddToLog(mpt::format(str_PatternSetTruncationNote)(patPos.size(), numPats));
	}

	if(!(loadFlags & loadPatternData))
	{
		numPats = 0;
	}

	// Checking for number of used channels, which is not explicitely specified in the file.
	for(PATTERNINDEX pat = 0; pat < numPats; pat++)
	{
		if(patPos[pat] == 0 || !file.Seek(patPos[pat]))
			continue;

		uint16 len = file.ReadUint16LE();
		ROWINDEX numRows = file.ReadUint16LE();

		if(numRows < 1
			|| numRows > MAX_PATTERN_ROWS
			|| !file.Skip(4))
			continue;

		FileReader patternData = file.ReadChunk(len);
		ROWINDEX row = 0;
		std::vector<uint8> chnMask(GetNumChannels());

		while(row < numRows && patternData.CanRead(1))
		{
			uint8 b = patternData.ReadUint8();
			if(!b)
			{
				row++;
				continue;
			}

			CHANNELINDEX ch = (b & IT_bitmask_patternChanField_c);   // 0x7f We have some data grab a byte keeping only 7 bits
			if(ch)
			{
				ch = (ch - 1);// & IT_bitmask_patternChanMask_c;   // 0x3f mask of the byte again, keeping only 6 bits
			}

			if(ch >= chnMask.size())
			{
				chnMask.resize(ch + 1, 0);
			}

			if(b & IT_bitmask_patternChanEnabled_c)            // 0x80 check if the upper bit is enabled.
			{
				chnMask[ch] = patternData.ReadUint8();       // set the channel mask for this channel.
			}
			// Channel used
			if(chnMask[ch] & 0x0F)         // if this channel is used set m_nChannels
			{
				if(ch >= GetNumChannels() && ch < MAX_BASECHANNELS)
				{
					m_nChannels = ch + 1;
				}
			}
			// Now we actually update the pattern-row entry the note,instrument etc.
			// Note
			if(chnMask[ch] & 1) patternData.Skip(1);
			// Instrument
			if(chnMask[ch] & 2) patternData.Skip(1);
			// Volume
			if(chnMask[ch] & 4) patternData.Skip(1);
			// Effect
			if(chnMask[ch] & 8) patternData.Skip(2);
		}
		lastSampleOffset = std::max(lastSampleOffset, file.GetPosition());
	}

	// Compute extra instruments settings position
	if(lastSampleOffset > 0)
	{
		file.Seek(lastSampleOffset);
		if(lastSampleCompressed)
		{
			// If the last sample was compressed, we do not know where it ends.
			// Hence, in case we decided not to decode the sample data, we now
			// have to emulate this until we reach EOF or some instrument / song properties.
			while(file.CanRead(4))
			{
				if(file.ReadMagic("XTPM") || file.ReadMagic("STPM"))
				{
					uint32 id = file.ReadUint32LE();
					file.SkipBack(8);
					// Our chunk IDs should only contain ASCII characters
					if(!(id & 0x80808080) && (id & 0x60606060))
					{
						break;
					}
				}
				file.Skip(file.ReadUint16LE());
			}
		}
	}

	// Load instrument and song extensions.
	LoadExtendedInstrumentProperties(file, &interpretModPlugMade);
	if(interpretModPlugMade && m_madeWithTracker != MPT_USTRING("BeRoTracker"))
	{
		m_playBehaviour.reset();
		m_nMixLevels = mixLevelsOriginal;
	}
	// Need to do this before reading the patterns because m_nChannels might be modified by LoadExtendedSongProperties. *sigh*
	LoadExtendedSongProperties(file, &interpretModPlugMade);

	// Reading Patterns
	Patterns.ResizeArray(numPats);
	for(PATTERNINDEX pat = 0; pat < numPats; pat++)
	{
		if(patPos[pat] == 0 || !file.Seek(patPos[pat]))
		{
			// Empty 64-row pattern
			if(!Patterns.Insert(pat, 64))
			{
				AddToLog(mpt::format("Allocating patterns failed starting from pattern %1")(pat));
				break;
			}
			// Now (after the Insert() call), we can read the pattern name.
			CopyPatternName(Patterns[pat], patNames);
			continue;
		}

		uint16 len = file.ReadUint16LE();
		ROWINDEX numRows = file.ReadUint16LE();

		if(!file.Skip(4)
			|| !Patterns.Insert(pat, numRows))
			continue;
			
		FileReader patternData = file.ReadChunk(len);

		// Now (after the Insert() call), we can read the pattern name.
		CopyPatternName(Patterns[pat], patNames);

		std::vector<uint8> chnMask(GetNumChannels());
		std::vector<ModCommand> lastValue(GetNumChannels(), ModCommand::Empty());

		auto patData = Patterns[pat].begin();
		ROWINDEX row = 0;
		while(row < numRows && patternData.CanRead(1))
		{
			uint8 b = patternData.ReadUint8();
			if(!b)
			{
				row++;
				patData += GetNumChannels();
				continue;
			}

			CHANNELINDEX ch = b & IT_bitmask_patternChanField_c; // 0x7f

			if(ch)
			{
				ch = (ch - 1); //& IT_bitmask_patternChanMask_c; // 0x3f
			}

			if(ch >= chnMask.size())
			{
				chnMask.resize(ch + 1, 0);
				lastValue.resize(ch + 1, ModCommand::Empty());
				MPT_ASSERT(chnMask.size() <= GetNumChannels());
			}

			if(b & IT_bitmask_patternChanEnabled_c)  // 0x80
			{
				chnMask[ch] = patternData.ReadUint8();
			}

			// Now we grab the data for this particular row/channel.
			ModCommand dummy;
			ModCommand &m = ch < m_nChannels ? patData[ch] : dummy;

			if(chnMask[ch] & 0x10)
			{
				m.note = lastValue[ch].note;
			}
			if(chnMask[ch] & 0x20)
			{
				m.instr = lastValue[ch].instr;
			}
			if(chnMask[ch] & 0x40)
			{
				m.volcmd = lastValue[ch].volcmd;
				m.vol = lastValue[ch].vol;
			}
			if(chnMask[ch] & 0x80)
			{
				m.command = lastValue[ch].command;
				m.param = lastValue[ch].param;
			}
			if(chnMask[ch] & 1)	// Note
			{
				uint8 note = patternData.ReadUint8();
				if(note < 0x80) note += NOTE_MIN;
				if(!(GetType() & MOD_TYPE_MPT))
				{
					if(note > NOTE_MAX && note < 0xFD) note = NOTE_FADE;
					else if(note == 0xFD) note = NOTE_NONE;
				}
				m.note = note;
				lastValue[ch].note = note;
			}
			if(chnMask[ch] & 2)
			{
				uint8 instr = patternData.ReadUint8();
				m.instr = instr;
				lastValue[ch].instr = instr;
			}
			if(chnMask[ch] & 4)
			{
				uint8 vol = patternData.ReadUint8();
				// 0-64: Set Volume
				if(vol <= 64) { m.volcmd = VOLCMD_VOLUME; m.vol = vol; } else
				// 128-192: Set Panning
				if(vol >= 128 && vol <= 192) { m.volcmd = VOLCMD_PANNING; m.vol = vol - 128; } else
				// 65-74: Fine Volume Up
				if(vol < 75) { m.volcmd = VOLCMD_FINEVOLUP; m.vol = vol - 65; } else
				// 75-84: Fine Volume Down
				if(vol < 85) { m.volcmd = VOLCMD_FINEVOLDOWN; m.vol = vol - 75; } else
				// 85-94: Volume Slide Up
				if(vol < 95) { m.volcmd = VOLCMD_VOLSLIDEUP; m.vol = vol - 85; } else
				// 95-104: Volume Slide Down
				if(vol < 105) { m.volcmd = VOLCMD_VOLSLIDEDOWN; m.vol = vol - 95; } else
				// 105-114: Pitch Slide Up
				if(vol < 115) { m.volcmd = VOLCMD_PORTADOWN; m.vol = vol - 105; } else
				// 115-124: Pitch Slide Down
				if(vol < 125) { m.volcmd = VOLCMD_PORTAUP; m.vol = vol - 115; } else
				// 193-202: Portamento To
				if(vol >= 193 && vol <= 202) { m.volcmd = VOLCMD_TONEPORTAMENTO; m.vol = vol - 193; } else
				// 203-212: Vibrato depth
				if(vol >= 203 && vol <= 212)
				{
					m.volcmd = VOLCMD_VIBRATODEPTH;
					m.vol = vol - 203;
					// Old versions of ModPlug saved this as vibrato speed instead, so let's fix that.
					if(m.vol && m_dwLastSavedWithVersion && m_dwLastSavedWithVersion <= MAKE_VERSION_NUMERIC(1, 17, 02, 54))
						m.volcmd = VOLCMD_VIBRATOSPEED;
				} else
				// 213-222: Unused (was velocity)
				// 223-232: Offset
				if(vol >= 223 && vol <= 232) { m.volcmd = VOLCMD_OFFSET; m.vol = vol - 223; }
				lastValue[ch].volcmd = m.volcmd;
				lastValue[ch].vol = m.vol;
			}
			// Reading command/param
			if(chnMask[ch] & 8)
			{
				m.command = patternData.ReadUint8();
				m.param = patternData.ReadUint8();
				S3MConvert(m, true);
				// In some IT-compatible trackers, it is possible to input a parameter without a command.
				// In this case, we still need to update the last value memory. OpenMPT didn't do this until v1.25.01.07.
				// Example: ckbounce.it
				lastValue[ch].command = m.command;
				lastValue[ch].param = m.param;
			}
		}
	}

	if(!m_dwLastSavedWithVersion && fileHeader.cwtv == 0x0888)
	{
		// Up to OpenMPT 1.17.02.45 (r165), it was possible that the "last saved with" field was 0
		// when saving a file in OpenMPT for the first time.
		m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 17, 00, 00);
	}

	if(m_dwLastSavedWithVersion && m_madeWithTracker.empty())
	{
		m_madeWithTracker = MPT_USTRING("OpenMPT ") + MptVersion::ToUString(m_dwLastSavedWithVersion);
		if(memcmp(&fileHeader.reserved, "OMPT", 4) && (fileHeader.cwtv & 0xF000) == 0x5000)
		{
			m_madeWithTracker += MPT_USTRING(" (compatibility export)");
		} else if(MptVersion::IsTestBuild(m_dwLastSavedWithVersion))
		{
			m_madeWithTracker += MPT_USTRING(" (test build)");
		}
	} else
	{
		switch(fileHeader.cwtv >> 12)
		{
		case 0:
			if(!m_madeWithTracker.empty())
			{
				// BeRoTracker has been detected above.
			} else if(fileHeader.cwtv == 0x0214 && fileHeader.cmwt == 0x0200 && fileHeader.flags == 9 && fileHeader.special == 0
				&& fileHeader.highlight_major == 0 && fileHeader.highlight_minor == 0
				&& fileHeader.insnum == 0 && fileHeader.patnum + 1 == fileHeader.ordnum
				&& fileHeader.globalvol == 128 && fileHeader.mv == 100 && fileHeader.speed == 1 && fileHeader.sep == 128 && fileHeader.pwd == 0
				&& fileHeader.msglength == 0 && fileHeader.msgoffset == 0 && fileHeader.reserved == 0)
			{
				m_madeWithTracker = MPT_USTRING("OpenSPC conversion");
			} else if(fileHeader.cwtv == 0x0214 && fileHeader.cmwt == 0x0200 && fileHeader.reserved == 0)
			{
				// ModPlug Tracker 1.00a5, instruments 560 bytes apart
				m_dwLastSavedWithVersion = MAKE_VERSION_NUMERIC(1, 00, 00, A5);
				m_madeWithTracker = MPT_USTRING("ModPlug Tracker 1.00a5");
				interpretModPlugMade = true;
			} else if(fileHeader.cwtv == 0x0214 && fileHeader.cmwt == 0x0214 && !memcmp(&fileHeader.reserved, "CHBI", 4))
			{
				m_madeWithTracker = MPT_USTRING("ChibiTracker");
			} else if(fileHeader.cwtv == 0x0214 && fileHeader.cmwt == 0x0214 && fileHeader.special <= 1 && fileHeader.pwd == 0 && fileHeader.reserved == 0
				&& (fileHeader.flags & (ITFileHeader::vol0Optimisations | ITFileHeader::instrumentMode | ITFileHeader::useMIDIPitchController | ITFileHeader::reqEmbeddedMIDIConfig | ITFileHeader::extendedFilterRange)) == ITFileHeader::instrumentMode
				&& m_nSamples > 0 && !strcmp(Samples[1].filename, "XXXXXXXX.YYY"))
			{
				m_madeWithTracker = MPT_USTRING("CheeseTracker");
			} else if(fileHeader.cmwt < 0x0300)
			{
				if(fileHeader.cmwt > 0x0214)
				{
					m_madeWithTracker = MPT_USTRING("Impulse Tracker 2.15");
				} else if(fileHeader.cwtv > 0x0214)
				{
					// Patched update of IT 2.14 (0x0215 - 0x0217 == p1 - p3)
					// p4 (as found on modland) adds the ITVSOUND driver, but doesn't seem to change
					// anything as far as file saving is concerned.
					m_madeWithTracker = mpt::format(MPT_USTRING("Impulse Tracker 2.14p%1"))(fileHeader.cwtv - 0x0214);
				} else
				{
					m_madeWithTracker = mpt::format(MPT_USTRING("Impulse Tracker %1.%2"))((fileHeader.cwtv & 0x0F00) >> 8, mpt::ufmt::hex0<2>((fileHeader.cwtv & 0xFF)));
				}
				if(m_FileHistory.empty() && fileHeader.reserved != 0)
				{
					// Starting from  version 2.07, IT encrypts the total edit time of a module in the "reserved" field
					uint32 editTime = fileHeader.reserved;
					if(fileHeader.cwtv >= 0x0208)
					{
						editTime ^= 0x4954524B;	// 'ITRK'
						editTime = (editTime >> 7) | (editTime << (32 - 7));
						editTime = -(int32)editTime;
						editTime = (editTime << 4) | (editTime >> (32 - 4));
						editTime ^= 0x4A54484C;	// 'JTHL'
					}

					FileHistory hist;
					MemsetZero(hist);
					hist.openTime = static_cast<uint32>(editTime * (HISTORY_TIMER_PRECISION / 18.2f));
					m_FileHistory.push_back(hist);
				}
			}
			break;
		case 1:
			m_madeWithTracker = GetSchismTrackerVersion(fileHeader.cwtv);
			// Hertz in linear mode: Added 2015-01-29, https://github.com/schismtracker/schismtracker/commit/671b30311082a0e7df041fca25f989b5d2478f69
			if(fileHeader.cwtv < SchismVersionFromDate<2015, 01, 29>::Version())
				m_playBehaviour.reset(kHertzInLinearMode);
			// Qxx with short samples: Added 2016-05-13, https://github.com/schismtracker/schismtracker/commit/e7b1461fe751554309fd403713c2a1ef322105ca
			if(fileHeader.cwtv < SchismVersionFromDate<2016, 05, 13>::Version())
				m_playBehaviour.reset(kITShortSampleRetrig);
			break;
		case 4:
			m_madeWithTracker = mpt::format(MPT_USTRING("pyIT %1.%2"))((fileHeader.cwtv & 0x0F00) >> 8, mpt::ufmt::hex0<2>((fileHeader.cwtv & 0xFF)));
			break;
		case 6:
			m_madeWithTracker = MPT_USTRING("BeRoTracker");
			break;
		case 7:
			m_madeWithTracker = mpt::format(MPT_USTRING("ITMCK %1.%2.%3"))((fileHeader.cwtv >> 8) & 0x0F, (fileHeader.cwtv >> 4) & 0x0F, fileHeader.cwtv & 0x0F);
			break;
		case 0xD:
			m_madeWithTracker = MPT_USTRING("spc2it");
			break;
		}
	}

	if(GetType() == MOD_TYPE_MPT)
	{
		// START - mpt specific:
		if(fileHeader.cwtv > 0x0889 && file.Seek(mptStartPos))
		{
			LoadMPTMProperties(file, fileHeader.cwtv);
		}
	}

	return true;
}


void CSoundFile::LoadMPTMProperties(FileReader &file, uint16 cwtv)
{
	mpt::istringstream iStrm(file.GetRawDataAsString());

	if(cwtv >= 0x88D)
	{
		srlztn::SsbRead ssb(iStrm);
		ssb.BeginRead("mptm", MptVersion::num);
		ssb.ReadItem(GetTuneSpecificTunings(), "0", &ReadTuningCollection);
		ssb.ReadItem(*this, "1", &ReadTuningMap);
		ssb.ReadItem(Order, "2", &ReadModSequenceOld);
		ssb.ReadItem(Patterns, FileIdPatterns, &ReadModPatterns);
		ssb.ReadItem(Order, FileIdSequences, &ReadModSequences);

		if(ssb.GetStatus() & srlztn::SNT_FAILURE)
		{
			AddToLog(LogError, MPT_USTRING("Unknown error occurred while deserializing file."));
		}
	} else
	{
		// Loading for older files.
		std::string name;
		if(GetTuneSpecificTunings().Deserialize(iStrm, name) != Tuning::SerializationResult::Success)
		{
			AddToLog(LogError, MPT_USTRING("Loading tune specific tunings failed."));
		} else
		{
			ReadTuningMapImpl(iStrm, *this, 0, cwtv < 0x88C);
		}
	}
}


#ifndef MODPLUG_NO_FILESAVE

// Save edit history. Pass a null pointer for *f to retrieve the number of bytes that would be written.
static uint32 SaveITEditHistory(const CSoundFile &sndFile, FILE *f)
{
	size_t num = sndFile.GetFileHistory().size();
#ifdef MODPLUG_TRACKER
	const CModDoc *pModDoc = sndFile.GetpModDoc();
	num += (pModDoc != nullptr) ? 1 : 0;	// + 1 for this session
#endif // MODPLUG_TRACKER

	uint16 fnum = mpt::saturate_cast<uint16>(num);	// Number of entries that are actually going to be written
	const uint32 bytesWritten = 2 + fnum * 8;		// Number of bytes that are actually going to be written

	if(f == nullptr)
		return bytesWritten;

	// Write number of history entries
	mpt::IO::WriteIntLE(f, fnum);

	// Write history data
	const size_t start = (num > uint16_max) ? num - uint16_max : 0;
	for(size_t n = start; n < num; n++)
	{
		FileHistory mptHistory;

#ifdef MODPLUG_TRACKER
		if(n < sndFile.GetFileHistory().size())
#endif // MODPLUG_TRACKER
		{
			// Previous timestamps
			mptHistory = sndFile.GetFileHistory().at(n);
#ifdef MODPLUG_TRACKER
		} else
		{
			// Current ("new") timestamp
			const time_t creationTime = pModDoc->GetCreationTime();

			MemsetZero(mptHistory.loadDate);
			//localtime_s(&loadDate, &creationTime);
			const tm* const p = localtime(&creationTime);
			if (p != nullptr)
				mptHistory.loadDate = *p;
			else
				sndFile.AddToLog("Unable to retrieve current time.");

			mptHistory.openTime = (uint32)(difftime(time(nullptr), creationTime) * (double)HISTORY_TIMER_PRECISION);
#endif // MODPLUG_TRACKER
		}

		ITHistoryStruct itHistory;
		itHistory.ConvertToIT(mptHistory);
		mpt::IO::Write(f, itHistory);
	}

	return bytesWritten;
}


bool CSoundFile::SaveIT(const mpt::PathString &filename, bool compatibilityExport)
{
	const CModSpecifications &specs = (GetType() == MOD_TYPE_MPT ? ModSpecs::mptm : (compatibilityExport ? ModSpecs::it : ModSpecs::itEx));

	uint32 dwChnNamLen;
	ITFileHeader itHeader;
	uint64 dwPos = 0;
	uint32 dwHdrPos = 0, dwExtra = 0;
	FILE *f;

	if(filename.empty() || ((f = mpt_fopen(filename, "wb")) == NULL)) return false;

	// Writing Header
	MemsetZero(itHeader);
	dwChnNamLen = 0;
	memcpy(itHeader.id, "IMPM", 4);
	mpt::String::Write<mpt::String::nullTerminated>(itHeader.songname, m_songName);

	itHeader.highlight_minor = (uint8)std::min(m_nDefaultRowsPerBeat, ROWINDEX(uint8_max));
	itHeader.highlight_major = (uint8)std::min(m_nDefaultRowsPerMeasure, ROWINDEX(uint8_max));

	if(GetType() == MOD_TYPE_MPT)
	{
		itHeader.ordnum = Order().GetLengthTailTrimmed();
		if(Order().NeedsExtraDatafield() && itHeader.ordnum > 256)
		{
			// If there are more order items, write them elsewhere.
			itHeader.ordnum = 256;
		}
	} else
	{
		// An additional "---" pattern is appended so Impulse Tracker won't ignore the last order item.
		// Interestingly, this can exceed IT's 256 order limit. Also, IT will always save at least two orders.
		itHeader.ordnum = std::min(Order().GetLengthTailTrimmed(), specs.ordersMax) + 1;
		if(itHeader.ordnum < 2) itHeader.ordnum = 2;
	}

	itHeader.insnum = std::min(m_nInstruments, specs.instrumentsMax);
	itHeader.smpnum = std::min(m_nSamples, specs.samplesMax);
	itHeader.patnum = std::min(Patterns.GetNumPatterns(), specs.patternsMax);

	// Parapointers
	std::vector<uint32> patpos(itHeader.patnum, 0);
	std::vector<uint32> smppos(itHeader.smpnum, 0);
	std::vector<uint32> inspos(itHeader.insnum, 0);

	//VERSION
	if(GetType() == MOD_TYPE_MPT)
	{
		// MPTM
		itHeader.cwtv = verMptFileVer;	// Used in OMPT-hack versioning.
		itHeader.cmwt = 0x888;
	} else
	{
		// IT
		MptVersion::VersionNum vVersion = MptVersion::num;
		itHeader.cwtv = 0x5000 | (uint16)((vVersion >> 16) & 0x0FFF); // format: txyy (t = tracker ID, x = version major, yy = version minor), e.g. 0x5117 (OpenMPT = 5, 117 = v1.17)
		itHeader.cmwt = 0x0214;	// Common compatible tracker :)
		// Hack from schism tracker:
		for(INSTRUMENTINDEX nIns = 1; nIns <= GetNumInstruments(); nIns++)
		{
			if(Instruments[nIns] && Instruments[nIns]->PitchEnv.dwFlags[ENV_FILTER])
			{
				itHeader.cmwt = 0x0216;
				break;
			}
		}

		if(!compatibilityExport)
		{
			// This way, we indicate that the file will most likely contain OpenMPT hacks. Compatibility export puts 0 here.
			memcpy(&itHeader.reserved, "OMPT", 4);
		}
	}

	itHeader.flags = ITFileHeader::useStereoPlayback | ITFileHeader::useMIDIPitchController;
	itHeader.special = ITFileHeader::embedEditHistory | ITFileHeader::embedPatternHighlights;
	if(m_nInstruments) itHeader.flags |= ITFileHeader::instrumentMode;
	if(m_SongFlags[SONG_LINEARSLIDES]) itHeader.flags |= ITFileHeader::linearSlides;
	if(m_SongFlags[SONG_ITOLDEFFECTS]) itHeader.flags |= ITFileHeader::itOldEffects;
	if(m_SongFlags[SONG_ITCOMPATGXX]) itHeader.flags |= ITFileHeader::itCompatGxx;
	if(m_SongFlags[SONG_EXFILTERRANGE] && !compatibilityExport) itHeader.flags |= ITFileHeader::extendedFilterRange;

	itHeader.globalvol = (uint8)(m_nDefaultGlobalVolume >> 1);
	itHeader.mv = (uint8)MIN(m_nSamplePreAmp, 128u);
	itHeader.speed = (uint8)MIN(m_nDefaultSpeed, 255u);
 	itHeader.tempo = (uint8)MIN(m_nDefaultTempo.GetInt(), 255u);  //Limit this one to 255, we save the real one as an extension below.
	itHeader.sep = 128; // pan separation
	// IT doesn't have a per-instrument Pitch Wheel Depth setting, so we just store the first non-zero PWD setting in the header.
	for(INSTRUMENTINDEX ins = 1; ins < GetNumInstruments(); ins++)
	{
		if(Instruments[ins] != nullptr && Instruments[ins]->midiPWD != 0)
		{
			itHeader.pwd = (uint8)mpt::abs(Instruments[ins]->midiPWD);
			break;
		}
	}

	dwHdrPos = sizeof(itHeader) + itHeader.ordnum;
	// Channel Pan and Volume
	memset(itHeader.chnpan, 0xA0, 64);
	memset(itHeader.chnvol, 64, 64);

	for(CHANNELINDEX ich = 0; ich < std::min(m_nChannels, CHANNELINDEX(64)); ich++) // Header only has room for settings for 64 chans...
	{
		itHeader.chnpan[ich] = (uint8)(ChnSettings[ich].nPan >> 2);
		if (ChnSettings[ich].dwFlags[CHN_SURROUND]) itHeader.chnpan[ich] = 100;
		itHeader.chnvol[ich] = (uint8)(ChnSettings[ich].nVolume);
#ifdef MODPLUG_TRACKER
		if(TrackerSettings::Instance().MiscSaveChannelMuteStatus)
#endif
		if (ChnSettings[ich].dwFlags[CHN_MUTE]) itHeader.chnpan[ich] |= 0x80;
	}

	// Channel names
	if(!compatibilityExport)
	{
		for(CHANNELINDEX i = 0; i < m_nChannels; i++)
		{
			if(ChnSettings[i].szName[0])
			{
				dwChnNamLen = (i + 1) * MAX_CHANNELNAME;
			}
		}
		if(dwChnNamLen) dwExtra += dwChnNamLen + 8;
	}

	if(!m_MidiCfg.IsMacroDefaultSetupUsed())
	{
		itHeader.flags |= ITFileHeader::reqEmbeddedMIDIConfig;
		itHeader.special |= ITFileHeader::embedMIDIConfiguration;
		dwExtra += sizeof(MIDIMacroConfigData);
	}

	// Pattern Names
	const PATTERNINDEX numNamedPats = compatibilityExport ? 0 : Patterns.GetNumNamedPatterns();
	if(numNamedPats > 0)
	{
		dwExtra += (numNamedPats * MAX_PATTERNNAME) + 8;
	}

	// Mix Plugins. Just calculate the size of this extra block for now.
	if(!compatibilityExport)
	{
		dwExtra += SaveMixPlugins(nullptr, true);
	}

	// Edit History. Just calculate the size of this extra block for now.
	dwExtra += SaveITEditHistory(*this, nullptr);

	// Comments
	uint16 msglength = 0;
	if(!m_songMessage.empty())
	{
		itHeader.special |= ITFileHeader::embedSongMessage;
		itHeader.msglength = msglength = mpt::saturate_cast<uint16>(m_songMessage.length() + 1u);
		itHeader.msgoffset = dwHdrPos + dwExtra + (itHeader.insnum + itHeader.smpnum + itHeader.patnum) * 4;
	}

	// Write file header
	mpt::IO::Write(f, itHeader);

	Order().WriteAsByte(f, itHeader.ordnum);
	for(uint16 i = 0; i < itHeader.insnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, inspos[i]);
	}
	for(uint16 i = 0; i < itHeader.smpnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, smppos[i]);
	}
	for(uint16 i = 0; i < itHeader.patnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, patpos[i]);
	}

	// Writing edit history information
	SaveITEditHistory(*this, f);

	// Writing midi cfg
	if(itHeader.flags & ITFileHeader::reqEmbeddedMIDIConfig)
	{
		mpt::IO::Write(f, static_cast<MIDIMacroConfigData &>(m_MidiCfg));
	}

	// Writing pattern names
	if(numNamedPats)
	{
		mpt::IO::WriteRaw(f, "PNAM", 4);
		mpt::IO::WriteIntLE<uint32>(f, numNamedPats * MAX_PATTERNNAME);

		for(PATTERNINDEX pat = 0; pat < numNamedPats; pat++)
		{
			char name[MAX_PATTERNNAME];
			mpt::String::Write<mpt::String::maybeNullTerminated>(name, Patterns[pat].GetName());
			mpt::IO::Write(f, name);
		}
	}

	// Writing channel names
	if(dwChnNamLen && !compatibilityExport)
	{
		mpt::IO::WriteRaw(f, "CNAM", 4);
		mpt::IO::WriteIntLE<uint32>(f, dwChnNamLen);
		uint32 nChnNames = dwChnNamLen / MAX_CHANNELNAME;
		for(uint32 inam = 0; inam < nChnNames; inam++)
		{
			char name[MAX_CHANNELNAME];
			mpt::String::Write<mpt::String::maybeNullTerminated>(name, ChnSettings[inam].szName);
			mpt::IO::Write(f, name);
		}
	}

	// Writing mix plugins info
	if(!compatibilityExport)
	{
		SaveMixPlugins(f, false);
	}

	// Writing song message
	dwPos = dwHdrPos + dwExtra + (itHeader.insnum + itHeader.smpnum + itHeader.patnum) * 4;
	if(itHeader.special & ITFileHeader::embedSongMessage)
	{
		dwPos += msglength;
		mpt::IO::WriteRaw(f, m_songMessage.c_str(), msglength);
	}

	// Writing instruments
	const ModInstrument dummyInstr;
	for(INSTRUMENTINDEX nins = 1; nins <= itHeader.insnum; nins++)
	{
		ITInstrumentEx iti;
		uint32 instSize;

		const ModInstrument &instr = (Instruments[nins] != nullptr) ? *Instruments[nins] : dummyInstr;
		instSize = iti.ConvertToIT(instr, compatibilityExport, *this);

		// Writing instrument
		inspos[nins - 1] = static_cast<uint32>(dwPos);
		dwPos += instSize;
		mpt::IO::WritePartial(f, iti, instSize);
	}

	// Writing dummy sample headers (until we know the correct sample data offset)
	ITSample itss;
	MemsetZero(itss);
	for(SAMPLEINDEX smp = 0; smp < itHeader.smpnum; smp++)
	{
		smppos[smp] = static_cast<uint32>(dwPos);
		dwPos += sizeof(ITSample);
		mpt::IO::Write(f, itss);
	}

	// Writing Patterns
	bool bNeedsMptPatSave = false;
	for(PATTERNINDEX pat = 0; pat < itHeader.patnum; pat++)
	{
		uint32 dwPatPos = static_cast<uint32>(dwPos);
		if (!Patterns.IsValidPat(pat)) continue;

		if(Patterns[pat].GetOverrideSignature())
			bNeedsMptPatSave = true;

		// Check for empty pattern
		if(Patterns[pat].GetNumRows() == 64 && Patterns.IsPatternEmpty(pat))
		{
			patpos[pat] = 0;
			continue;
		}

		patpos[pat] = static_cast<uint32>(dwPos);

		// Write pattern header
		ROWINDEX writeRows = mpt::saturate_cast<uint16>(Patterns[pat].GetNumRows());
		uint16 writeSize = 0;
		uint16le patinfo[4];
		patinfo[0] = 0;
		patinfo[1] = (uint16)writeRows;
		patinfo[2] = 0;
		patinfo[3] = 0;

		mpt::IO::Write(f, patinfo);
		dwPos += 8;

		const CHANNELINDEX maxChannels = std::min(specs.channelsMax, GetNumChannels());
		std::vector<uint8> chnmask(maxChannels, 0xFF);
		std::vector<ModCommand> lastvalue(maxChannels, ModCommand::Empty());

		for(ROWINDEX row = 0; row < writeRows; row++)
		{
			uint32 len = 0;
			// Maximum 7 bytes per cell, plus end of row marker, so this buffer is always large enough to cover one row.
			uint8 buf[7 * MAX_BASECHANNELS + 1];
			const ModCommand *m = Patterns[pat].GetRow(row);

			for(CHANNELINDEX ch = 0; ch < maxChannels; ch++, m++)
			{
				// Skip mptm-specific notes.
				if(m->IsPcNote())
				{
					bNeedsMptPatSave = true;
					continue;
				}

				uint8 b = 0;
				uint8 command = m->command;
				uint8 param = m->param;
				uint8 vol = 0xFF;
				uint8 note = m->note;
				if (note != NOTE_NONE) b |= 1;
				if (m->IsNote()) note -= NOTE_MIN;
				if (note == NOTE_FADE && GetType() != MOD_TYPE_MPT) note = 0xF6;
				if (m->instr) b |= 2;
				if (m->volcmd != VOLCMD_NONE)
				{
					vol = std::min(m->vol, uint8(9));
					switch(m->volcmd)
					{
					case VOLCMD_VOLUME:			vol = std::min(m->vol, uint8(64)); break;
					case VOLCMD_PANNING:		vol = std::min(m->vol, uint8(64)) + 128; break;
					case VOLCMD_VOLSLIDEUP:		vol += 85; break;
					case VOLCMD_VOLSLIDEDOWN:	vol += 95; break;
					case VOLCMD_FINEVOLUP:		vol += 65; break;
					case VOLCMD_FINEVOLDOWN:	vol += 75; break;
					case VOLCMD_VIBRATODEPTH:	vol += 203; break;
					case VOLCMD_VIBRATOSPEED:	if(command == CMD_NONE)
												{
													// illegal command -> move if possible
													command = CMD_VIBRATO;
													param = std::min(m->vol, uint8(15)) << 4;
												} else
												{
													vol = 203;
												}
												break;
					case VOLCMD_TONEPORTAMENTO:	vol += 193; break;
					case VOLCMD_PORTADOWN:		vol += 105; break;
					case VOLCMD_PORTAUP:		vol += 115; break;
					case VOLCMD_OFFSET:			if(!compatibilityExport) vol += 223;
												break;
					default:					vol = 0xFF;
					}
				}
				if (vol != 0xFF) b |= 4;
				if (command != CMD_NONE)
				{
					S3MSaveConvert(command, param, true, compatibilityExport);
					if (command) b |= 8;
				}
				// Packing information
				if (b)
				{
					// Same note ?
					if (b & 1)
					{
						if ((note == lastvalue[ch].note) && (lastvalue[ch].volcmd & 1))
						{
							b &= ~1;
							b |= 0x10;
						} else
						{
							lastvalue[ch].note = note;
							lastvalue[ch].volcmd |= 1;
						}
					}
					// Same instrument ?
					if (b & 2)
					{
						if ((m->instr == lastvalue[ch].instr) && (lastvalue[ch].volcmd & 2))
						{
							b &= ~2;
							b |= 0x20;
						} else
						{
							lastvalue[ch].instr = m->instr;
							lastvalue[ch].volcmd |= 2;
						}
					}
					// Same volume column byte ?
					if (b & 4)
					{
						if ((vol == lastvalue[ch].vol) && (lastvalue[ch].volcmd & 4))
						{
							b &= ~4;
							b |= 0x40;
						} else
						{
							lastvalue[ch].vol = vol;
							lastvalue[ch].volcmd |= 4;
						}
					}
					// Same command / param ?
					if (b & 8)
					{
						if ((command == lastvalue[ch].command) && (param == lastvalue[ch].param) && (lastvalue[ch].volcmd & 8))
						{
							b &= ~8;
							b |= 0x80;
						} else
						{
							lastvalue[ch].command = command;
							lastvalue[ch].param = param;
							lastvalue[ch].volcmd |= 8;
						}
					}
					if (b != chnmask[ch])
					{
						chnmask[ch] = b;
						buf[len++] = uint8((ch + 1) | IT_bitmask_patternChanEnabled_c);
						buf[len++] = b;
					} else
					{
						buf[len++] = uint8(ch + 1);
					}
					if (b & 1) buf[len++] = note;
					if (b & 2) buf[len++] = m->instr;
					if (b & 4) buf[len++] = vol;
					if (b & 8)
					{
						buf[len++] = command;
						buf[len++] = param;
					}
				}
			}
			buf[len++] = 0;
			if(writeSize > uint16_max - len)
			{
				AddToLog(mpt::format("%1 (%2 %3)")(str_tooMuchPatternData, str_pattern, pat));
				break;
			} else
			{
				dwPos += len;
				writeSize += (uint16)len;
				mpt::IO::WriteRaw(f, buf, len);
			}
		}

		fseek(f, dwPatPos, SEEK_SET);
		patinfo[0] = writeSize;
		mpt::IO::Write(f, patinfo);
		fseek(f, static_cast<long>(dwPos), SEEK_SET);
	}
	// Writing Sample Data
	for(SAMPLEINDEX smp = 1; smp <= itHeader.smpnum; smp++)
	{
#ifdef MODPLUG_TRACKER
		uint32 type = GetType() == MOD_TYPE_IT ? 1 : 4;
		if(compatibilityExport) type = 2;
		bool compress = ((((Samples[smp].GetNumChannels() > 1) ? TrackerSettings::Instance().MiscITCompressionStereo : TrackerSettings::Instance().MiscITCompressionMono) & type) != 0);
#else
		bool compress = false;
#endif // MODPLUG_TRACKER
		// Old MPT, DUMB and probably other libraries will only consider the IT2.15 compression flag if the header version also indicates IT2.15.
		// MilkyTracker <= 0.90.85 will only assume IT2.15 compression with cmwt == 0x215, ignoring the delta flag completely.
		itss.ConvertToIT(Samples[smp], GetType(), compress, itHeader.cmwt >= 0x215, GetType() == MOD_TYPE_MPT);
		const bool isExternal = itss.cvt == ITSample::cvtExternalSample;

		mpt::String::Write<mpt::String::nullTerminated>(itss.name, m_szNames[smp]);

		itss.samplepointer = static_cast<uint32>(dwPos);
		if(dwPos > uint32_max)
		{
			// Sample position does not fit into sample pointer!
			AddToLog(mpt::format("Cannot save sample %1: File size exceeds 4 GB.")(smp));
			itss.samplepointer = 0;
			itss.length = 0;
		}
		SmpLength smpLength = itss.length;	// Possibly truncated to 2^32 samples
		fseek(f, smppos[smp - 1], SEEK_SET);
		mpt::IO::Write(f, itss);
		if(dwPos > uint32_max)
		{
			continue;
		}
		// TODO this actually wraps around at 2 GB, so we either need to use the 64-bit seek API or warn earlier!
		fseek(f, static_cast<long>(dwPos), SEEK_SET);
		if(!isExternal)
		{
			if(Samples[smp].nLength != smpLength)
			{
				// Sample length does not fit into IT header!
				AddToLog(mpt::format("Truncating sample %1: Length exceeds exceeds 4 gigasamples.")(smp));
			}
			dwPos += itss.GetSampleFormat().WriteSample(f, Samples[smp], smpLength);
		} else
		{
#ifdef MPT_EXTERNAL_SAMPLES
			const std::string filenameU8 = GetSamplePath(smp).AbsolutePathToRelative(filename.GetPath()).ToUTF8();
			const size_t strSize = filenameU8.size();
			size_t intBytes = 0;
			if(mpt::IO::WriteVarInt(f, strSize, &intBytes))
			{
				dwPos += intBytes + strSize;
				mpt::IO::WriteRaw(f, filenameU8.data(), strSize);
			}
#endif // MPT_EXTERNAL_SAMPLES
		}
	}

	//Save hacked-on extra info
	if(!compatibilityExport)
	{
		if(GetNumInstruments())
		{
			SaveExtendedInstrumentProperties(itHeader.insnum, f);
		}
		SaveExtendedSongProperties(f);
	}

	// Updating offsets
	fseek(f, dwHdrPos, SEEK_SET);
	for(uint16 i = 0; i < itHeader.insnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, inspos[i]);
	}
	for(uint16 i = 0; i < itHeader.smpnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, smppos[i]);
	}
	for(uint16 i = 0; i < itHeader.patnum; ++i)
	{
		mpt::IO::WriteIntLE<uint32>(f, patpos[i]);
	}

	if(GetType() == MOD_TYPE_IT)
	{
		fclose(f);
		return true;
	}

	//hack
	//BEGIN: MPT SPECIFIC:

	bool success = true;

	fseek(f, 0, SEEK_END);
	{
	mpt::FILE_ostream fout(f);

	const uint32 MPTStartPos = (uint32)fout.tellp();
	
	// catch standard library truncating files
	MPT_ASSERT_ALWAYS(MPTStartPos > 0);

	srlztn::SsbWrite ssb(fout);
	ssb.BeginWrite("mptm", MptVersion::num);

	if(GetTuneSpecificTunings().GetNumTunings() > 0)
		ssb.WriteItem(GetTuneSpecificTunings(), "0", &WriteTuningCollection);
	if(AreNonDefaultTuningsUsed(*this))
		ssb.WriteItem(*this, "1", &WriteTuningMap);
	if(Order().NeedsExtraDatafield())
		ssb.WriteItem(Order, "2", &WriteModSequenceOld);
	if(bNeedsMptPatSave)
		ssb.WriteItem(Patterns, FileIdPatterns, &WriteModPatterns);
	ssb.WriteItem(Order, FileIdSequences, &WriteModSequences);

	ssb.FinishWrite();

	if(ssb.GetStatus() & srlztn::SNT_FAILURE)
	{
		AddToLog(LogError, MPT_USTRING("Error occurred in writing MPTM extensions."));
	}

	//Last 4 bytes should tell where the hack mpt things begin.
	if(!fout.good())
	{
		fout.clear();
		success = false;
	}
	mpt::IO::WriteIntLE<uint32>(fout, MPTStartPos);

	fout.seekp(0, std::ios_base::end);
	}
	fclose(f);
	f = nullptr;

	//END  : MPT SPECIFIC

	//NO WRITING HERE ANYMORE.

	return success;
}


#endif // MODPLUG_NO_FILESAVE


#ifndef MODPLUG_NO_FILESAVE

uint32 CSoundFile::SaveMixPlugins(FILE *f, bool bUpdate)
{
#ifndef NO_PLUGINS
	uint32 chinfo[MAX_BASECHANNELS];
	char id[4];
	uint32 nPluginSize;
	uint32 nTotalSize = 0;
	uint32 nChInfo = 0;

	for(PLUGINDEX i = 0; i < MAX_MIXPLUGINS; i++)
	{
		const SNDMIXPLUGIN &plugin = m_MixPlugins[i];
		if(plugin.IsValidPlugin())
		{
			nPluginSize = sizeof(SNDMIXPLUGININFO) + 4; // plugininfo+4 (datalen)
			if((plugin.pMixPlugin) && (bUpdate))
			{
				plugin.pMixPlugin->SaveAllParameters();
			}
			nPluginSize += plugin.pluginData.size();

			uint32 MPTxPlugDataSize = 4 + sizeof(float32) +		// 4 for ID and size of dryRatio
									 4 + sizeof(int32);			// Default Program
								// for each extra entity, add 4 for ID, plus 4 for size of entity, plus size of entity

			nPluginSize += MPTxPlugDataSize + 4; //+4 is for size itself: sizeof(uint32) is 4
			if(f)
			{
				// write plugin ID
				id[0] = 'F';
				id[1] = i < 100 ? 'X' : '0' + i / 100;
				id[2] = '0' + (i / 10) % 10u;
				id[3] = '0' + (i % 10u);
				mpt::IO::WriteRaw(f, id, 4);

				// write plugin size:
				mpt::IO::WriteIntLE<uint32>(f, nPluginSize);
				mpt::IO::Write(f, m_MixPlugins[i].Info);
				uint32 dataSize = mpt::saturate_cast<uint32>(m_MixPlugins[i].pluginData.size());
				mpt::IO::WriteIntLE<uint32>(f, dataSize);
				if(dataSize)
				{
					mpt::IO::WriteRaw(f, m_MixPlugins[i].pluginData.data(), dataSize);
				}

				mpt::IO::WriteIntLE<uint32>(f, MPTxPlugDataSize);

				// Dry/Wet ratio
				memcpy(id, "DWRT", 4);
				mpt::IO::WriteRaw(f, id, 4);
				// DWRT chunk does not include a size, so better make sure we always write 4 bytes here.
				STATIC_ASSERT(sizeof(IEEE754binary32LE) == 4);
				mpt::IO::Write(f, IEEE754binary32LE(m_MixPlugins[i].fDryRatio));

				// Default program
				memcpy(id, "PROG", 4);
				mpt::IO::WriteRaw(f, id, 4);
				// PROG chunk does not include a size, so better make sure we always write 4 bytes here.
				STATIC_ASSERT(sizeof(m_MixPlugins[i].defaultProgram) == sizeof(int32));
				mpt::IO::WriteIntLE<int32>(f, m_MixPlugins[i].defaultProgram);

				// Please, if you add any more chunks here, don't repeat history (see above) and *do* add a size field for your chunk, mmmkay?
			}
			nTotalSize += nPluginSize + 8;
		}
	}
	for(CHANNELINDEX j = 0; j < GetNumChannels(); j++)
	{
		if(j < MAX_BASECHANNELS)
		{
			if((chinfo[j] = ChnSettings[j].nMixPlugin) != 0)
			{
				nChInfo = j + 1;
			}
		}
	}
	if(nChInfo)
	{
		if(f)
		{
			memcpy(id, "CHFX", 4);
			mpt::IO::WriteRaw(f, id, 4);
			mpt::IO::WriteIntLE<uint32>(f, nChInfo * 4);
			for(uint32 i = 0; i < nChInfo; ++i)
			{
				mpt::IO::WriteIntLE<uint32>(f, chinfo[i]);
			}
		}
		nTotalSize += nChInfo * 4 + 8;
	}
	return nTotalSize;
#else
	MPT_UNREFERENCED_PARAMETER(f);
	MPT_UNREFERENCED_PARAMETER(bUpdate);
	return 0;
#endif // NO_PLUGINS
}

#endif // MODPLUG_NO_FILESAVE


void CSoundFile::LoadMixPlugins(FileReader &file)
{
	while(file.CanRead(9))
	{
		char code[4];
		file.ReadArray(code);
		const uint32 chunkSize = file.ReadUint32LE();
		if(!memcmp(code, "IMPI", 4)	// IT instrument, we definitely read too far
			|| !memcmp(code, "IMPS", 4)	// IT sample, ditto
			|| !memcmp(code, "XTPM", 4)	// Instrument extensions, ditto
			|| !memcmp(code, "STPM", 4)	// Song extensions, ditto
			|| !file.CanRead(chunkSize))
		{
			file.SkipBack(8);
			return;
		}
		FileReader chunk = file.ReadChunk(chunkSize);

		// Channel FX
		if(!memcmp(code, "CHFX", 4))
		{
			for (size_t ch = 0; ch < MAX_BASECHANNELS; ch++)
			{
				ChnSettings[ch].nMixPlugin = static_cast<PLUGINDEX>(chunk.ReadUint32LE());
			}
#ifndef NO_PLUGINS
		}
		// Plugin Data FX00, ... FX99, F100, ... F255
#define ISNUMERIC(x) (code[(x)] >= '0' && code[(x)] <= '9')
		else if(code[0] == 'F' && (code[1] == 'X' || ISNUMERIC(1)) && ISNUMERIC(2) && ISNUMERIC(3))
#undef ISNUMERIC
		{
			PLUGINDEX plug = (code[2] - '0') * 10 + (code[3] - '0');	//calculate plug-in number.
			if(code[1] != 'X') plug += (code[1] - '0') * 100;

			if(plug < MAX_MIXPLUGINS)
			{
				ReadMixPluginChunk(chunk, m_MixPlugins[plug]);
			}
#endif // NO_PLUGINS
		} else if(!memcmp(code, "MODU", 4))
		{
			m_madeWithTracker = MPT_USTRING("BeRoTracker");
			m_dwLastSavedWithVersion = 0;	// Reset MPT detection for old files that have a similar fingerprint
		}
	}
}


#ifndef NO_PLUGINS
void CSoundFile::ReadMixPluginChunk(FileReader &file, SNDMIXPLUGIN &plugin)
{
	// MPT's standard plugin data. Size not specified in file.. grrr..
	file.ReadStruct(plugin.Info);
	mpt::String::SetNullTerminator(plugin.Info.szName);
	mpt::String::SetNullTerminator(plugin.Info.szLibraryName);
	plugin.editorX = plugin.editorY = int32_min;

	// Plugin user data
	const uint32 pluginDataChunkSize = file.ReadUint32LE();
	FileReader pluginDataChunk = file.ReadChunk(pluginDataChunkSize);

	if(pluginDataChunk.IsValid())
	{
		plugin.pluginData.resize(pluginDataChunkSize);
		pluginDataChunk.ReadRaw(plugin.pluginData.data(), pluginDataChunkSize);
	}

	FileReader modularData = file.ReadChunk(file.ReadUint32LE());

	//if dwMPTExtra is positive and there are dwMPTExtra bytes left in nPluginSize, we have some more data!
	if(modularData.IsValid())
	{
		while(modularData.CanRead(5))
		{
			// do we recognize this chunk?
			char code[4];
			modularData.ReadArray(code);
			uint32 dataSize = 0;
			if(!memcmp(code, "DWRT", 4) || !memcmp(code, "PROG", 4))
			{
				// Legacy system with fixed size chunks
				dataSize = 4;
			} else
			{
				dataSize = modularData.ReadUint32LE();
			}
			FileReader dataChunk = modularData.ReadChunk(dataSize);

			if(!memcmp(code, "DWRT", 4))
			{
				plugin.fDryRatio = dataChunk.ReadFloatLE();
			} else if(!memcmp(code, "PROG", 4))
			{
				plugin.defaultProgram = dataChunk.ReadUint32LE();
			} else if(!memcmp(code, "MCRO", 4))
			{
				// Read plugin-specific macros
				//dataChunk.ReadStructPartial(plugin.macros, dataChunk.GetLength());
			}
		}
	}
}
#endif // NO_PLUGINS


#ifndef MODPLUG_NO_FILESAVE

void CSoundFile::SaveExtendedSongProperties(FILE* f) const
{
	const CModSpecifications &specs = GetModSpecifications();
	// Extra song data - Yet Another Hack.
	mpt::IO::WriteIntLE<uint32>(f, MAGIC4BE('M','P','T','S'));

#define WRITEMODULARHEADER(code, fsize) \
	{ \
		mpt::IO::WriteIntLE<uint32>(f, code); \
		MPT_ASSERT(Util::TypeCanHoldValue<uint16>(fsize)); \
		const uint16 _size = fsize; \
		mpt::IO::WriteIntLE<uint16>(f, _size); \
	}
#define WRITEMODULAR(code, field) \
	{ \
		WRITEMODULARHEADER(code, sizeof(field)) \
		mpt::IO::WriteIntLE(f, field); \
	}

	if(m_nDefaultTempo.GetInt() > 255)
	{
		uint32 tempo = m_nDefaultTempo.GetInt();
		WRITEMODULAR(MAGIC4BE('D','T','.','.'), tempo);
	}
	if(m_nDefaultTempo.GetFract() != 0 && specs.hasFractionalTempo)
	{
		uint32 tempo = m_nDefaultTempo.GetFract();
		WRITEMODULAR(MAGIC4LE('D','T','F','R'), tempo);
	}

	WRITEMODULAR(MAGIC4BE('R','P','B','.'), m_nDefaultRowsPerBeat);
	WRITEMODULAR(MAGIC4BE('R','P','M','.'), m_nDefaultRowsPerMeasure);

	if(GetType() != MOD_TYPE_XM)
	{
		WRITEMODULAR(MAGIC4BE('C','.','.','.'), m_nChannels);
	}

	if((GetType() & (MOD_TYPE_IT | MOD_TYPE_MPT)) && GetNumChannels() > 64)
	{
		// IT header has only room for 64 channels. Save the settings that do not fit to the header here as an extension.
		WRITEMODULARHEADER(MAGIC4BE('C','h','n','S'), (GetNumChannels() - 64) * 2);
		for(CHANNELINDEX chn = 64; chn < GetNumChannels(); chn++)
		{
			uint8 panvol[2];
			panvol[0] = (uint8)(ChnSettings[chn].nPan >> 2);
			if (ChnSettings[chn].dwFlags[CHN_SURROUND]) panvol[0] = 100;
			if (ChnSettings[chn].dwFlags[CHN_MUTE]) panvol[0] |= 0x80;
			panvol[1] = (uint8)ChnSettings[chn].nVolume;
			mpt::IO::Write(f, panvol);
		}
	}

	{
		WRITEMODULARHEADER(MAGIC4BE('T','M','.','.'), 1);
		uint8 mode = static_cast<uint8>(m_nTempoMode);
		mpt::IO::WriteIntLE(f, mode);
	}

	const int32 tmpMixLevels = static_cast<int32>(m_nMixLevels);
	WRITEMODULAR(MAGIC4BE('P','M','M','.'), tmpMixLevels);

	if(m_dwCreatedWithVersion)
	{
		WRITEMODULAR(MAGIC4BE('C','W','V','.'), m_dwCreatedWithVersion);
	}

	WRITEMODULAR(MAGIC4BE('L','S','W','V'), MptVersion::num);
	WRITEMODULAR(MAGIC4BE('S','P','A','.'), m_nSamplePreAmp);
	WRITEMODULAR(MAGIC4BE('V','S','T','V'), m_nVSTiVolume);

	if(GetType() == MOD_TYPE_XM && m_nDefaultGlobalVolume != MAX_GLOBAL_VOLUME)
	{
		WRITEMODULAR(MAGIC4BE('D','G','V','.'), m_nDefaultGlobalVolume);
	}

	if(GetType() != MOD_TYPE_XM && Order().GetRestartPos() != 0)
	{
		WRITEMODULAR(MAGIC4BE('R','P','.','.'), Order().GetRestartPos());
	}

	if(m_nResampling != SRCMODE_DEFAULT && specs.hasDefaultResampling)
	{
		WRITEMODULAR(MAGIC4LE('R','S','M','P'), static_cast<uint32>(m_nResampling));
	}

	// Sample cues
	if(GetType() == MOD_TYPE_MPT)
	{
		for(SAMPLEINDEX smp = 1; smp <= GetNumSamples(); smp++)
		{
			const ModSample &sample = Samples[smp];
			if(sample.nLength && sample.HasCustomCuePoints())
			{
				// Write one chunk for every sample.
				// Rationale: chunks are limited to 65536 bytes, which can easily be reached
				// with the amount of samples that OpenMPT supports.
				WRITEMODULARHEADER(MAGIC4LE('C','U','E','S'), 2 + CountOf(sample.cues) * 4);
				mpt::IO::WriteIntLE<uint16>(f, smp);
				for(std::size_t i = 0; i < CountOf(sample.cues); i++)
				{
					mpt::IO::WriteIntLE<uint32>(f, sample.cues[i]);
				}
			}
		}
	}

	// Tempo Swing Factors
	if(!m_tempoSwing.empty())
	{
		mpt::ostringstream oStrm;
		TempoSwing::Serialize(oStrm, m_tempoSwing);
		std::string data = oStrm.str();
		uint16 length = mpt::saturate_cast<uint16>(data.size());
		WRITEMODULARHEADER(MAGIC4LE('S','W','N','G'), length);
		mpt::IO::WriteRaw(f, data.data(), length);
	}

	// Playback compatibility flags
	{
		uint8 bits[(kMaxPlayBehaviours + 7) / 8u];
		MemsetZero(bits);
		size_t maxBit = 0;
		for(size_t i = 0; i < kMaxPlayBehaviours; i++)
		{
			if(m_playBehaviour[i])
			{
				bits[i >> 3] |= 1 << (i & 0x07);
				maxBit = i + 8;
			}
		}
		uint16 numBytes = static_cast<uint16>(maxBit / 8u);
		WRITEMODULARHEADER(MAGIC4BE('M','S','F','.'), numBytes);
		mpt::IO::WriteRaw(f, bits, numBytes);
	}

	if(!m_songArtist.empty() && specs.hasArtistName)
	{
		std::string songArtistU8 = mpt::ToCharset(mpt::CharsetUTF8, m_songArtist);
		uint16 length = mpt::saturate_cast<uint16>(songArtistU8.length());
		WRITEMODULARHEADER(MAGIC4LE('A','U','T','H'), length);
		mpt::IO::WriteRaw(f, songArtistU8.c_str(), length);
	}

#ifdef MODPLUG_TRACKER
	// MIDI mapping directives
	if(GetMIDIMapper().GetCount() > 0)
	{
		const size_t objectsize = GetMIDIMapper().Serialize();
		if(!Util::TypeCanHoldValue<uint16>(objectsize))
		{
			AddToLog("Too many MIDI Mapping directives to save; data won't be written.");
		} else
		{
			WRITEMODULARHEADER(MAGIC4BE('M','I','M','A'), static_cast<uint16>(objectsize));
			GetMIDIMapper().Serialize(f);
		}
	}
#endif

#undef WRITEMODULAR
#undef WRITEMODULARHEADER
	return;
}

#endif // MODPLUG_NO_FILESAVE


template<typename T>
void ReadField(FileReader &chunk, std::size_t size, T &field)
{
	field = chunk.ReadSizedIntLE<T>(size);
}


template<typename T>
void ReadFieldCast(FileReader &chunk, std::size_t size, T &field)
{
	STATIC_ASSERT(sizeof(T) <= sizeof(int32));
	field = static_cast<T>(chunk.ReadSizedIntLE<int32>(size));
}


void CSoundFile::LoadExtendedSongProperties(FileReader &file, bool *pInterpretMptMade)
{
	if(!file.ReadMagic("STPM"))	// 'MPTS'
	{
		return;
	}

	// Found MPTS, interpret the file MPT made.
	if(pInterpretMptMade != nullptr)
		*pInterpretMptMade = true;

	// HACK: Reset mod flags to default values here, as they are not always written.
	m_playBehaviour.reset();

	while(file.CanRead(7))
	{
		const uint32 code = file.ReadUint32LE();
		const uint16 size = file.ReadUint16LE();

		// Start of MPTM extensions, non-ASCII ID or truncated field
		if(code == MAGIC4LE('2','2','8',4))
		{
			file.SkipBack(6);
			break;
		} else if((code & 0x80808080) || !(code & 0x60606060) || !file.CanRead(size))
		{
			break;
		}

		FileReader chunk = file.ReadChunk(size);

		switch (code)					// interpret field code
		{
			case MAGIC4BE('D','T','.','.'): { uint32 tempo; ReadField(chunk, size, tempo); m_nDefaultTempo.Set(tempo, m_nDefaultTempo.GetFract()); break; }
			case MAGIC4LE('D','T','F','R'): { uint32 tempoFract; ReadField(chunk, size, tempoFract); m_nDefaultTempo.Set(m_nDefaultTempo.GetInt(), tempoFract); break; }
			case MAGIC4BE('R','P','B','.'): ReadField(chunk, size, m_nDefaultRowsPerBeat); break;
			case MAGIC4BE('R','P','M','.'): ReadField(chunk, size, m_nDefaultRowsPerMeasure); break;
				// FIXME: If there are only PC events on the last few channels in an MPTM MO3, they won't be imported!
			case MAGIC4BE('C','.','.','.'): if(GetType() != MOD_TYPE_XM && m_ContainerType != MOD_CONTAINERTYPE_MO3) { CHANNELINDEX chn = 0; ReadField(chunk, size, chn); m_nChannels = Clamp(chn, m_nChannels, MAX_BASECHANNELS); } break;
			case MAGIC4BE('T','M','.','.'): ReadFieldCast(chunk, size, m_nTempoMode); break;
			case MAGIC4BE('P','M','M','.'): ReadFieldCast(chunk, size, m_nMixLevels); break;
			case MAGIC4BE('C','W','V','.'): ReadField(chunk, size, m_dwCreatedWithVersion); break;
			case MAGIC4BE('L','S','W','V'): { uint32 ver; ReadField(chunk, size, ver); if(ver != 0) { m_dwLastSavedWithVersion = ver; } break; }
			case MAGIC4BE('S','P','A','.'): ReadField(chunk, size, m_nSamplePreAmp); break;
			case MAGIC4BE('V','S','T','V'): ReadField(chunk, size, m_nVSTiVolume); break;
			case MAGIC4BE('D','G','V','.'): ReadField(chunk, size, m_nDefaultGlobalVolume); break;
			case MAGIC4BE('R','P','.','.'): if(GetType() != MOD_TYPE_XM) { ORDERINDEX restartPos; ReadField(chunk, size, restartPos); Order().SetRestartPos(restartPos); } break;
			case MAGIC4LE('R','S','M','P'):
				ReadFieldCast(chunk, size, m_nResampling);
				if(!IsKnownResamplingMode(m_nResampling)) m_nResampling = SRCMODE_DEFAULT;
				break;
#ifdef MODPLUG_TRACKER
			case MAGIC4BE('M','I','M','A'): GetMIDIMapper().Deserialize(chunk); break;
#endif
			case MAGIC4LE('A','U','T','H'):
				{
					std::string artist;
					chunk.ReadString<mpt::String::spacePadded>(artist, chunk.GetLength());
					m_songArtist = mpt::ToUnicode(mpt::CharsetUTF8, artist);
				}
				break;
			case MAGIC4BE('C','h','n','S'):
				// Channel settings for channels 65+
				if(size <= (MAX_BASECHANNELS - 64) * 2 && (size % 2u) == 0)
				{
					STATIC_ASSERT(CountOf(ChnSettings) >= 64);
					const CHANNELINDEX loopLimit = std::min(uint16(64 + size / 2), uint16(CountOf(ChnSettings)));

					for(CHANNELINDEX chn = 64; chn < loopLimit; chn++)
					{
						uint8 pan = chunk.ReadUint8(), vol = chunk.ReadUint8();
						if(pan != 0xFF)
						{
							ChnSettings[chn].nVolume = vol;
							ChnSettings[chn].nPan = 128;
							ChnSettings[chn].dwFlags.reset();
							if(pan & 0x80) ChnSettings[chn].dwFlags.set(CHN_MUTE);
							pan &= 0x7F;
							if(pan <= 64) ChnSettings[chn].nPan = pan << 2;
							if(pan == 100) ChnSettings[chn].dwFlags.set(CHN_SURROUND);
						}
					}
				}
				break;

			case MAGIC4LE('C','U','E','S'):
				// Sample cues
				if(size > 2)
				{
					SAMPLEINDEX smp = chunk.ReadUint16LE();
					if(smp > 0 && smp <= GetNumSamples())
					{
						ModSample &sample = Samples[smp];
						for(std::size_t i = 0; i < CountOf(sample.cues); i++)
						{
							sample.cues[i] = chunk.ReadUint32LE();
						}
					}
				}
				break;

			case MAGIC4LE('S','W','N','G'):
				// Tempo Swing Factors
				if(size > 2)
				{
					mpt::istringstream iStrm(chunk.ReadRawDataAsString());
					TempoSwing::Deserialize(iStrm, m_tempoSwing, chunk.GetLength());
				}
				break;

			case MAGIC4BE('M','S','F','.'):
				// Playback compatibility flags
				{
					size_t bit = 0;
					m_playBehaviour.reset();
					while(chunk.CanRead(1) && bit < m_playBehaviour.size())
					{
						uint8 b = chunk.ReadUint8();
						for(uint8 i = 0; i < 8; i++, bit++)
						{
							if((b & (1 << i)) && bit < m_playBehaviour.size())
							{
								m_playBehaviour.set(bit);
							}
						}
					}
				}
				break;
		}

	}

	// Validate read values.
	Limit(m_nDefaultTempo, GetModSpecifications().GetTempoMin(), GetModSpecifications().GetTempoMax());
	if(m_nDefaultRowsPerMeasure < m_nDefaultRowsPerBeat) m_nDefaultRowsPerMeasure = m_nDefaultRowsPerBeat;
	if(m_nTempoMode >= tempoModeMax) m_nTempoMode = tempoModeClassic;
	if(m_nMixLevels >= mixLevelsMax) m_nMixLevels = mixLevelsOriginal;
	//m_dwCreatedWithVersion
	//m_dwLastSavedWithVersion
	//m_nSamplePreAmp
	//m_nVSTiVolume
	//m_nDefaultGlobalVolume
	LimitMax(m_nDefaultGlobalVolume, MAX_GLOBAL_VOLUME);
	//m_nRestartPos
	//m_ModFlags
	if(!m_tempoSwing.empty()) m_tempoSwing.resize(m_nDefaultRowsPerBeat);
}


OPENMPT_NAMESPACE_END
