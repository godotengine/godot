/*
 * Sndfile.cpp
 * -----------
 * Purpose: Core class of the playback engine. Every song is represented by a CSoundFile object.
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/Mptrack.h"	// For CTrackApp::OpenURL
#include "../mptrack/TrackerSettings.h"
#include "../mptrack/Moddoc.h"
#include "../mptrack/Reporting.h"
#include "../mptrack/Mainfrm.h"
#endif // MODPLUG_TRACKER
#ifdef MPT_EXTERNAL_SAMPLES
#include "../common/mptFileIO.h"
#endif // MPT_EXTERNAL_SAMPLES
#include "../common/version.h"
#include "../soundlib/AudioCriticalSection.h"
#include "../common/mptIO.h"
#include "../common/serialization_utils.h"
#include "Sndfile.h"
#include "Tables.h"
#include "mod_specifications.h"
#include "tuningcollection.h"
#include "plugins/PluginManager.h"
#include "plugins/PlugInterface.h"
#include "../common/StringFixer.h"
#include "../common/FileReader.h"
#include "Container.h"
#include <sstream>
#include <time.h>

#ifndef NO_ARCHIVE_SUPPORT
#include "../unarchiver/unarchiver.h"
#endif // NO_ARCHIVE_SUPPORT


OPENMPT_NAMESPACE_BEGIN


mpt::ustring FileHistory::AsISO8601() const
{
	tm date = loadDate;
	if(openTime > 0)
	{
		// Calculate the date when editing finished.
		double openSeconds = (double)openTime / (double)HISTORY_TIMER_PRECISION;
		tm tmpLoadDate = loadDate;
		int64 loadDateSinceEpoch = mpt::Date::Unix::FromUTC(tmpLoadDate);
		int64 saveDateSinceEpoch = loadDateSinceEpoch + Util::Round<int64>(openSeconds);
		date = mpt::Date::Unix(saveDateSinceEpoch).AsUTC();
	}
	return mpt::Date::ToShortenedISO8601(date);
}


//////////////////////////////////////////////////////////
// CSoundFile

#ifdef MODPLUG_TRACKER
const NoteName *CSoundFile::m_NoteNames = NoteNamesFlat;
#endif

CSoundFile::CSoundFile() :
#ifndef MODPLUG_TRACKER
	m_NoteNames(NoteNamesSharp),
#endif
	m_pTuningsTuneSpecific(nullptr),
	m_pModSpecs(&ModSpecs::itEx),
	m_nType(MOD_TYPE_NONE),
	Patterns(*this),
	Order(*this),
#ifdef MODPLUG_TRACKER
	m_MIDIMapper(*this),
	m_pModDoc(nullptr),
#endif
	m_PRNG(mpt::make_prng<mpt::fast_prng>(mpt::global_prng())),
	visitedSongRows(*this),
	m_pCustomLog(nullptr)
{
	MemsetZero(MixSoundBuffer);
	MemsetZero(MixRearBuffer);
	MemsetZero(MixFloatBuffer);
	gnDryLOfsVol = 0;
	gnDryROfsVol = 0;
	m_nType = MOD_TYPE_NONE;
	m_ContainerType = MOD_CONTAINERTYPE_NONE;
	m_nMixChannels = 0;
	m_nSamples = 0;
	m_nInstruments = 0;
#ifndef MODPLUG_TRACKER
	m_nFreqFactor = m_nTempoFactor = 65536;
#endif
	m_nRepeatCount = 0;
	m_nTempoMode = tempoModeClassic;
	m_bIsRendering = false;

#ifdef MODPLUG_TRACKER
	m_lockRowStart = m_lockRowEnd = ROWINDEX_INVALID;
	m_lockOrderStart = m_lockOrderEnd = ORDERINDEX_INVALID;
	m_bChannelMuteTogglePending.reset();

	m_nDefaultRowsPerBeat = m_PlayState.m_nCurrentRowsPerBeat = (TrackerSettings::Instance().m_nRowHighlightBeats) ? TrackerSettings::Instance().m_nRowHighlightBeats : 4;
	m_nDefaultRowsPerMeasure = m_PlayState.m_nCurrentRowsPerMeasure = (TrackerSettings::Instance().m_nRowHighlightMeasures >= m_nDefaultRowsPerBeat) ? TrackerSettings::Instance().m_nRowHighlightMeasures : m_nDefaultRowsPerBeat * 4;
#else
	m_nDefaultRowsPerBeat = m_PlayState.m_nCurrentRowsPerBeat = 4;
	m_nDefaultRowsPerMeasure = m_PlayState.m_nCurrentRowsPerMeasure = 16;
#endif // MODPLUG_TRACKER

	MemsetZero(Instruments);
	MemsetZero(m_szNames);

	m_pTuningsTuneSpecific = new CTuningCollection();
}


CSoundFile::~CSoundFile()
{
	Destroy();
	delete m_pTuningsTuneSpecific;
	m_pTuningsTuneSpecific = nullptr;
}


void CSoundFile::AddToLog(LogLevel level, const mpt::ustring &text) const
{
	if(m_pCustomLog)
	{
		m_pCustomLog->AddToLog(level, text);
	} else
	{
		#ifdef MODPLUG_TRACKER
			if(GetpModDoc()) GetpModDoc()->AddToLog(level, text);
		#else
			MPT_LOG(level, "soundlib", text);
		#endif
	}
}


// Global variable initializer for loader functions
void CSoundFile::InitializeGlobals(MODTYPE type)
{
	// Do not add or change any of these values! And if you do, review each and every loader to check if they require these defaults!
	m_nType = type;

	MODTYPE bestType = GetBestSaveFormat();
	m_playBehaviour = GetDefaultPlaybackBehaviour(bestType);
	SetModSpecsPointer(m_pModSpecs, bestType);

	// Delete instruments in case some previously called loader already created them.
	for(INSTRUMENTINDEX i = 1; i <= m_nInstruments; i++)
	{
		delete Instruments[i];
		Instruments[i] = nullptr;
	}

	m_ContainerType = MOD_CONTAINERTYPE_NONE;
	m_nChannels = 0;
	m_nInstruments = 0;
	m_nSamples = 0;
	m_nSamplePreAmp = 48;
	m_nVSTiVolume = 48;
	m_nDefaultSpeed = 6;
	m_nDefaultTempo.Set(125);
	m_nDefaultGlobalVolume = MAX_GLOBAL_VOLUME;
	m_SongFlags.reset();
	m_nMinPeriod = 16;
	m_nMaxPeriod = 32767;
	m_nResampling = SRCMODE_DEFAULT;
	m_dwLastSavedWithVersion = m_dwCreatedWithVersion = 0;

	SetMixLevels(mixLevelsCompatible);

	Patterns.ClearPatterns();
	Order.Initialize();

	m_songName.clear();
	m_songArtist.clear();
	m_songMessage.clear();
	m_madeWithTracker.clear();
	m_FileHistory.clear();
	m_tempoSwing.clear();

	// Note: we do not use the Amiga resampler for DBM as it's a multichannel format and can make use of higher-quality Amiga soundcards instead of Paula.
	if(GetType() & (/*MOD_TYPE_DBM | */MOD_TYPE_DIGI | MOD_TYPE_MED | MOD_TYPE_MOD | MOD_TYPE_OKT | MOD_TYPE_SFX | MOD_TYPE_STP))
	{
		m_SongFlags.set(SONG_ISAMIGA);
	}
}


void CSoundFile::InitializeChannels()
{
	for(CHANNELINDEX nChn = 0; nChn < MAX_BASECHANNELS; nChn++)
	{
		InitChannel(nChn);
	}
}


CSoundFile::ProbeResult CSoundFile::ProbeAdditionalSize(MemoryFileReader &file, const uint64 *pfilesize, uint64 minimumAdditionalSize)
{
	const uint64 availableFileSize = file.GetLength();
	const uint64 fileSize = (pfilesize ? *pfilesize : file.GetLength());
	//const uint64 validFileSize = std::min<uint64>(fileSize, ProbeRecommendedSize);
	const uint64 goalSize = file.GetPosition() + minimumAdditionalSize;
	//const uint64 goalMinimumSize = std::min<uint64>(goalSize, ProbeRecommendedSize);
	if(pfilesize)
	{
		if(availableFileSize < std::min<uint64>(fileSize, ProbeRecommendedSize))
		{
			if(availableFileSize < goalSize)
			{
				return ProbeWantMoreData;
			}
		} else
		{
			if(fileSize < goalSize)
			{
				return ProbeFailure;
			}
		}
		return ProbeSuccess;
	}
	return ProbeSuccess;
}


const std::size_t CSoundFile::ProbeRecommendedSize = PROBE_RECOMMENDED_SIZE;


#define MPT_DO_PROBE( storedResult , call ) \
	MPT_DO { \
		ProbeResult lastResult = call ; \
		if(lastResult == ProbeSuccess) { \
			return ProbeSuccess; \
		} else if(lastResult == ProbeWantMoreData) { \
			storedResult = ProbeWantMoreData; \
		} \
	} MPT_WHILE_0 \
/**/


CSoundFile::ProbeResult CSoundFile::Probe(ProbeFlags flags, mpt::span<const mpt::byte> data, const uint64 *pfilesize)
{
	ProbeResult result = ProbeFailure;
	if(pfilesize && (*pfilesize < data.size()))
	{
		throw std::out_of_range("");
	}
	if(!data.data())
	{
		throw std::invalid_argument("");
	}
	MemoryFileReader file(data);
	if(flags & ProbeContainers)
	{
		MPT_DO_PROBE(result, ProbeFileHeaderMMCMP(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPP20(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderUMX(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderXPK(file, pfilesize));
	}
	if(flags & ProbeModules)
	{
		MPT_DO_PROBE(result, ProbeFileHeader669(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderAM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderAMF_Asylum(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderAMF_DSMI(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderAMS(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderAMS2(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderDBM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderDTM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderDIGI(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderDMF(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderDSM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderFAR(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderGDM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderICE(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderIMF(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderIT(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderITP(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderJ2B(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderM15(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMDL(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMED(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMO3(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMOD(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMT2(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderMTM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderOKT(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPLM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPSM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPSM16(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPT36(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderPTM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderS3M(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderSFX(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderSTM(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderSTP(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderULT(file, pfilesize));
		MPT_DO_PROBE(result, ProbeFileHeaderXM(file, pfilesize));
	}
	if(pfilesize)
	{
		if((result == ProbeWantMoreData) && (mpt::saturate_cast<std::size_t>(*pfilesize) <= data.size()))
		{
			// If the prober wants more data but we already reached EOF,
			// probing must fail.
			result = ProbeFailure;
		}
	} else
	{
		if((result == ProbeWantMoreData) && (data.size() >= ProbeRecommendedSize))
		{
			// If the prober wants more daat but we already provided the recommended required maximum,
			// just return success as this is th ebest we can do for the suggestesd probing size.
			result = ProbeSuccess;
		}
	}
	return result;
}


#ifdef MODPLUG_TRACKER
bool CSoundFile::Create(FileReader file, ModLoadingFlags loadFlags, CModDoc *pModDoc)
{
	m_pModDoc = pModDoc;
#else
bool CSoundFile::Create(FileReader file, ModLoadingFlags loadFlags)
{
#endif // MODPLUG_TRACKER

	m_nMixChannels = 0;
#ifndef MODPLUG_TRACKER
	m_nFreqFactor = m_nTempoFactor = 65536;
#endif

	MemsetZero(Instruments);
	MemsetZero(m_szNames);
#ifndef NO_PLUGINS
	std::fill(std::begin(m_MixPlugins), std::end(m_MixPlugins), SNDMIXPLUGIN());
#endif // NO_PLUGINS

	if(file.IsValid())
	{
		try
		{

#ifndef NO_ARCHIVE_SUPPORT
			CUnarchiver unarchiver(file);
			if(!(loadFlags & skipContainer))
			{
				if (unarchiver.ExtractBestFile(GetSupportedExtensions(true)))
				{
					file = unarchiver.GetOutputFile();
				}
			}
#endif

			std::vector<ContainerItem> containerItems;
			MODCONTAINERTYPE packedContainerType = MOD_CONTAINERTYPE_NONE;
			if(!(loadFlags & skipContainer))
			{
				ContainerLoadingFlags containerLoadFlags = (loadFlags == onlyVerifyHeader) ? ContainerOnlyVerifyHeader : ContainerUnwrapData;
				if(packedContainerType == MOD_CONTAINERTYPE_NONE && UnpackXPK(containerItems, file, containerLoadFlags)) packedContainerType = MOD_CONTAINERTYPE_XPK;
				if(packedContainerType == MOD_CONTAINERTYPE_NONE && UnpackPP20(containerItems, file, containerLoadFlags)) packedContainerType = MOD_CONTAINERTYPE_PP20;
				if(packedContainerType == MOD_CONTAINERTYPE_NONE && UnpackMMCMP(containerItems, file, containerLoadFlags)) packedContainerType = MOD_CONTAINERTYPE_MMCMP;
				if(packedContainerType == MOD_CONTAINERTYPE_NONE && UnpackUMX(containerItems, file, containerLoadFlags)) packedContainerType = MOD_CONTAINERTYPE_UMX;
				if(packedContainerType != MOD_CONTAINERTYPE_NONE)
				{
					if(loadFlags == onlyVerifyHeader)
					{
						return true;
					}
					if(!containerItems.empty())
					{
						file = containerItems[0].file;
					}
				}
			}

			if(loadFlags & skipModules)
			{
				return false;
			}

			if(!ReadXM(file, loadFlags)
			 && !ReadIT(file, loadFlags)
			 && !ReadS3M(file, loadFlags)
			 && !ReadSTM(file, loadFlags)
			 && !ReadMed(file, loadFlags)
			 && !ReadMTM(file, loadFlags)
			 && !ReadMDL(file, loadFlags)
			 && !ReadDBM(file, loadFlags)
			 && !ReadFAR(file, loadFlags)
			 && !ReadAMS(file, loadFlags)
			 && !ReadAMS2(file, loadFlags)
			 && !ReadOKT(file, loadFlags)
			 && !ReadPTM(file, loadFlags)
			 && !ReadUlt(file, loadFlags)
			 && !ReadDMF(file, loadFlags)
			 && !ReadDSM(file, loadFlags)
#if defined(MODPLUG_TRACKER) || defined(MPT_FUZZ_TRACKER)
			 && !ReadUAX(file, loadFlags)
#endif // MODPLUG_TRACKER || MPT_FUZZ_TRACKER
			 && !ReadAMF_Asylum(file, loadFlags)
			 && !ReadAMF_DSMI(file, loadFlags)
			 && !ReadPSM(file, loadFlags)
			 && !ReadPSM16(file, loadFlags)
			 && !ReadMT2(file, loadFlags)
			 && !ReadITProject(file, loadFlags)
#if defined(MODPLUG_TRACKER) || defined(MPT_FUZZ_TRACKER)
			 // this makes little sense for a module player library
			 && !ReadWav(file, loadFlags)
			 && !ReadMID(file, loadFlags)
#endif // MODPLUG_TRACKER || MPT_FUZZ_TRACKER
			 && !ReadGDM(file, loadFlags)
			 && !ReadIMF(file, loadFlags)
			 && !ReadDIGI(file, loadFlags)
			 && !ReadDTM(file, loadFlags)
			 && !ReadPLM(file, loadFlags)
			 && !ReadAM(file, loadFlags)
			 && !ReadJ2B(file, loadFlags)
			 && !ReadPT36(file, loadFlags)
			 && !ReadSFX(file, loadFlags)
			 && !ReadSTP(file, loadFlags)
			 && !ReadMod(file, loadFlags)
			 && !ReadICE(file, loadFlags)
			 && !Read669(file, loadFlags)
			 && !ReadMO3(file, loadFlags)
			 && !ReadM15(file, loadFlags))
			{
				m_nType = MOD_TYPE_NONE;
				m_ContainerType = MOD_CONTAINERTYPE_NONE;
				if(loadFlags == onlyVerifyHeader)
				{
					return false;
				}
			} else
			{
				if(loadFlags == onlyVerifyHeader)
				{
					return true;
				}
			}

			if(packedContainerType != MOD_CONTAINERTYPE_NONE && m_ContainerType == MOD_CONTAINERTYPE_NONE)
			{
				m_ContainerType = packedContainerType;
			}

			if(m_madeWithTracker.empty())
			{
				m_madeWithTracker = ModTypeToTracker(GetType());
			}

#ifndef NO_ARCHIVE_SUPPORT
			// Read archive comment if there is no song comment
			if(m_songMessage.empty())
			{
				m_songMessage.assign(mpt::ToCharset(mpt::CharsetLocale, unarchiver.GetComment()));
			}
#endif
		} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
		{
			MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
#ifdef MODPLUG_TRACKER
			return false;
#else
			// libopenmpt already handles this.
			throw;
#endif // MODPLUG_TRACKER
		}
	} else
	{
		// New song
		InitializeGlobals();
		m_dwCreatedWithVersion = MptVersion::num;
	}

	// Adjust channels
	for(CHANNELINDEX ich = 0; ich < MAX_BASECHANNELS; ich++)
	{
		LimitMax(ChnSettings[ich].nVolume, uint16(64));
		if (ChnSettings[ich].nPan > 256) ChnSettings[ich].nPan = 128;
		m_PlayState.Chn[ich].Reset(ModChannel::resetTotal, *this, ich);
	}

	// Checking samples, load external samples
	for(SAMPLEINDEX nSmp = 1; nSmp <= m_nSamples; nSmp++)
	{
		// Sanitize sample names
		mpt::String::SetNullTerminator(m_szNames[nSmp]);
		ModSample &sample = Samples[nSmp];

#ifdef MPT_EXTERNAL_SAMPLES
		if(SampleHasPath(nSmp))
		{
			mpt::PathString filename = GetSamplePath(nSmp);
			if(!file.GetFileName().empty())
			{
				filename = filename.RelativePathToAbsolute(file.GetFileName().GetPath());
			} else if(GetpModDoc() != nullptr)
			{
				filename = filename.RelativePathToAbsolute(GetpModDoc()->GetPathNameMpt().GetPath());
			}
			if(!LoadExternalSample(nSmp, filename))
			{
#ifndef MODPLUG_TRACKER
				// OpenMPT has its own way of reporting this error in CModDoc.
				AddToLog(LogError, mpt::format(MPT_USTRING("Unable to load sample %1: %2"))(i, filename.ToUnicode()));
#endif // MODPLUG_TRACKER
			}
		} else
		{
			sample.uFlags.reset(SMP_KEEPONDISK);
		}
#endif // MPT_EXTERNAL_SAMPLES

		if(sample.pSample)
		{
			sample.PrecomputeLoops(*this, false);
		} else if(!sample.uFlags[SMP_KEEPONDISK])
		{
			sample.nLength = 0;
			sample.nLoopStart = 0;
			sample.nLoopEnd = 0;
			sample.nSustainStart = 0;
			sample.nSustainEnd = 0;
			sample.uFlags.reset(CHN_LOOP | CHN_PINGPONGLOOP | CHN_SUSTAINLOOP | CHN_PINGPONGSUSTAIN);
		}
		if(sample.nGlobalVol > 64) sample.nGlobalVol = 64;
	}
	// Check invalid instruments
	INSTRUMENTINDEX maxInstr = 0;
	for(INSTRUMENTINDEX i = 0; i <= m_nInstruments; i++)
	{
		if(Instruments[i] != nullptr)
		{
			maxInstr = i;
			Instruments[i]->Sanitize(GetType());
		}
	}
	m_nInstruments = maxInstr;

	// Set default play state values
	if (!m_nDefaultTempo.GetInt()) m_nDefaultTempo.Set(125);
	if (!m_nDefaultSpeed) m_nDefaultSpeed = 6;
	m_PlayState.m_nMusicSpeed = m_nDefaultSpeed;
	m_PlayState.m_nMusicTempo = m_nDefaultTempo;
	m_PlayState.m_nCurrentRowsPerBeat = m_nDefaultRowsPerBeat;
	m_PlayState.m_nCurrentRowsPerMeasure = m_nDefaultRowsPerMeasure;
	m_PlayState.m_nGlobalVolume = static_cast<int32>(m_nDefaultGlobalVolume);
	m_PlayState.m_lHighResRampingGlobalVolume = m_PlayState.m_nGlobalVolume<<VOLUMERAMPPRECISION;
	m_PlayState.m_nGlobalVolumeDestination = m_PlayState.m_nGlobalVolume;
	m_PlayState.m_nSamplesToGlobalVolRampDest = 0;
	m_PlayState.m_nGlobalVolumeRampAmount = 0;
	m_PlayState.m_nNextOrder = 0;
	m_PlayState.m_nCurrentOrder = 0;
	m_PlayState.m_nPattern = 0;
	m_PlayState.m_nBufferCount = 0;
	m_PlayState.m_dBufferDiff = 0;
	m_PlayState.m_nTickCount = m_PlayState.m_nMusicSpeed;
	m_PlayState.m_nNextRow = 0;
	m_PlayState.m_nRow = 0;
	m_PlayState.m_nPatternDelay = 0;
	m_PlayState.m_nFrameDelay = 0;
	m_PlayState.m_nNextPatStartRow = 0;
	m_PlayState.m_nSeqOverride = ORDERINDEX_INVALID;

	m_nMaxOrderPosition = 0;

	RecalculateSamplesPerTick();
	visitedSongRows.Initialize(true);

	for(auto &order : Order)
	{
		order.Shrink();
		if(order.GetRestartPos() >= order.size())
		{
			order.SetRestartPos(0);
		}
	}

#ifndef NO_PLUGINS
	// Load plugins
#ifdef MODPLUG_TRACKER
	std::string notFoundText;
#endif // MODPLUG_TRACKER
	std::vector<const SNDMIXPLUGININFO *> notFoundIDs;

	if((loadFlags & (loadPluginData | loadPluginInstance)) == (loadPluginData | loadPluginInstance))
	{
		for(PLUGINDEX plug = 0; plug < MAX_MIXPLUGINS; plug++)
		{
			auto &plugin = m_MixPlugins[plug];
			if(plugin.IsValidPlugin())
			{
#ifdef MODPLUG_TRACKER
				// Provide some visual feedback
				{
					mpt::ustring s = mpt::format(MPT_USTRING("Loading Plugin FX%1: %2 (%3)"))(
						mpt::ufmt::dec0<2>(plug + 1),
						mpt::ToUnicode(mpt::CharsetUTF8, plugin.Info.szLibraryName),
						mpt::ToUnicode(mpt::CharsetLocale, plugin.Info.szName));
					CMainFrame::GetMainFrame()->SetHelpText(mpt::ToCString(s));
				}
#endif // MODPLUG_TRACKER
				CreateMixPluginProc(plugin, *this);
				if(plugin.pMixPlugin)
				{
					// Plugin was found
					plugin.pMixPlugin->RestoreAllParameters(plugin.defaultProgram);
				} else
				{
					// Plugin not found - add to list
					bool found = std::find_if(notFoundIDs.cbegin(), notFoundIDs.cend(),
						[&plugin](const SNDMIXPLUGININFO *info) { return info->dwPluginId2 == plugin.Info.dwPluginId2 && info->dwPluginId1 == plugin.Info.dwPluginId1; }) != notFoundIDs.cend();

					if(!found)
					{
						notFoundIDs.push_back(&plugin.Info);
#ifdef MODPLUG_TRACKER
						notFoundText.append(plugin.GetLibraryName());
						notFoundText.append("\n");
#else
						AddToLog(LogWarning, MPT_USTRING("Plugin not found: ") + mpt::ToUnicode(mpt::CharsetUTF8, plugin.GetLibraryName()));
#endif // MODPLUG_TRACKER
					}
				}
			}
		}
	}

#ifdef MODPLUG_TRACKER
	// Display a nice message so the user sees which plugins are missing
	// TODO: Use IDD_MODLOADING_WARNINGS dialog (NON-MODAL!) to display all warnings that are encountered when loading a module.
	if(!notFoundIDs.empty())
	{
		if(notFoundIDs.size() == 1)
		{
			notFoundText = "The following plugin has not been found:\n\n" + notFoundText + "\nDo you want to search for it online?";
		} else
		{
			notFoundText = "The following plugins have not been found:\n\n" + notFoundText + "\nDo you want to search for them online?";
		}
		if (Reporting::Confirm(mpt::ToWide(mpt::CharsetUTF8, notFoundText.c_str()), L"OpenMPT - Plugins missing", false, true) == cnfYes)
		{
			std::string url = "https://resources.openmpt.org/plugins/search.php?p=";
			for(auto &id : notFoundIDs)
			{
				url += mpt::fmt::HEX0<8>(id->dwPluginId2.get());
				url += id->szLibraryName;
				url += "%0a";
			}
			CTrackApp::OpenURL(mpt::PathString::FromUTF8(url));
		}
	}
#endif // MODPLUG_TRACKER
#endif // NO_PLUGINS

	// Set up mix levels
	SetMixLevels(m_nMixLevels);

	if(GetType() == MOD_TYPE_NONE)
	{
		return false;
	}

	SetModSpecsPointer(m_pModSpecs, GetBestSaveFormat());

	// When reading a file made with an older version of MPT, it might be necessary to upgrade some settings automatically.
	if(m_dwLastSavedWithVersion)
	{
		UpgradeModule();
	}
	return true;
}


bool CSoundFile::Destroy()
{
	for(auto &chn : m_PlayState.Chn)
	{
		chn.pModInstrument = nullptr;
		chn.pModSample = nullptr;
		chn.pCurrentSample = nullptr;
		chn.nLength = 0;
	}

	Patterns.DestroyPatterns();

	m_songName.clear();
	m_songArtist.clear();
	m_songMessage.clear();
	m_madeWithTracker.clear();
	m_FileHistory.clear();

	for(auto &smp : Samples)
	{
		smp.FreeSample();
	}
	for(auto &ins : Instruments)
	{
		delete ins;
		ins = nullptr;
	}
#ifndef NO_PLUGINS
	for(auto &plug : m_MixPlugins)
	{
		plug.Destroy();
	}
#endif // NO_PLUGINS

	m_nType = MOD_TYPE_NONE;
	m_ContainerType = MOD_CONTAINERTYPE_NONE;
	m_nChannels = m_nSamples = m_nInstruments = 0;
	return true;
}


//////////////////////////////////////////////////////////////////////////
// Misc functions


void CSoundFile::SetDspEffects(uint32 DSPMask)
{
#ifdef ENABLE_ASM
#ifndef NO_REVERB
	if(!(GetProcSupport() & PROCSUPPORT_MMX)) DSPMask &= ~SNDDSP_REVERB;
#endif
#endif
	m_MixerSettings.DSPMask = DSPMask;
	InitPlayer(false);
}


void CSoundFile::SetPreAmp(uint32 nVol)
{
	if (nVol < 1) nVol = 1;
	if (nVol > 0x200) nVol = 0x200;	// x4 maximum
#ifndef NO_AGC
	if ((nVol < m_MixerSettings.m_nPreAmp) && (nVol) && (m_MixerSettings.DSPMask & SNDDSP_AGC))
	{
		m_AGC.Adjust(m_MixerSettings.m_nPreAmp, nVol);
	}
#endif
	m_MixerSettings.m_nPreAmp = nVol;
}


double CSoundFile::GetCurrentBPM() const
{
	double bpm;

	if (m_nTempoMode == tempoModeModern)
	{
		// With modern mode, we trust that true bpm is close enough to what user chose.
		// This avoids oscillation due to tick-to-tick corrections.
		bpm = m_PlayState.m_nMusicTempo.ToDouble();
	} else
	{
		//with other modes, we calculate it:
		double ticksPerBeat = m_PlayState.m_nMusicSpeed * m_PlayState.m_nCurrentRowsPerBeat; //ticks/beat = ticks/row * rows/beat
		double samplesPerBeat = m_PlayState.m_nSamplesPerTick * ticksPerBeat;                //samps/beat = samps/tick * ticks/beat
		bpm =  m_MixerSettings.gdwMixingFreq / samplesPerBeat * 60;                          //beats/sec  = samps/sec  / samps/beat
	}	                                                                                     //beats/min  =  beats/sec * 60

	return bpm;
}


void CSoundFile::ResetPlayPos()
{
	for(CHANNELINDEX i = 0; i < MAX_CHANNELS; i++)
		m_PlayState.Chn[i].Reset(ModChannel::resetSetPosFull, *this, i);

	visitedSongRows.Initialize(true);
	m_SongFlags.reset(SONG_FADINGSONG | SONG_ENDREACHED);

	m_PlayState.m_nGlobalVolume = m_nDefaultGlobalVolume;
	m_PlayState.m_nMusicSpeed = m_nDefaultSpeed;
	m_PlayState.m_nMusicTempo = m_nDefaultTempo;

	// do not ramp global volume when starting playback
	m_PlayState.m_lHighResRampingGlobalVolume = m_PlayState.m_nGlobalVolume<<VOLUMERAMPPRECISION;
	m_PlayState.m_nGlobalVolumeDestination = m_PlayState.m_nGlobalVolume;
	m_PlayState.m_nSamplesToGlobalVolRampDest = 0;
	m_PlayState.m_nGlobalVolumeRampAmount = 0;

	m_PlayState.m_nNextOrder = 0;
	m_PlayState.m_nNextRow = 0;
	m_PlayState.m_nTickCount = m_PlayState.m_nMusicSpeed;
	m_PlayState.m_nBufferCount = 0;
	m_PlayState.m_nPatternDelay = 0;
	m_PlayState.m_nFrameDelay = 0;
	m_PlayState.m_nNextPatStartRow = 0;
	m_PlayState.m_lTotalSampleCount = 0;
	//m_nSeqOverride = 0;
}



void CSoundFile::SetCurrentOrder(ORDERINDEX nOrder)
{
	while ((nOrder < Order().size()) && (Order()[nOrder] == Order.GetIgnoreIndex())) nOrder++;
	if ((nOrder >= Order().size()) || (Order()[nOrder] >= Patterns.Size())) return;
	for(auto &chn : m_PlayState.Chn)
	{
		chn.nPeriod = 0;
		chn.nNote = NOTE_NONE;
		chn.nPortamentoDest = 0;
		chn.nCommand = 0;
		chn.nPatternLoopCount = 0;
		chn.nPatternLoop = 0;
		chn.nVibratoPos = chn.nTremoloPos = chn.nPanbrelloPos = 0;
		//IT compatibility 15. Retrigger
		if(m_playBehaviour[kITRetrigger])
		{
			chn.nRetrigCount = 0;
			chn.nRetrigParam = 1;
		}
		chn.nTremorCount = 0;
	}

#ifndef NO_PLUGINS
	// Stop hanging notes from VST instruments as well
	StopAllVsti();
#endif // NO_PLUGINS

	if (!nOrder)
	{
		ResetPlayPos();
	} else
	{
		m_PlayState.m_nNextOrder = nOrder;
		m_PlayState.m_nRow = m_PlayState.m_nNextRow = 0;
		m_PlayState.m_nPattern = 0;
		m_PlayState.m_nTickCount = m_PlayState.m_nMusicSpeed;
		m_PlayState.m_nBufferCount = 0;
		m_PlayState.m_nPatternDelay = 0;
		m_PlayState.m_nFrameDelay = 0;
		m_PlayState.m_nNextPatStartRow = 0;
	}

	m_SongFlags.reset(SONG_FADINGSONG | SONG_ENDREACHED);
}

void CSoundFile::SuspendPlugins()
{
#ifndef NO_PLUGINS
	for(auto &plug : m_MixPlugins)
	{
		IMixPlugin *pPlugin = plug.pMixPlugin;
		if(pPlugin != nullptr && pPlugin->IsResumed())
		{
			pPlugin->NotifySongPlaying(false);
			pPlugin->HardAllNotesOff();
			pPlugin->Suspend();
		}
	}
#endif // NO_PLUGINS
}

void CSoundFile::ResumePlugins()
{
#ifndef NO_PLUGINS
	for(auto &plugin : m_MixPlugins)
	{
		IMixPlugin *pPlugin = plugin.pMixPlugin;
		if(pPlugin != nullptr && !pPlugin->IsResumed())
		{
			pPlugin->NotifySongPlaying(true);
			pPlugin->Resume();
		}
	}
#endif // NO_PLUGINS
}


void CSoundFile::StopAllVsti()
{
#ifndef NO_PLUGINS
	for(auto &plugin : m_MixPlugins)
	{
		IMixPlugin *pPlugin = plugin.pMixPlugin;
		if(pPlugin != nullptr && pPlugin->IsResumed())
		{
			pPlugin->HardAllNotesOff();
		}
	}
#endif // NO_PLUGINS
}


void CSoundFile::SetMixLevels(MixLevels levels)
{
	m_nMixLevels = levels;
	m_PlayConfig.SetMixLevels(m_nMixLevels);
	RecalculateGainForAllPlugs();
}


void CSoundFile::RecalculateGainForAllPlugs()
{
#ifndef NO_PLUGINS
	for(auto &plugin : m_MixPlugins)
	{
		if(plugin.pMixPlugin != nullptr)
			plugin.pMixPlugin->RecalculateGain();
	}
#endif // NO_PLUGINS
}


void CSoundFile::ResetChannels()
{
	m_SongFlags.reset(SONG_FADINGSONG | SONG_ENDREACHED);
	m_PlayState.m_nBufferCount = 0;
	for(auto &chn : m_PlayState.Chn)
	{
		chn.nROfs = chn.nLOfs = 0;
		chn.nLength = 0;
	}
}


#ifdef MODPLUG_TRACKER

void CSoundFile::PatternTranstionChnSolo(const CHANNELINDEX chnIndex)
{
	if(chnIndex >= m_nChannels)
		return;

	for(CHANNELINDEX i = 0; i < m_nChannels; i++)
	{
		m_bChannelMuteTogglePending[i] = !ChnSettings[i].dwFlags[CHN_MUTE];
	}
	m_bChannelMuteTogglePending[chnIndex] = ChnSettings[chnIndex].dwFlags[CHN_MUTE];
}


void CSoundFile::PatternTransitionChnUnmuteAll()
{
	for(CHANNELINDEX i = 0; i < m_nChannels; i++)
	{
		m_bChannelMuteTogglePending[i] = ChnSettings[i].dwFlags[CHN_MUTE];
	}
}

#endif // MODPLUG_TRACKER


void CSoundFile::LoopPattern(PATTERNINDEX nPat, ROWINDEX nRow)
{
	if(!Patterns.IsValidPat(nPat))
	{
		m_SongFlags.reset(SONG_PATTERNLOOP);
	} else
	{
		if(nRow >= Patterns[nPat].GetNumRows()) nRow = 0;
		m_PlayState.m_nPattern = nPat;
		m_PlayState.m_nRow = m_PlayState.m_nNextRow = nRow;
		m_PlayState.m_nTickCount = m_PlayState.m_nMusicSpeed;
		m_PlayState.m_nPatternDelay = 0;
		m_PlayState.m_nFrameDelay = 0;
		m_PlayState.m_nBufferCount = 0;
		m_PlayState.m_nNextPatStartRow = 0;
		m_SongFlags.set(SONG_PATTERNLOOP);
	}
}


void CSoundFile::DontLoopPattern(PATTERNINDEX nPat, ROWINDEX nRow)
{
	if(!Patterns.IsValidPat(nPat)) nPat = 0;
	if(nRow >= Patterns[nPat].GetNumRows()) nRow = 0;
	m_PlayState.m_nPattern = nPat;
	m_PlayState.m_nRow = m_PlayState.m_nNextRow = nRow;
	m_PlayState.m_nTickCount = m_PlayState.m_nMusicSpeed;
	m_PlayState.m_nPatternDelay = 0;
	m_PlayState.m_nFrameDelay = 0;
	m_PlayState.m_nBufferCount = 0;
	m_PlayState.m_nNextPatStartRow = 0;
	m_SongFlags.reset(SONG_PATTERNLOOP);
}


void CSoundFile::SetDefaultPlaybackBehaviour(MODTYPE type)
{
	m_playBehaviour = GetDefaultPlaybackBehaviour(type);
}


PlayBehaviourSet CSoundFile::GetSupportedPlaybackBehaviour(MODTYPE type)
{
	PlayBehaviourSet playBehaviour;
	switch(type)
	{
	case MOD_TYPE_MPT:
	case MOD_TYPE_IT:
		playBehaviour.set(MSF_COMPATIBLE_PLAY);
		playBehaviour.set(kHertzInLinearMode);
		playBehaviour.set(kTempoClamp);
		playBehaviour.set(kPerChannelGlobalVolSlide);
		playBehaviour.set(kPanOverride);
		playBehaviour.set(kITInstrWithoutNote);
		playBehaviour.set(kITVolColFinePortamento);
		playBehaviour.set(kITArpeggio);
		playBehaviour.set(kITOutOfRangeDelay);
		playBehaviour.set(kITPortaMemoryShare);
		playBehaviour.set(kITPatternLoopTargetReset);
		playBehaviour.set(kITFT2PatternLoop);
		playBehaviour.set(kITPingPongNoReset);
		playBehaviour.set(kITEnvelopeReset);
		playBehaviour.set(kITClearOldNoteAfterCut);
		playBehaviour.set(kITVibratoTremoloPanbrello);
		playBehaviour.set(kITTremor);
		playBehaviour.set(kITRetrigger);
		playBehaviour.set(kITMultiSampleBehaviour);
		playBehaviour.set(kITPortaTargetReached);
		playBehaviour.set(kITPatternLoopBreak);
		playBehaviour.set(kITOffset);
		playBehaviour.set(kITSwingBehaviour);
		playBehaviour.set(kITNNAReset);
		playBehaviour.set(kITSCxStopsSample);
		playBehaviour.set(kITEnvelopePositionHandling);
		playBehaviour.set(kITPortamentoInstrument);
		playBehaviour.set(kITPingPongMode);
		playBehaviour.set(kITRealNoteMapping);
		playBehaviour.set(kITHighOffsetNoRetrig);
		playBehaviour.set(kITFilterBehaviour);
		playBehaviour.set(kITNoSurroundPan);
		playBehaviour.set(kITShortSampleRetrig);
		playBehaviour.set(kITPortaNoNote);
		playBehaviour.set(kITDontResetNoteOffOnPorta);
		playBehaviour.set(kITVolColMemory);
		playBehaviour.set(kITPortamentoSwapResetsPos);
		playBehaviour.set(kITEmptyNoteMapSlot);
		playBehaviour.set(kITFirstTickHandling);
		playBehaviour.set(kITSampleAndHoldPanbrello);
		playBehaviour.set(kITClearPortaTarget);
		playBehaviour.set(kITPanbrelloHold);
		playBehaviour.set(kITPanningReset);
		playBehaviour.set(kITPatternLoopWithJumps);
		playBehaviour.set(kITInstrWithNoteOff);
		playBehaviour.set(kITMultiSampleInstrumentNumber);
		playBehaviour.set(kRowDelayWithNoteDelay);
		break;

	case MOD_TYPE_XM:
		playBehaviour.set(MSF_COMPATIBLE_PLAY);
		playBehaviour.set(kFT2VolumeRamping);
		playBehaviour.set(kTempoClamp);
		playBehaviour.set(kPerChannelGlobalVolSlide);
		playBehaviour.set(kPanOverride);
		playBehaviour.set(kITFT2PatternLoop);
		playBehaviour.set(kFT2Arpeggio);
		playBehaviour.set(kFT2Retrigger);
		playBehaviour.set(kFT2VolColVibrato);
		playBehaviour.set(kFT2PortaNoNote);
		playBehaviour.set(kFT2KeyOff);
		playBehaviour.set(kFT2PanSlide);
		playBehaviour.set(kFT2OffsetOutOfRange);
		playBehaviour.set(kFT2RestrictXCommand);
		playBehaviour.set(kFT2RetrigWithNoteDelay);
		playBehaviour.set(kFT2SetPanEnvPos);
		playBehaviour.set(kFT2PortaIgnoreInstr);
		playBehaviour.set(kFT2VolColMemory);
		playBehaviour.set(kFT2LoopE60Restart);
		playBehaviour.set(kFT2ProcessSilentChannels);
		playBehaviour.set(kFT2ReloadSampleSettings);
		playBehaviour.set(kFT2PortaDelay);
		playBehaviour.set(kFT2Transpose);
		playBehaviour.set(kFT2PatternLoopWithJumps);
		playBehaviour.set(kFT2PortaTargetNoReset);
		playBehaviour.set(kFT2EnvelopeEscape);
		playBehaviour.set(kFT2Tremor);
		playBehaviour.set(kFT2OutOfRangeDelay);
		playBehaviour.set(kFT2Periods);
		playBehaviour.set(kFT2PanWithDelayedNoteOff);
		playBehaviour.set(kFT2VolColDelay);
		playBehaviour.set(kFT2FinetunePrecision);
		playBehaviour.set(kFT2NoteOffFlags);
		playBehaviour.set(kRowDelayWithNoteDelay);
		playBehaviour.set(kFT2TremoloRampWaveform);
		playBehaviour.set(kFT2PortaUpDownMemory);
		break;

	case MOD_TYPE_S3M:
		playBehaviour.set(MSF_COMPATIBLE_PLAY);
		playBehaviour.set(kTempoClamp);
		playBehaviour.set(kPanOverride);
		playBehaviour.set(kITPanbrelloHold);
		playBehaviour.set(kST3NoMutedChannels);
		playBehaviour.set(kST3PortaSampleChange);
		playBehaviour.set(kST3EffectMemory);
		playBehaviour.set(kST3VibratoMemory);
		playBehaviour.set(KST3PortaAfterArpeggio);
		playBehaviour.set(kRowDelayWithNoteDelay);
		break;

	case MOD_TYPE_MOD:
		playBehaviour.set(kMODVBlankTiming);
		playBehaviour.set(kMODOneShotLoops);
		playBehaviour.set(kMODIgnorePanning);
		playBehaviour.set(kMODSampleSwap);
		playBehaviour.set(kMODOutOfRangeNoteDelay);
		playBehaviour.set(kMODTempoOnSecondTick);
		playBehaviour.set(kRowDelayWithNoteDelay);
		break;

	default:
		playBehaviour.set(MSF_COMPATIBLE_PLAY);
		playBehaviour.set(kHertzInLinearMode);
		playBehaviour.set(kTempoClamp);
		playBehaviour.set(kPanOverride);
		break;
	}
	return playBehaviour;
}


PlayBehaviourSet CSoundFile::GetDefaultPlaybackBehaviour(MODTYPE type)
{
	PlayBehaviourSet playBehaviour;
	switch(type)
	{
	case MOD_TYPE_MPT:
		playBehaviour.set(kHertzInLinearMode);
		playBehaviour.set(kPerChannelGlobalVolSlide);
		playBehaviour.set(kPanOverride);
		playBehaviour.set(kITMultiSampleBehaviour);
		playBehaviour.set(kITSampleAndHoldPanbrello);
		playBehaviour.set(kITPanbrelloHold);
		break;

	case MOD_TYPE_XM:
		playBehaviour = GetSupportedPlaybackBehaviour(type);
		// Only set this explicitely for FT2-made XMs.
		playBehaviour.reset(kFT2VolumeRamping);
		break;

	case MOD_TYPE_MOD:
		playBehaviour.set(kMODSampleSwap);
		playBehaviour.set(kRowDelayWithNoteDelay);
		break;

	default:
		playBehaviour = GetSupportedPlaybackBehaviour(type);
		break;
	}
	return playBehaviour;
}


MODTYPE CSoundFile::GetBestSaveFormat() const
{
	switch(GetType())
	{
	case MOD_TYPE_MOD:
	case MOD_TYPE_S3M:
	case MOD_TYPE_XM:
	case MOD_TYPE_IT:
	case MOD_TYPE_MPT:
		return GetType();
	case MOD_TYPE_AMF0:
	case MOD_TYPE_DIGI:
	case MOD_TYPE_SFX:
	case MOD_TYPE_STP:
		return MOD_TYPE_MOD;
	case MOD_TYPE_MED:
		if(m_nDefaultTempo == TEMPO(125, 0) && m_nDefaultSpeed == 6 && !m_nInstruments)
		{
			for(const auto &pat : Patterns)
			{
				if(pat.IsValid() && pat.GetNumRows() != 64)
					return MOD_TYPE_XM;
			}
			return MOD_TYPE_MOD;
		}
		return MOD_TYPE_XM;
	case MOD_TYPE_PSM:
		if(GetNumChannels() > 16)
			return MOD_TYPE_IT;
		for(CHANNELINDEX i = 0; i < GetNumChannels(); i++)
		{
			if(ChnSettings[i].dwFlags[CHN_SURROUND] || ChnSettings[i].nVolume != 64)
			{
				return MOD_TYPE_IT;
				break;
			}
		}
		return MOD_TYPE_S3M;
	case MOD_TYPE_669:
	case MOD_TYPE_FAR:
	case MOD_TYPE_STM:
	case MOD_TYPE_DSM:
	case MOD_TYPE_AMF:
	case MOD_TYPE_MTM:
		return MOD_TYPE_S3M;
	case MOD_TYPE_AMS:
	case MOD_TYPE_AMS2:
	case MOD_TYPE_DMF:
	case MOD_TYPE_DBM:
	case MOD_TYPE_IMF:
	case MOD_TYPE_J2B:
	case MOD_TYPE_ULT:
	case MOD_TYPE_OKT:
	case MOD_TYPE_MT2:
	case MOD_TYPE_MDL:
	case MOD_TYPE_PTM:
	case MOD_TYPE_DTM:
	default:
		return MOD_TYPE_IT;
	case MOD_TYPE_MID:
		return MOD_TYPE_MPT;
	}
}


const char *CSoundFile::GetSampleName(SAMPLEINDEX nSample) const
{
	MPT_ASSERT(nSample <= GetNumSamples());
	if (nSample < MAX_SAMPLES)
	{
		return m_szNames[nSample];
	} else
	{
		return "";
	}
}


const char *CSoundFile::GetInstrumentName(INSTRUMENTINDEX nInstr) const
{
	if((nInstr >= MAX_INSTRUMENTS) || (!Instruments[nInstr]))
		return "";

	MPT_ASSERT(nInstr <= GetNumInstruments());
	return Instruments[nInstr]->name;
}


bool CSoundFile::InitChannel(CHANNELINDEX nChn)
{
	if(nChn >= MAX_BASECHANNELS) return true;

	ChnSettings[nChn].Reset();
	m_PlayState.Chn[nChn].Reset(ModChannel::resetTotal, *this, nChn);

#ifdef MODPLUG_TRACKER
	if(GetpModDoc() != nullptr)
	{
		GetpModDoc()->Record1Channel(nChn, false);
		GetpModDoc()->Record2Channel(nChn, false);
	}
#endif // MODPLUG_TRACKER

#ifdef MODPLUG_TRACKER
	m_bChannelMuteTogglePending[nChn] = false;
#endif // MODPLUG_TRACKER

	return false;
}


void CSoundFile::InitAmigaResampler()
{
	if(m_SongFlags[SONG_ISAMIGA] && m_Resampler.m_Settings.emulateAmiga)
	{
		const Paula::State defaultState(GetSampleRate());
		for(auto &chn : m_PlayState.Chn)
		{
			chn.paulaState = defaultState;
		}
	}
}


// Detect samples that are referenced by an instrument, but actually not used in a song.
// Only works in instrument mode. Unused samples are marked as false in the vector.
SAMPLEINDEX CSoundFile::DetectUnusedSamples(std::vector<bool> &sampleUsed) const
{
	sampleUsed.assign(GetNumSamples() + 1, false);

	if(GetNumInstruments() == 0)
	{
		return 0;
	}
	SAMPLEINDEX unused = 0;
	std::vector<ModCommand::INSTR> lastIns;

	for(const auto &pat : Patterns) if(pat.IsValid())
	{
		lastIns.assign(GetNumChannels(), 0);
		auto p = pat.cbegin();
		for(ROWINDEX row = 0; row < pat.GetNumRows(); row++)
		{
			for(CHANNELINDEX c = 0; c < GetNumChannels(); c++, p++)
			{
				if(p->IsNote())
				{
					ModCommand::INSTR instr = p->instr;
					if(!p->instr) instr = lastIns[c];
					if (instr)
					{
						if(IsInRange(instr, (INSTRUMENTINDEX)0, MAX_INSTRUMENTS))
						{
							ModInstrument *pIns = Instruments[p->instr];
							if (pIns)
							{
								SAMPLEINDEX n = pIns->Keyboard[p->note - NOTE_MIN];
								if (n <= GetNumSamples()) sampleUsed[n] = true;
							}
						}
						lastIns[c] = instr;
					} else
					{
						// No idea which instrument this note belongs to, so mark it used in any instruments.
						for (INSTRUMENTINDEX k = GetNumInstruments(); k >= 1; k--)
						{
							ModInstrument *pIns = Instruments[k];
							if (pIns)
							{
								SAMPLEINDEX n = pIns->Keyboard[p->note - NOTE_MIN];
								if (n <= GetNumSamples()) sampleUsed[n] = true;
							}
						}
					}
				}
			}
		}
	}
	for (SAMPLEINDEX ichk = GetNumSamples(); ichk >= 1; ichk--)
	{
		if ((!sampleUsed[ichk]) && (Samples[ichk].pSample)) unused++;
	}

	return unused;
}


// Destroy samples where keepSamples index is false. First sample is keepSamples[1]!
SAMPLEINDEX CSoundFile::RemoveSelectedSamples(const std::vector<bool> &keepSamples)
{
	if(keepSamples.empty())
	{
		return 0;
	}

	SAMPLEINDEX nRemoved = 0;
	for(SAMPLEINDEX nSmp = std::min(GetNumSamples(), static_cast<SAMPLEINDEX>(keepSamples.size() - 1)); nSmp >= 1; nSmp--)
	{
		if(!keepSamples[nSmp])
		{
			CriticalSection cs;

#ifdef MODPLUG_TRACKER
			if(GetpModDoc())
			{
				GetpModDoc()->GetSampleUndo().PrepareUndo(nSmp, sundo_replace, "Remove Sample");
			}
#endif // MODPLUG_TRACKER

			if(DestroySample(nSmp))
			{
				strcpy(m_szNames[nSmp], "");
				nRemoved++;
			}
			if((nSmp == GetNumSamples()) && (nSmp > 1)) m_nSamples--;
		}
	}
	return nRemoved;
}


bool CSoundFile::DestroySample(SAMPLEINDEX nSample)
{
	if(!nSample || nSample >= MAX_SAMPLES)
	{
		return false;
	}
	if(Samples[nSample].pSample == nullptr)
	{
		return true;
	}

	ModSample &sample = Samples[nSample];

	for(auto &chn : m_PlayState.Chn)
	{
		if(chn.pModSample == &sample)
		{
			chn.position.Set(0);
			chn.nLength = 0;
			chn.pCurrentSample = nullptr;
		}
	}

	sample.FreeSample();
	sample.nLength = 0;
	sample.uFlags.reset(CHN_16BIT | CHN_STEREO);

#ifdef MODPLUG_TRACKER
	ResetSamplePath(nSample);
#endif
	return true;
}


bool CSoundFile::DestroySampleThreadsafe(SAMPLEINDEX nSample)
{
	CriticalSection cs;
	return DestroySample(nSample);
}


CTuning* CSoundFile::CreateTuning12TET(const std::string &name)
{
	CTuning* pT = CTuning::CreateGeometric(name, 12, 2, 15);
	for(ModCommand::NOTE note = 0; note < 12; ++note)
	{
		pT->SetNoteName(note, NoteNamesSharp[note]);
	}
	return pT;
}


std::string CSoundFile::GetNoteName(const ModCommand::NOTE note, const INSTRUMENTINDEX inst) const
{
	// For MPTM instruments with custom tuning, find the appropriate note name. Else, use default note names.
	if(ModCommand::IsNote(note) && GetType() == MOD_TYPE_MPT && inst >= 1 && inst <= GetNumInstruments() && Instruments[inst] && Instruments[inst]->pTuning)
	{
		return Instruments[inst]->pTuning->GetNoteName(note - NOTE_MIDDLEC);
	} else
	{
		return GetNoteName(note);
	}
}


std::string CSoundFile::GetNoteName(const ModCommand::NOTE note) const
{
	return GetNoteName(note, m_NoteNames);
}


std::string CSoundFile::GetNoteName(const ModCommand::NOTE note, const char (*noteNames)[4])
{
	if(ModCommand::IsSpecialNote(note))
	{
		const char specialNoteNames[][4] = { "PCs",  "PC ", "~~~", "^^^", "===" };
		STATIC_ASSERT(CountOf(specialNoteNames) == NOTE_MAX_SPECIAL - NOTE_MIN_SPECIAL + 1);
		return specialNoteNames[note - NOTE_MIN_SPECIAL];
	} else if(ModCommand::IsNote(note))
	{
		return std::string()
			.append(noteNames[(note - NOTE_MIN) % 12])
			.append(1, '0' + (note - NOTE_MIN) / 12)
			;	// e.g. "C#" + "5"
	} else if(note == NOTE_NONE)
	{
		return "...";
	}
	return "???";
}


#ifdef MODPLUG_TRACKER

void CSoundFile::SetDefaultNoteNames()
{
	m_NoteNames = TrackerSettings::Instance().accidentalFlats ? NoteNamesFlat : NoteNamesSharp;
}

const NoteName *CSoundFile::GetDefaultNoteNames()
{
	return m_NoteNames;
}

#endif // MODPLUG_TRACKER


void CSoundFile::SetModSpecsPointer(const CModSpecifications*& pModSpecs, const MODTYPE type)
{
	switch(type)
	{
		case MOD_TYPE_MPT:
			pModSpecs = &ModSpecs::mptm;
		break;

		case MOD_TYPE_IT:
			pModSpecs = &ModSpecs::itEx;
		break;

		case MOD_TYPE_XM:
			pModSpecs = &ModSpecs::xmEx;
		break;

		case MOD_TYPE_S3M:
			pModSpecs = &ModSpecs::s3mEx;
		break;

		case MOD_TYPE_MOD:
		default:
			pModSpecs = &ModSpecs::mod;
			break;
	}
}


void CSoundFile::SetType(MODTYPE type)
{
	m_nType = type;
	m_playBehaviour = GetDefaultPlaybackBehaviour(GetBestSaveFormat());
	SetModSpecsPointer(m_pModSpecs, GetBestSaveFormat());
}


#ifdef MODPLUG_TRACKER

void CSoundFile::ChangeModTypeTo(const MODTYPE& newType)
{
	const MODTYPE oldtype = GetType();
	m_nType = newType;
	SetModSpecsPointer(m_pModSpecs, m_nType);

	if(oldtype == newType)
		return;

	SetupMODPanning(); // Setup LRRL panning scheme if needed

	// Only keep supported play behaviour flags
	PlayBehaviourSet oldAllowedFlags = GetSupportedPlaybackBehaviour(oldtype);
	PlayBehaviourSet newAllowedFlags = GetSupportedPlaybackBehaviour(newType);
	PlayBehaviourSet newDefaultFlags = GetDefaultPlaybackBehaviour(newType);
	for(size_t i = 0; i < m_playBehaviour.size(); i++)
	{
		// If a flag is supported in both formats, keep its status
		if(m_playBehaviour[i]) m_playBehaviour.set(i, newAllowedFlags[i]);
		// Set allowed flags to their defaults if they were not supported in the old format
		if(!oldAllowedFlags[i]) m_playBehaviour.set(i, newDefaultFlags[i]);
	}

	Order.OnModTypeChanged(oldtype);
	Patterns.OnModTypeChanged(oldtype);
}

#endif // MODPLUG_TRACKER


bool CSoundFile::SetTitle(const std::string &newTitle)
{
	if(m_songName != newTitle)
	{
		m_songName = newTitle;
		return true;
	}
	return false;
}


double CSoundFile::GetPlaybackTimeAt(ORDERINDEX ord, ROWINDEX row, bool updateVars, bool updateSamplePos)
{
	const GetLengthType t = GetLength(updateVars ? (updateSamplePos ? eAdjustSamplePositions : eAdjust) : eNoAdjust, GetLengthTarget(ord, row)).back();
	if(t.targetReached) return t.duration;
	else return -1; //Given position not found from play sequence.
}


// Calculate the length of a tick, depending on the tempo mode.
// This differs from GetTickDuration() by not accumulating errors
// because this is not called once per tick but in unrelated
// circumstances. So this should not update error accumulation.
void CSoundFile::RecalculateSamplesPerTick()
{
	switch(m_nTempoMode)
	{
	case tempoModeClassic:
	default:
		m_PlayState.m_nSamplesPerTick = Util::muldiv(m_MixerSettings.gdwMixingFreq, 5 * TEMPO::fractFact, std::max(TEMPO::store_t(1), m_PlayState.m_nMusicTempo.GetRaw() << 1));
		break;

	case tempoModeModern:
		m_PlayState.m_nSamplesPerTick = static_cast<uint32>((Util::mul32to64_unsigned(m_MixerSettings.gdwMixingFreq, 60 * TEMPO::fractFact) / std::max(uint64(1),  Util::mul32to64_unsigned(m_PlayState.m_nMusicSpeed, m_PlayState.m_nCurrentRowsPerBeat) * m_PlayState.m_nMusicTempo.GetRaw())));
		break;

	case tempoModeAlternative:
		m_PlayState.m_nSamplesPerTick = Util::muldiv(m_MixerSettings.gdwMixingFreq, TEMPO::fractFact, std::max(TEMPO::store_t(1), m_PlayState.m_nMusicTempo.GetRaw()));
		break;
	}
#ifndef MODPLUG_TRACKER
	m_PlayState.m_nSamplesPerTick = Util::muldivr(m_PlayState.m_nSamplesPerTick, m_nTempoFactor, 65536);
#endif // !MODPLUG_TRACKER
	if(!m_PlayState.m_nSamplesPerTick)
		m_PlayState.m_nSamplesPerTick = 1;
}


// Get length of a tick in sample, with tick-to-tick tempo correction in modern tempo mode.
// This has to be called exactly once per tick because otherwise the error accumulation
// goes wrong.
uint32 CSoundFile::GetTickDuration(PlayState &playState) const
{
	uint32 retval = 0;
	switch(m_nTempoMode)
	{
	case tempoModeClassic:
	default:
		retval = Util::muldiv(m_MixerSettings.gdwMixingFreq, 5 * TEMPO::fractFact, std::max(TEMPO::store_t(1), playState.m_nMusicTempo.GetRaw() << 1));
		break;

	case tempoModeAlternative:
		retval = Util::muldiv(m_MixerSettings.gdwMixingFreq, TEMPO::fractFact, std::max(TEMPO::store_t(1), playState.m_nMusicTempo.GetRaw()));
		break;

	case tempoModeModern:
		{
			double accurateBufferCount = static_cast<double>(m_MixerSettings.gdwMixingFreq) * (60.0 / (playState.m_nMusicTempo.ToDouble() * Util::mul32to64_unsigned(playState.m_nMusicSpeed, playState.m_nCurrentRowsPerBeat)));
			const TempoSwing &swing = (Patterns.IsValidPat(playState.m_nPattern) && Patterns[playState.m_nPattern].HasTempoSwing())
				? Patterns[playState.m_nPattern].GetTempoSwing()
				: m_tempoSwing;
			if(!swing.empty())
			{
				// Apply current row's tempo swing factor
				TempoSwing::value_type swingFactor = swing[playState.m_nRow % swing.size()];
				accurateBufferCount = accurateBufferCount * swingFactor / double(TempoSwing::Unity);
			}
			uint32 bufferCount = static_cast<int>(accurateBufferCount);
			playState.m_dBufferDiff += accurateBufferCount - bufferCount;

			//tick-to-tick tempo correction:
			if(playState.m_dBufferDiff >= 1)
			{
				bufferCount++;
				playState.m_dBufferDiff--;
			} else if(m_PlayState.m_dBufferDiff <= -1)
			{
				bufferCount--;
				playState.m_dBufferDiff++;
			}
			MPT_ASSERT(mpt::abs(playState.m_dBufferDiff) < 1.0);
			retval = bufferCount;
		}
		break;
	}
#ifndef MODPLUG_TRACKER
	// when the user modifies the tempo, we do not really care about accurate tempo error accumulation
	retval = Util::muldivr_unsigned(retval, m_nTempoFactor, 65536);
#endif // !MODPLUG_TRACKER
	if(!retval)
		retval  = 1;
	return retval;
}


// Get the duration of a row in milliseconds, based on the current rows per beat and given speed and tempo settings.
double CSoundFile::GetRowDuration(TEMPO tempo, uint32 speed) const
{
	switch(m_nTempoMode)
	{
	case tempoModeClassic:
	default:
		return static_cast<double>(2500 * speed) / tempo.ToDouble();

	case tempoModeModern:
		{
			// If there are any row delay effects, the row length factor compensates for those.
			return 60000.0 / tempo.ToDouble() / static_cast<double>(m_PlayState.m_nCurrentRowsPerBeat);
		}

	case tempoModeAlternative:
		return static_cast<double>(1000 * speed) / tempo.ToDouble();
	}
}


const CModSpecifications& CSoundFile::GetModSpecifications(const MODTYPE type)
{
	const CModSpecifications* p = nullptr;
	SetModSpecsPointer(p, type);
	return *p;
}


// Find an unused sample slot. If it is going to be assigned to an instrument, targetInstrument should be specified.
// SAMPLEINDEX_INVLAID is returned if no free sample slot could be found.
SAMPLEINDEX CSoundFile::GetNextFreeSample(INSTRUMENTINDEX targetInstrument, SAMPLEINDEX start) const
{
	// Find empty slot in two passes - in the first pass, we only search for samples with empty sample names,
	// in the second pass we check all samples with non-empty sample names.
	for(int passes = 0; passes < 2; passes++)
	{
		for(SAMPLEINDEX i = start; i <= GetModSpecifications().samplesMax; i++)
		{
			// When loading into an instrument, ignore non-empty sample names. Else, only use this slot if the sample name is empty or we're in second pass.
			if((i > GetNumSamples() && passes == 1)
				|| (Samples[i].pSample == nullptr && (!m_szNames[i][0] || passes == 1 || targetInstrument != INSTRUMENTINDEX_INVALID))
				|| (targetInstrument != INSTRUMENTINDEX_INVALID && IsSampleReferencedByInstrument(i, targetInstrument)))	// Not empty, but already used by this instrument. XXX this should only be done when replacing an instrument with a single sample! Otherwise it will use an inconsistent sample map!
			{
				// Empty slot, so it's a good candidate already.

				// In instrument mode, check whether any instrument references this sample slot. If that is the case, we won't use it as it could lead to unwanted conflicts.
				// If we are loading the sample *into* an instrument, we should also not consider that instrument's sample map, since it might be inconsistent at this time.
				bool isReferenced = false;
				for(INSTRUMENTINDEX ins = 1; ins <= GetNumInstruments(); ins++)
				{
					if(ins == targetInstrument)
					{
						continue;
					}
					if(IsSampleReferencedByInstrument(i, ins))
					{
						isReferenced = true;
						break;
					}
				}
				if(!isReferenced)
				{
					return i;
				}
			}
		}
	}

	return SAMPLEINDEX_INVALID;
}


// Find an unused instrument slot.
// INSTRUMENTINDEX_INVALID is returned if no free instrument slot could be found.
INSTRUMENTINDEX CSoundFile::GetNextFreeInstrument(INSTRUMENTINDEX start) const
{
	for(INSTRUMENTINDEX i = start; i <= GetModSpecifications().instrumentsMax; i++)
	{
		if(Instruments[i] == nullptr)
		{
			return i;
		}
	}

	return INSTRUMENTINDEX_INVALID;
}


// Check whether a given sample is used by a given instrument.
bool CSoundFile::IsSampleReferencedByInstrument(SAMPLEINDEX sample, INSTRUMENTINDEX instr) const
{
	ModInstrument *targetIns = nullptr;
	if(instr > 0 && instr <= GetNumInstruments())
	{
		targetIns = Instruments[instr];
	}
	if(targetIns != nullptr)
	{
		for(size_t note = 0; note < NOTE_MAX /*CountOf(targetIns->Keyboard)*/; note++)
		{
			if(targetIns->Keyboard[note] == sample)
			{
				return true;
			}
		}
	}
	return false;
}


ModInstrument *CSoundFile::AllocateInstrument(INSTRUMENTINDEX instr, SAMPLEINDEX assignedSample)
{
	if(instr == 0 || instr >= MAX_INSTRUMENTS)
	{
		return nullptr;
	}

	ModInstrument *ins = Instruments[instr];
	if(ins != nullptr)
	{
		// Re-initialize instrument
		*ins = ModInstrument(assignedSample);
	} else
	{
		// Create new instrument
		Instruments[instr] = ins = new (std::nothrow) ModInstrument(assignedSample);
	}
	if(ins != nullptr)
	{
		m_nInstruments = std::max(m_nInstruments, instr);
	}
	return ins;
}


void CSoundFile::PrecomputeSampleLoops(bool updateChannels)
{
	for(SAMPLEINDEX i = 1; i <= GetNumSamples(); i++)
	{
		Samples[i].PrecomputeLoops(*this, updateChannels);
	}
}


#ifdef MPT_EXTERNAL_SAMPLES
// Load external waveform, but keep sample properties like frequency, panning, etc...
// Returns true if the file could be loaded.
bool CSoundFile::LoadExternalSample(SAMPLEINDEX smp, const mpt::PathString &filename)
{
	bool ok = false;
	InputFile f(filename);

	if(f.IsValid())
	{
		const ModSample origSample = Samples[smp];
		char origName[MAX_SAMPLENAME];
		mpt::String::Copy(origName, m_szNames[smp]);

		FileReader file = GetFileReader(f);
		ok = ReadSampleFromFile(smp, file, false);
		if(ok)
		{
			// Copy over old attributes, but keep new sample data
			ModSample &sample = GetSample(smp);
			SmpLength newLength = sample.nLength;
			void *newData = sample.pSample;
			SampleFlags newFlags = sample.uFlags;

			sample = origSample;
			sample.nLength = newLength;
			sample.pSample = newData;
			sample.uFlags.set(CHN_16BIT, newFlags[CHN_16BIT]);
			sample.uFlags.set(CHN_STEREO, newFlags[CHN_STEREO]);
			sample.uFlags.reset(SMP_MODIFIED);
			sample.SanitizeLoops();
		}
		mpt::String::Copy(m_szNames[smp], origName);
	}
	SetSamplePath(smp, filename);
	return ok;
}
#endif // MPT_EXTERNAL_SAMPLES


// Set up channel panning and volume suitable for MOD + similar files. If the current mod type is not MOD, bForceSetup has to be set to true.
void CSoundFile::SetupMODPanning(bool bForceSetup)
{
	// Setup LRRL panning, max channel volume
	if(!(GetType() & MOD_TYPE_MOD) && bForceSetup == false) return;

	for(CHANNELINDEX nChn = 0; nChn < MAX_BASECHANNELS; nChn++)
	{
		ChnSettings[nChn].nVolume = 64;
		ChnSettings[nChn].dwFlags.reset(CHN_SURROUND);
		if(m_MixerSettings.MixerFlags & SNDMIX_MAXDEFAULTPAN)
			ChnSettings[nChn].nPan = (((nChn & 3) == 1) || ((nChn & 3) == 2)) ? 256 : 0;
		else
			ChnSettings[nChn].nPan = (((nChn & 3) == 1) || ((nChn & 3) == 2)) ? 0xC0 : 0x40;
	}
}


void CSoundFile::PropagateXMAutoVibrato(INSTRUMENTINDEX ins, uint8 type, uint8 sweep, uint8 depth, uint8 rate)
{
	if(ins > m_nInstruments || Instruments[ins] == nullptr)
		return;
	const std::set<SAMPLEINDEX> referencedSamples = Instruments[ins]->GetSamples();

	// Propagate changes to all samples that belong to this instrument.
	for(auto sample : referencedSamples)
	{
		if(sample <= m_nSamples)
		{
			Samples[sample].nVibDepth = depth;
			Samples[sample].nVibType = type;
			Samples[sample].nVibRate = rate;
			Samples[sample].nVibSweep = sweep;
		}
	}
}


// Normalize the tempo swing coefficients so that they add up to exactly the specified tempo again
void TempoSwing::Normalize()
{
	if(empty()) return;
	uint64 sum = 0;
	for(auto &i : *this)
	{
		Limit(i, Unity / 4u, Unity * 4u);
		sum += i;
	}
	sum /= size();
	int64 remain = Unity * size();
	for(auto &i : *this)
	{
		i = Util::muldivr_unsigned(i, Unity, static_cast<int32>(sum));
		remain -= i;
	}
	//MPT_ASSERT(static_cast<uint32>(mpt::abs(static_cast<int32>(remain))) <= size());
	at(0) += static_cast<int32>(remain);
}


void TempoSwing::Serialize(std::ostream &oStrm, const TempoSwing &swing)
{
	mpt::IO::WriteIntLE<uint16>(oStrm, static_cast<uint16>(swing.size()));
	for(std::size_t i = 0; i < swing.size(); i++)
	{
		mpt::IO::WriteIntLE<uint32>(oStrm, swing[i]);
	}
}


void TempoSwing::Deserialize(std::istream &iStrm, TempoSwing &swing, const size_t)
{
	uint16 numEntries;
	mpt::IO::ReadIntLE<uint16>(iStrm, numEntries);
	swing.resize(numEntries);
	for(uint16 i = 0; i < numEntries; i++)
	{
		mpt::IO::ReadIntLE<uint32>(iStrm, swing[i]);
	}
	swing.Normalize();
}


OPENMPT_NAMESPACE_END
