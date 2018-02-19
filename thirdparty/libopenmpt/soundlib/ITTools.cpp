/*
 * ITTools.cpp
 * -----------
 * Purpose: Definition of IT file structures and helper functions
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "ITTools.h"
#include "Tables.h"
#include "../common/StringFixer.h"
#include "../common/version.h"


OPENMPT_NAMESPACE_BEGIN


// Convert OpenMPT's internal envelope format into an IT/MPTM envelope.
void ITEnvelope::ConvertToIT(const InstrumentEnvelope &mptEnv, uint8 envOffset, uint8 envDefault)
{
	// Envelope Flags
	if(mptEnv.dwFlags[ENV_ENABLED]) flags |= ITEnvelope::envEnabled;
	if(mptEnv.dwFlags[ENV_LOOP]) flags |= ITEnvelope::envLoop;
	if(mptEnv.dwFlags[ENV_SUSTAIN]) flags |= ITEnvelope::envSustain;
	if(mptEnv.dwFlags[ENV_CARRY]) flags |= ITEnvelope::envCarry;

	// Nodes and Loops
	num = (uint8)std::min(mptEnv.size(), uint32(25));
	lpb = (uint8)mptEnv.nLoopStart;
	lpe = (uint8)mptEnv.nLoopEnd;
	slb = (uint8)mptEnv.nSustainStart;
	sle = (uint8)mptEnv.nSustainEnd;

	// Envelope Data
	MemsetZero(data);
	if(!mptEnv.empty())
	{
		// Attention: Full MPTM envelope is stored in extended instrument properties
		for(uint32 ev = 0; ev < num; ev++)
		{
			data[ev].value = static_cast<int8>(mptEnv[ev].value) - envOffset;
			data[ev].tick = mptEnv[ev].tick;
		}
	} else
	{
		// Fix non-existing envelopes so that they can still be edited in Impulse Tracker.
		num = 2;
		data[0].value = data[1].value = envDefault - envOffset;
		data[1].tick = 10;
	}
}


// Convert IT/MPTM envelope data into OpenMPT's internal envelope format - To be used by ITInstrToMPT()
void ITEnvelope::ConvertToMPT(InstrumentEnvelope &mptEnv, uint8 envOffset, uint8 maxNodes) const
{
	// Envelope Flags
	mptEnv.dwFlags.set(ENV_ENABLED, (flags & ITEnvelope::envEnabled) != 0);
	mptEnv.dwFlags.set(ENV_LOOP, (flags & ITEnvelope::envLoop) != 0);
	mptEnv.dwFlags.set(ENV_SUSTAIN, (flags & ITEnvelope::envSustain) != 0);
	mptEnv.dwFlags.set(ENV_CARRY, (flags & ITEnvelope::envCarry) != 0);

	// Nodes and Loops
	mptEnv.resize(std::min(num, maxNodes));
	mptEnv.nLoopStart = std::min(lpb, maxNodes);
	mptEnv.nLoopEnd = Clamp(lpe, mptEnv.nLoopStart, maxNodes);
	mptEnv.nSustainStart = std::min(slb, maxNodes);
	mptEnv.nSustainEnd = Clamp(sle, mptEnv.nSustainStart, maxNodes);

	// Envelope Data
	// Attention: Full MPTM envelope is stored in extended instrument properties
	for(uint32 ev = 0; ev < std::min<uint32>(25, num); ev++)
	{
		mptEnv[ev].value = Clamp<int8, int8>(data[ev].value + envOffset, 0, 64);
		mptEnv[ev].tick = data[ev].tick;
		if(ev > 0 && ev < num && mptEnv[ev].tick < mptEnv[ev - 1].tick)
		{
			// Fix broken envelopes... Instruments 2 and 3 in NoGap.it by Werewolf have envelope points where the high byte of envelope nodes is missing.
			// NoGap.it was saved with MPT 1.07 or MPT 1.09, which *normally* doesn't do this in IT files.
			// However... It turns out that MPT 1.07 omitted the high byte of envelope nodes when saving an XI instrument file, and it looks like
			// Instrument 2 and 3 in NoGap.it were loaded from XI files.
			mptEnv[ev].tick &= 0xFF;
			mptEnv[ev].tick |= (mptEnv[ev].tick & ~0xFF);
			if(mptEnv[ev].tick < mptEnv[ev - 1].tick)
			{
				mptEnv[ev].tick += 0x100;
			}
		}
	}
}


// Convert an ITOldInstrument to OpenMPT's internal instrument representation.
void ITOldInstrument::ConvertToMPT(ModInstrument &mptIns) const
{
	// Header
	if(memcmp(id, "IMPI", 4))
	{
		return;
	}

	mpt::String::Read<mpt::String::spacePadded>(mptIns.name, name);
	mpt::String::Read<mpt::String::nullTerminated>(mptIns.filename, filename);

	// Volume / Panning
	mptIns.nFadeOut = fadeout << 6;
	mptIns.nGlobalVol = 64;
	mptIns.nPan = 128;

	// NNA Stuff
	mptIns.nNNA = nna;
	mptIns.nDCT = dnc;

	// Sample Map
	for(size_t i = 0; i < 120; i++)
	{
		uint8 note = keyboard[i * 2];
		SAMPLEINDEX ins = keyboard[i * 2 + 1];
		if(ins < MAX_SAMPLES)
		{
			mptIns.Keyboard[i] = ins;
		}
		if(note < 120)
		{
			mptIns.NoteMap[i] = note + 1u;
		} else
		{
			mptIns.NoteMap[i] = static_cast<uint8>(i + 1);
		}
	}

	// Volume Envelope Flags
	mptIns.VolEnv.dwFlags.set(ENV_ENABLED, (flags & ITOldInstrument::envEnabled) != 0);
	mptIns.VolEnv.dwFlags.set(ENV_LOOP, (flags & ITOldInstrument::envLoop) != 0);
	mptIns.VolEnv.dwFlags.set(ENV_SUSTAIN, (flags & ITOldInstrument::envSustain) != 0);

	// Volume Envelope Loops
	mptIns.VolEnv.nLoopStart = vls;
	mptIns.VolEnv.nLoopEnd = vle;
	mptIns.VolEnv.nSustainStart = sls;
	mptIns.VolEnv.nSustainEnd = sle;
	mptIns.VolEnv.resize(25);

	// Volume Envelope Data
	for(uint32 i = 0; i < 25; i++)
	{
		if((mptIns.VolEnv[i].tick = nodes[i * 2]) == 0xFF)
		{
			mptIns.VolEnv.resize(i);
			break;
		}
		mptIns.VolEnv[i].value = nodes[i * 2 + 1];
	}

	if(std::max(mptIns.VolEnv.nLoopStart, mptIns.VolEnv.nLoopEnd) >= mptIns.VolEnv.size()) mptIns.VolEnv.dwFlags.reset(ENV_LOOP);
	if(std::max(mptIns.VolEnv.nSustainStart, mptIns.VolEnv.nSustainEnd) >= mptIns.VolEnv.size()) mptIns.VolEnv.dwFlags.reset(ENV_SUSTAIN);
}


// Convert OpenMPT's internal instrument representation to an ITInstrument.
uint32 ITInstrument::ConvertToIT(const ModInstrument &mptIns, bool compatExport, const CSoundFile &sndFile)
{
	MemsetZero(*this);

	// Header
	memcpy(id, "IMPI", 4);
	trkvers = 0x5000 | static_cast<uint16>(MptVersion::num >> 16);

	mpt::String::Write<mpt::String::nullTerminated>(filename, mptIns.filename);
	mpt::String::Write<mpt::String::nullTerminated>(name, mptIns.name);

	// Volume / Panning
	fadeout = static_cast<uint16>(std::min<uint32>(mptIns.nFadeOut >> 5, 256u));
	gbv = static_cast<uint8>(std::min<uint32>(mptIns.nGlobalVol * 2u, 128u));
	dfp = static_cast<uint8>(std::min<uint32>(mptIns.nPan / 4u, 64u));
	if(!mptIns.dwFlags[INS_SETPANNING]) dfp |= ITInstrument::ignorePanning;

	// Random Variation
	rv = std::min(mptIns.nVolSwing, uint8(100));
	rp = std::min(mptIns.nPanSwing, uint8(64));

	// NNA Stuff
	nna = mptIns.nNNA;
	dct = (mptIns.nDCT < DCT_PLUGIN || !compatExport) ? mptIns.nDCT : DCT_NONE;
	dca = mptIns.nDNA;

	// Pitch / Pan Separation
	pps = mptIns.nPPS;
	ppc = mptIns.nPPC;

	// Filter Stuff
	ifc = mptIns.GetCutoff() | (mptIns.IsCutoffEnabled() ? ITInstrument::enableCutoff : 0x00);
	ifr = mptIns.GetResonance() | (mptIns.IsResonanceEnabled() ? ITInstrument::enableResonance : 0x00);

	// MIDI Setup
	if(mptIns.nMidiProgram > 0)
		mpr = mptIns.nMidiProgram - 1u;
	else
		mpr = 0xFF;
	if(mptIns.wMidiBank > 0)
	{
		mbank[0] = static_cast<uint8>((mptIns.wMidiBank - 1) & 0x7F);
		mbank[1] = static_cast<uint8>((mptIns.wMidiBank - 1) >> 7);
	} else
	{
		mbank[0] = 0xFF;
		mbank[1] = 0xFF;
	}
	if(mptIns.nMidiChannel != MidiNoChannel || mptIns.nMixPlug == 0 || mptIns.nMixPlug > 127 || compatExport)
	{
		// Default. Prefer MIDI channel over mixplug to keep the semantics intact.
		mch = mptIns.nMidiChannel;
	} else
	{
		// Keep compatibility with MPT 1.16's instrument format if possible, as XMPlay / BASS also uses this.
		mch = mptIns.nMixPlug + 128;
	}

	// Sample Map
	nos = 0;	// Only really relevant for ITI files
	std::vector<bool> smpCount(sndFile.GetNumSamples(), false);
	for(int i = 0; i < 120; i++)
	{
		keyboard[i * 2] = (mptIns.NoteMap[i] >= NOTE_MIN && mptIns.NoteMap[i] <= NOTE_MAX) ? (mptIns.NoteMap[i] - NOTE_MIN) : static_cast<uint8>(i);

		const SAMPLEINDEX smp = mptIns.Keyboard[i];
		if(smp < MAX_SAMPLES && smp < 256)
		{
			keyboard[i * 2 + 1] = static_cast<uint8>(smp);

			if(smp && smp <= sndFile.GetNumSamples() && !smpCount[smp - 1])
			{
				// We haven't considered this sample yet. Update number of samples.
				smpCount[smp - 1] = true;
				nos++;
			}
		}
	}

	// Writing Volume envelope
	volenv.ConvertToIT(mptIns.VolEnv, 0, 64);
	// Writing Panning envelope
	panenv.ConvertToIT(mptIns.PanEnv, 32, 32);
	// Writing Pitch Envelope
	pitchenv.ConvertToIT(mptIns.PitchEnv, 32, 32);
	if(mptIns.PitchEnv.dwFlags[ENV_FILTER]) pitchenv.flags |= ITEnvelope::envFilter;

	return sizeof(ITInstrument);
}


// Convert an ITInstrument to OpenMPT's internal instrument representation. Returns size of the instrument data that has been read.
uint32 ITInstrument::ConvertToMPT(ModInstrument &mptIns, MODTYPE modFormat) const
{
	if(memcmp(id, "IMPI", 4))
	{
		return 0;
	}

	mpt::String::Read<mpt::String::spacePadded>(mptIns.name, name);
	mpt::String::Read<mpt::String::nullTerminated>(mptIns.filename, filename);

	// Volume / Panning
	mptIns.nFadeOut = fadeout << 5;
	mptIns.nGlobalVol = gbv / 2;
	LimitMax(mptIns.nGlobalVol, 64u);
	mptIns.nPan = (dfp & 0x7F) * 4;
	if(mptIns.nPan > 256) mptIns.nPan = 128;
	mptIns.dwFlags.set(INS_SETPANNING, !(dfp & ITInstrument::ignorePanning));

	// Random Variation
	mptIns.nVolSwing = std::min<uint8>(rv, 100);
	mptIns.nPanSwing = std::min<uint8>(rp, 64);

	// NNA Stuff
	mptIns.nNNA = nna;
	mptIns.nDCT = dct;
	mptIns.nDNA = dca;

	// Pitch / Pan Separation
	mptIns.nPPS = pps;
	mptIns.nPPC = ppc;

	// Filter Stuff
	mptIns.SetCutoff(ifc & 0x7F, (ifc & ITInstrument::enableCutoff) != 0);
	mptIns.SetResonance(ifr & 0x7F, (ifr & ITInstrument::enableResonance) != 0);

	// MIDI Setup

	// MPT used to have a slightly different encoding of MIDI program and banks which we are trying to fix here.
	// Impulse Tracker / Schism Tracker will set trkvers to 0 in IT files,
	// and we won't care about correctly importing MIDI programs and banks in ITI files.
	// Chibi Tracker sets trkvers to 0x214, but always writes mpr=mbank=0 anyway.
	// Old BeRoTracker versions set trkvers to 0x214 or 0x217.
	//        <= MPT 1.07          <= MPT 1.16       OpenMPT 1.17-?      <= OpenMPT 1.26     definitely not MPT
	if((trkvers == 0x0202 || trkvers == 0x0211 || trkvers == 0x0220 || trkvers == 0x0214) && mpr != 0xFF)
	{
		if(mpr <= 128)
		{
			mptIns.nMidiProgram = mpr;
		}
		uint16 bank = mbank[0] | (mbank[1] << 8);
		// These versions also ignored the high bank nibble (was only handled correctly in OpenMPT instrument extensions)
		if(bank <= 128)
		{
			mptIns.wMidiBank = bank;
		}
	} else
	{
		if(mpr < 128)
		{
			mptIns.nMidiProgram = mpr + 1;
		}
		uint16 bank = 0;
		if(mbank[0] < 128)
			bank = mbank[0] + 1;
		if(mbank[1] < 128)
			bank += (mbank[1] << 7);
		mptIns.wMidiBank = bank;
	}
	mptIns.nMidiChannel = mch;
	if(mptIns.nMidiChannel >= 128)
	{
		// Handle old format where MIDI channel and Plugin index are stored in the same variable
		mptIns.nMixPlug = mptIns.nMidiChannel - 128;
		mptIns.nMidiChannel = 0;
	}

	// Envelope point count. Limited to 25 in IT format.
	const uint8 maxNodes = (modFormat & MOD_TYPE_MPT) ? MAX_ENVPOINTS : 25;

	// Volume Envelope
	volenv.ConvertToMPT(mptIns.VolEnv, 0, maxNodes);
	// Panning Envelope
	panenv.ConvertToMPT(mptIns.PanEnv, 32, maxNodes);
	// Pitch Envelope
	pitchenv.ConvertToMPT(mptIns.PitchEnv, 32, maxNodes);
	mptIns.PitchEnv.dwFlags.set(ENV_FILTER, (pitchenv.flags & ITEnvelope::envFilter) != 0);

	// Sample Map
	for(int i = 0; i < 120; i++)
	{
		uint8 note = keyboard[i * 2];
		SAMPLEINDEX ins = keyboard[i * 2 + 1];
		if(ins < MAX_SAMPLES)
		{
			mptIns.Keyboard[i] = ins;
		}
		if(note < 120)
		{
			mptIns.NoteMap[i] = note + NOTE_MIN;
		} else
		{
			mptIns.NoteMap[i] = static_cast<uint8>(i + NOTE_MIN);
		}
	}

	return sizeof(ITInstrument);
}


// Convert OpenMPT's internal instrument representation to an ITInstrumentEx. Returns amount of bytes that need to be written to file.
uint32 ITInstrumentEx::ConvertToIT(const ModInstrument &mptIns, bool compatExport, const CSoundFile &sndFile)
{
	uint32 instSize = iti.ConvertToIT(mptIns, compatExport, sndFile);

	if(compatExport)
	{
		return instSize;
	}

	// Sample Map
	bool usedExtension = false;
	iti.nos = 0;
	std::vector<bool> smpCount(sndFile.GetNumSamples(), false);
	for(int i = 0; i < 120; i++)
	{
		const SAMPLEINDEX smp = mptIns.Keyboard[i];
		keyboardhi[i] = 0;
		if(smp < MAX_SAMPLES)
		{
			if(smp >= 256)
			{
				// We need to save the upper byte for this sample index.
				iti.keyboard[i * 2 + 1] = static_cast<uint8>(smp & 0xFF);
				keyboardhi[i] = static_cast<uint8>(smp >> 8);
				usedExtension = true;
			}

			if(smp && smp <= sndFile.GetNumSamples() && !smpCount[smp - 1])
			{
				// We haven't considered this sample yet. Update number of samples.
				smpCount[smp - 1] = true;
				iti.nos++;
			}
		}
	}

	if(usedExtension)
	{
		// If we actually had to extend the sample map, update the magic bytes and instrument size.
		memcpy(iti.dummy, "XTPM", 4);
		instSize = sizeof(ITInstrumentEx);
	}

	return instSize;
}


// Convert an ITInstrumentEx to OpenMPT's internal instrument representation. Returns size of the instrument data that has been read.
uint32 ITInstrumentEx::ConvertToMPT(ModInstrument &mptIns, MODTYPE fromType) const
{
	uint32 insSize = iti.ConvertToMPT(mptIns, fromType);

	// Is this actually an extended instrument?
	// Note: OpenMPT 1.20 - 1.22 accidentally wrote "MPTX" here (since revision 1203), while previous versions wrote the reversed version, "XTPM".
	if(insSize == 0 || (memcmp(iti.dummy, "MPTX", 4) && memcmp(iti.dummy, "XTPM", 4)))
	{
		return insSize;
	}

	// Olivier's MPT Instrument Extension
	for(int i = 0; i < 120; i++)
	{
		mptIns.Keyboard[i] |= ((SAMPLEINDEX)keyboardhi[i] << 8);
	}

	return sizeof(ITInstrumentEx);
}


// Convert OpenMPT's internal sample representation to an ITSample.
void ITSample::ConvertToIT(const ModSample &mptSmp, MODTYPE fromType, bool compress, bool compressIT215, bool allowExternal)
{
	MemsetZero(*this);

	// Header
	memcpy(id, "IMPS", 4);

	mpt::String::Write<mpt::String::nullTerminated>(filename, mptSmp.filename);
	//mpt::String::Write<mpt::String::nullTerminated>(name, m_szNames[nsmp]);

	// Volume / Panning
	gvl = static_cast<uint8>(mptSmp.nGlobalVol);
	vol = static_cast<uint8>(mptSmp.nVolume / 4);
	dfp = static_cast<uint8>(mptSmp.nPan / 4);
	if(mptSmp.uFlags[CHN_PANNING]) dfp |= ITSample::enablePanning;

	// Sample Format / Loop Flags
	if(mptSmp.nLength && mptSmp.pSample)
	{
		flags = ITSample::sampleDataPresent;
		if(mptSmp.uFlags[CHN_LOOP]) flags |= ITSample::sampleLoop;
		if(mptSmp.uFlags[CHN_SUSTAINLOOP]) flags |= ITSample::sampleSustain;
		if(mptSmp.uFlags[CHN_PINGPONGLOOP]) flags |= ITSample::sampleBidiLoop;
		if(mptSmp.uFlags[CHN_PINGPONGSUSTAIN]) flags |= ITSample::sampleBidiSustain;

		if(mptSmp.uFlags[CHN_STEREO])
		{
			flags |= ITSample::sampleStereo;
		}
		if(mptSmp.uFlags[CHN_16BIT])
		{
			flags |= ITSample::sample16Bit;
		}
		cvt = ITSample::cvtSignedSample;

		if(compress)
		{
			flags |= ITSample::sampleCompressed;
			if(compressIT215)
			{
				cvt |= ITSample::cvtDelta;
			}
		}
	} else
	{
		flags = 0x00;
	}

	// Frequency
	C5Speed = mptSmp.nC5Speed ? mptSmp.nC5Speed : 8363;

	// Size and loops
	length = mpt::saturate_cast<uint32>(mptSmp.nLength);
	loopbegin = mpt::saturate_cast<uint32>(mptSmp.nLoopStart);
	loopend = mpt::saturate_cast<uint32>(mptSmp.nLoopEnd);
	susloopbegin = mpt::saturate_cast<uint32>(mptSmp.nSustainStart);
	susloopend = mpt::saturate_cast<uint32>(mptSmp.nSustainEnd);

	// Auto Vibrato settings
	vit = AutoVibratoXM2IT[mptSmp.nVibType & 7];
	vis = std::min(mptSmp.nVibRate, uint8(64));
	vid = std::min(mptSmp.nVibDepth, uint8(32));
	vir = std::min(mptSmp.nVibSweep, uint8(255));

	if((vid | vis) != 0 && (fromType & MOD_TYPE_XM))
	{
		// Sweep is upside down in XM
		vir = 255 - vir;
	}

	if(mptSmp.uFlags[SMP_KEEPONDISK])
	{
#ifndef MPT_EXTERNAL_SAMPLES
		MPT_UNREFERENCED_PARAMETER(allowExternal);
#else
		// Save external sample (filename at sample pointer)
		if(allowExternal && mptSmp.HasSampleData())
		{
			cvt = ITSample::cvtExternalSample;
		} else
#endif // MPT_EXTERNAL_SAMPLES
		{
			length = loopbegin = loopend = susloopbegin = susloopend = 0;
		}
	}
}


// Convert an ITSample to OpenMPT's internal sample representation.
uint32 ITSample::ConvertToMPT(ModSample &mptSmp) const
{
	if(memcmp(id, "IMPS", 4))
	{
		return 0;
	}

	mptSmp.Initialize(MOD_TYPE_IT);
	mpt::String::Read<mpt::String::nullTerminated>(mptSmp.filename, filename);

	// Volume / Panning
	mptSmp.nVolume = vol * 4;
	LimitMax(mptSmp.nVolume, uint16(256));
	mptSmp.nGlobalVol = gvl;
	LimitMax(mptSmp.nGlobalVol, uint16(64));
	mptSmp.nPan = (dfp & 0x7F) * 4;
	LimitMax(mptSmp.nPan, uint16(256));
	if(dfp & ITSample::enablePanning) mptSmp.uFlags.set(CHN_PANNING);

	// Loop Flags
	if(flags & ITSample::sampleLoop) mptSmp.uFlags.set(CHN_LOOP);
	if(flags & ITSample::sampleSustain) mptSmp.uFlags.set(CHN_SUSTAINLOOP);
	if(flags & ITSample::sampleBidiLoop) mptSmp.uFlags.set(CHN_PINGPONGLOOP);
	if(flags & ITSample::sampleBidiSustain) mptSmp.uFlags.set(CHN_PINGPONGSUSTAIN);

	// Frequency
	mptSmp.nC5Speed = C5Speed;
	if(!mptSmp.nC5Speed) mptSmp.nC5Speed = 8363;
	if(mptSmp.nC5Speed < 256) mptSmp.nC5Speed = 256;

	// Size and loops
	mptSmp.nLength = length;
	mptSmp.nLoopStart = loopbegin;
	mptSmp.nLoopEnd = loopend;
	mptSmp.nSustainStart = susloopbegin;
	mptSmp.nSustainEnd = susloopend;
	mptSmp.SanitizeLoops();

	// Auto Vibrato settings
	mptSmp.nVibType = AutoVibratoIT2XM[vit & 7];
	mptSmp.nVibRate = vis;
	mptSmp.nVibDepth = vid & 0x7F;
	mptSmp.nVibSweep = vir;

	if(cvt == ITSample::cvtExternalSample)
	{
		// Read external sample (filename at sample pointer)
		mptSmp.uFlags.set(SMP_KEEPONDISK);
	}

	return samplepointer;
}


// Retrieve the internal sample format flags for this instrument.
SampleIO ITSample::GetSampleFormat(uint16 cwtv) const
{
	SampleIO sampleIO(
		(flags & ITSample::sample16Bit) ? SampleIO::_16bit : SampleIO::_8bit,
		SampleIO::mono,
		SampleIO::littleEndian,
		(cvt & ITSample::cvtSignedSample) ? SampleIO::signedPCM: SampleIO::unsignedPCM);

	// Some old version of IT didn't clear the stereo flag when importing samples. Luckily, all other trackers are identifying as IT 2.14+, so let's check for old IT versions.
	if((flags & ITSample::sampleStereo) && cwtv >= 0x214)
	{
		sampleIO |= SampleIO::stereoSplit;
	}

	if(flags & ITSample::sampleCompressed)
	{
		// IT 2.14 packed sample
		sampleIO |= (cvt & ITSample::cvtDelta) ? SampleIO::IT215 : SampleIO::IT214;
	} else
	{
		// MODPlugin :(
		if(!(flags & ITSample::sample16Bit) && cvt == ITSample::cvtADPCMSample)
		{
			sampleIO |= SampleIO::ADPCM;
		} else
		{
			// ITTECH.TXT says these convert flags are "safe to ignore". IT doesn't ignore them, though, so why should we? :)
			if(cvt & ITSample::cvtBigEndian)
			{
				sampleIO |= SampleIO::bigEndian;
			}
			if(cvt & ITSample::cvtDelta)
			{
				sampleIO |= SampleIO::deltaPCM;
			}
			if((cvt & ITSample::cvtPTM8to16) && (flags & ITSample::sample16Bit))
			{
				sampleIO |= SampleIO::PTM8Dto16;
			}
		}
	}

	return sampleIO;
}


// Convert an ITHistoryStruct to OpenMPT's internal edit history representation
void ITHistoryStruct::ConvertToMPT(FileHistory &mptHistory) const
{
	// Decode FAT date and time
	MemsetZero(mptHistory.loadDate);
	mptHistory.loadDate.tm_year = ((fatdate >> 9) & 0x7F) + 80;
	mptHistory.loadDate.tm_mon = Clamp((fatdate >> 5) & 0x0F, 1, 12) - 1;
	mptHistory.loadDate.tm_mday = Clamp(fatdate & 0x1F, 1, 31);
	mptHistory.loadDate.tm_hour = Clamp((fattime >> 11) & 0x1F, 0, 23);
	mptHistory.loadDate.tm_min = Clamp((fattime >> 5) & 0x3F, 0, 59);
	mptHistory.loadDate.tm_sec = Clamp((fattime & 0x1F) * 2, 0, 59);
	mptHistory.openTime = static_cast<uint32>(runtime * (HISTORY_TIMER_PRECISION / 18.2f));
}


// Convert OpenMPT's internal edit history representation to an ITHistoryStruct
void ITHistoryStruct::ConvertToIT(const FileHistory &mptHistory)
{
	// Create FAT file dates
	fatdate = static_cast<uint16>(mptHistory.loadDate.tm_mday | ((mptHistory.loadDate.tm_mon + 1) << 5) | ((mptHistory.loadDate.tm_year - 80) << 9));
	fattime = static_cast<uint16>((mptHistory.loadDate.tm_sec / 2) | (mptHistory.loadDate.tm_min << 5) | (mptHistory.loadDate.tm_hour << 11));
	runtime = static_cast<uint32>(mptHistory.openTime * (18.2f / HISTORY_TIMER_PRECISION));
}


OPENMPT_NAMESPACE_END
