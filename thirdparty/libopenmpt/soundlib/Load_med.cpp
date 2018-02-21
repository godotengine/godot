/*
 * Load_med.cpp
 * ------------
 * Purpose: OctaMed MED module loader
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Loaders.h"
#include "../common/StringFixer.h"

OPENMPT_NAMESPACE_BEGIN

//#define MED_LOG

#define MED_MAX_COMMENT_LENGTH 5*1024 //: Is 5 kB enough?

// flags
#define	MMD_FLAG_FILTERON	0x1
#define	MMD_FLAG_JUMPINGON	0x2
#define	MMD_FLAG_JUMP8TH	0x4
#define	MMD_FLAG_INSTRSATT	0x8 // instruments are attached (this is a module)
#define	MMD_FLAG_VOLHEX		0x10
#define MMD_FLAG_STSLIDE	0x20 // SoundTracker mode for slides
#define MMD_FLAG_8CHANNEL	0x40 // OctaMED 8 channel song
#define	MMD_FLAG_SLOWHQ		0x80 // HQ slows playing speed (V2-V4 compatibility)
// flags2
#define MMD_FLAG2_BMASK		0x1F
#define MMD_FLAG2_BPM		0x20
#define	MMD_FLAG2_MIX		0x80 // uses Mixing (V7+)
// flags3:
#define	MMD_FLAG3_STEREO	0x1	// mixing in Stereo mode
#define	MMD_FLAG3_FREEPAN	0x2	// free panning
#define MMD_FLAG3_GM		0x4 // module designed for GM/XG compatibility


// generic MMD tags
#define	MMDTAG_END		0
#define	MMDTAG_PTR		0x80000000	// data needs relocation
#define	MMDTAG_MUSTKNOW	0x40000000	// loader must fail if this isn't recognized
#define	MMDTAG_MUSTWARN	0x20000000	// loader must warn if this isn't recognized

// ExpData tags
// # of effect groups, including the global group (will
// override settings in MMDSong struct), default = 1
#define	MMDTAG_EXP_NUMFXGROUPS	1
#define	MMDTAG_TRK_NAME		(MMDTAG_PTR|1)	// trackinfo tags
#define	MMDTAG_TRK_NAMELEN	2				// namelen includes zero term.
#define	MMDTAG_TRK_FXGROUP	3
// effectinfo tags
#define	MMDTAG_FX_ECHOTYPE	1
#define MMDTAG_FX_ECHOLEN	2
#define	MMDTAG_FX_ECHODEPTH	3
#define	MMDTAG_FX_STEREOSEP	4
#define	MMDTAG_FX_GROUPNAME	(MMDTAG_PTR|5)	// the Global Effects group shouldn't have name saved!
#define	MMDTAG_FX_GRPNAMELEN 6	// namelen includes zero term.


struct MEDMODULEHEADER
{
	char     id[4];		// MMD1-MMD3
	uint32be modlen;	// Size of file
	uint32be song;		// Position in file for this song
	uint16be psecnum;
	uint16be pseq;
	uint32be blockarr;	// Position in file for blocks
	uint32be mmdflags;
	uint32be smplarr;	// Position in file for samples
	uint32be reserved;
	uint32be expdata;	// Absolute offset in file for ExpData (0 if not present)
	uint32be reserved2;
	uint16be pstate;
	uint16be pblock;
	uint16be pline;
	uint16be pseqnum;
	uint16be actplayline;
	uint8be  counter;
	uint8be  extra_songs;	// # of songs - 1
};

MPT_BINARY_STRUCT(MEDMODULEHEADER, 52)


struct MMD0SAMPLE
{
	uint16be rep, replen;
	uint8be  midich;
	uint8be  midipreset;
	uint8be  svol;
	int8be   strans;
};

MPT_BINARY_STRUCT(MMD0SAMPLE, 8)


// Sample header is immediately followed by sample data...
struct MMDSAMPLEHEADER
{
	uint32be length;     // length of *one* *unpacked* channel in *bytes*
	uint16be type;
				// if non-negative
					// bits 0-3 reserved for multi-octave instruments, not supported on the PC
					// 0x10: 16 bit (otherwise 8 bit)
					// 0x20: Stereo (otherwise mono)
					// 0x40: Uses DeltaCode
					// 0x80: Packed data
				// -1: Synth
				// -2: Hybrid
	// if type indicates packed data, these fields follow, otherwise we go right to the data
	uint16be packtype;	// Only 1 = ADPCM is supported
	uint16be subtype;	// Packing subtype
		// ADPCM subtype
		// 1: g723_40
		// 2: g721
		// 3: g723_24
	uint8be  commonflags;	// flags common to all packtypes (none defined so far)
	uint8be  packerflags;	// flags for the specific packtype
	uint32be leftchlen;	// packed length of left channel in bytes
	uint32be rightchlen;	// packed length of right channel in bytes (ONLY PRESENT IN STEREO SAMPLES)
	uint8be  SampleData[1];	// Sample Data
};

MPT_BINARY_STRUCT(MMDSAMPLEHEADER, 21)


// MMD0/MMD1 song header
struct MMD0SONGHEADER
{
	MMD0SAMPLE sample[63];
	uint16be numblocks;		// # of blocks
	uint16be songlen;		// # of entries used in playseq
	uint8be  playseq[256];	// Play sequence
	uint16be deftempo;		// BPM tempo
	int8be   playtransp;	// Play transpose
	uint8be  flags;			// 0x10: Hex Volumes | 0x20: ST/NT/PT Slides | 0x40: 8 Channels song
	uint8be  flags2;		// [b4-b0]+1: Tempo LPB, 0x20: tempo mode, 0x80: mix_conv=on
	uint8be  tempo2;		// tempo TPL
	uint8be  trkvol[16];	// track volumes
	uint8be  mastervol;		// master volume
	uint8be  numsamples;	// # of samples (max=63)
};

MPT_BINARY_STRUCT(MMD0SONGHEADER, 788)


// MMD2/MMD3 song header
struct MMD2SONGHEADER
{
	MMD0SAMPLE sample[63];
	uint16be numblocks;		// # of blocks
	uint16be numsections;	// # of sections
	uint32be playseqtable;	// filepos of play sequence
	uint32be sectiontable;	// filepos of sections table (uint16_be array)
	uint32be trackvols;		// filepos of tracks volume (uint8_be array)
	uint16be numtracks;		// # of tracks (max 64)
	uint16be numpseqs;		// # of play sequences
	uint32be trackpans;		// filepos of tracks pan values (uint8_be array)
	int32be  flags3;		// 0x1:stereo_mix, 0x2:free_panning, 0x4:GM/XG compatibility
	uint16be voladj;		// vol_adjust (set to 100 if 0)
	uint16be channels;		// # of channels (4 if =0)
	uint8be  mix_echotype;	// 1:normal,2:xecho
	uint8be  mix_echodepth;	// 1..6
	uint16be mix_echolen;	// > 0
	int8be   mix_stereosep;	// -4..4
	uint8be  pad0[223];
	uint16be deftempo;		// BPM tempo
	int8be   playtransp;	// play transpose
	uint8be  flags;			// 0x1:filteron, 0x2:jumpingon, 0x4:jump8th, 0x8:instr_attached, 0x10:hex_vol, 0x20:PT_slides, 0x40:8ch_conv,0x80:hq slows playing speed
	uint8be  flags2;		// 0x80:mix_conv=on, [b4-b0]+1:tempo LPB, 0x20:tempo_mode
	uint8be  tempo2;		// tempo TPL
	uint8be  pad1[16];
	uint8be  mastervol;		// master volume
	uint8be  numsamples;	// # of samples (max 63)
};

MPT_BINARY_STRUCT(MMD2SONGHEADER, 788)


// For MMD0 the note information is held in 3 bytes, byte0, byte1, byte2.  For reference we
// number the bits in each byte 0..7, where 0 is the low bit.
// The note is held as bits 5..0 of byte0
// The instrument is encoded in 6 bits,  bits 7 and 6 of byte0 and bits 7,6,5,4 of byte1
// The command number is bits 3,2,1,0 of byte1, command data is in byte2:
// For command 0, byte2 represents the second data byte, otherwise byte2
// represents the first data byte.
struct MMD0BLOCK
{
	uint8be numtracks;
	uint8be lines;		// File value is 1 less than actual, so 0 -> 1 line
};			// uint8_be data[lines+1][tracks][3];

MPT_BINARY_STRUCT(MMD0BLOCK, 2)


// For MMD1,MMD2,MMD3 the note information is carried in 4 bytes, byte0, byte1,
// byte2 and byte3
// The note is held as byte0 (values above 0x84 are ignored)
// The instrument is held as byte1
// The command number is held as byte2, command data is in byte3
// For commands 0 and 0x19 byte3 represents the second data byte,
// otherwise byte2 represents the first data byte.
struct MMD1BLOCK
{
	uint16be numtracks;	// Number of tracks, may be > 64, but then that data is skipped.
	uint16be lines;		// Stored value is 1 less than actual, so 0 -> 1 line
	uint32be info;			// Offset of BlockInfo (if 0, no block_info is present)
};

MPT_BINARY_STRUCT(MMD1BLOCK, 8)


struct MMD1BLOCKINFO
{
	uint32be hlmask;		// Unimplemented - ignore
	uint32be blockname;		// file offset of block name
	uint32be blocknamelen;	// length of block name (including term. 0)
	uint32be pagetable;		// file offset of command page table
	uint32be cmdexttable;	// file offset of command extension table
	uint32be reserved[4];	// future expansion
};

MPT_BINARY_STRUCT(MMD1BLOCKINFO, 36)


// A set of play sequences is stored as an array of uint32_be files offsets
// Each offset points to the play sequence itself.
struct MMD2PLAYSEQ
{
	char     name[32];
	uint32be command_offs;	// filepos of command table
	uint32be reserved;
	uint16be length;
	uint16be seq[512];		// skip if > 0x8000
};

MPT_BINARY_STRUCT(MMD2PLAYSEQ, 1066)


// A command table contains commands that effect a particular play sequence
// entry.  The only commands read in are STOP or POSJUMP, all others are ignored
// POSJUMP is presumed to have extra bytes containing a uint16 for the position
struct MMDCOMMAND
{
	uint16be offset;			// Offset within current sequence entry
	uint8be cmdnumber;			// STOP (537) or POSJUMP (538) (others skipped)
	uint8be extra_count;
	uint8be extra_bytes[4];	// [extra_count];
};  // Last entry has offset == 0xFFFF, cmd_number == 0 and 0 extrabytes

MPT_BINARY_STRUCT(MMDCOMMAND, 8)


struct MMD0EXP
{
	uint32be nextmod;			// File offset of next Hdr
	uint32be exp_smp;			// Pointer to extra instrument data
	uint16be s_ext_entries;		// Number of extra instrument entries
	uint16be s_ext_entrsz;		// Size of extra instrument data
	uint32be annotxt;
	uint32be annolen;
	uint32be iinfo;				// Instrument names
	uint16be i_ext_entries;
	uint16be i_ext_entrsz;
	uint32be jumpmask;
	uint32be rgbtable;
	uint8be  channelsplit[4];	// Only used if 8ch_conv (extra channel for every nonzero entry)
	uint32be n_info;
	uint32be songname;			// Song name
	uint32be songnamelen;
	uint32be dumps;
	uint32be mmdinfo;
	uint32be mmdrexx;
	uint32be mmdcmd3x;
	uint32be trackinfo_ofs;		// ptr to song->numtracks ptrs to tag lists
	uint32be effectinfo_ofs;	// ptr to group ptrs
	uint32be tag_end;
};

MPT_BINARY_STRUCT(MMD0EXP, 80)


static const uint8 bpmvals[9] = { 179,164,152,141,131,123,116,110,104};

static void MedConvert(ModCommand &p, const MMD0SONGHEADER *pmsh)
{
	ModCommand::COMMAND command = p.command;
	uint32 param = p.param;
	switch(command)
	{
	case 0x00:	if (param) command = CMD_ARPEGGIO; else command = CMD_NONE; break;
	case 0x01:	command = CMD_PORTAMENTOUP; break;
	case 0x02:	command = CMD_PORTAMENTODOWN; break;
	case 0x03:	command = CMD_TONEPORTAMENTO; break;
	case 0x04:	command = CMD_VIBRATO; break;
	case 0x05:	command = CMD_TONEPORTAVOL; break;
	case 0x06:	command = CMD_VIBRATOVOL; break;
	case 0x07:	command = CMD_TREMOLO; break;
	case 0x0A:	if (param & 0xF0) param &= 0xF0; command = CMD_VOLUMESLIDE; if (!param) command = CMD_NONE; break;
	case 0x0B:	command = CMD_POSITIONJUMP; break;
	case 0x0C:	command = CMD_VOLUME;
				if (pmsh->flags & MMD_FLAG_VOLHEX)
				{
					if (param < 0x80)
					{
						param = (param+1) / 2;
					} else command = CMD_NONE;
				} else
				{
					if (param <= 0x99)
					{
						param = (param >> 4)*10+((param & 0x0F) % 10);
						if (param > 64) param = 64;
					} else command = CMD_NONE;
				}
				break;
	case 0x09:	command = static_cast<ModCommand::COMMAND>((param <= 0x20) ? CMD_SPEED : CMD_TEMPO); break;
	case 0x0D:	if (param & 0xF0) param &= 0xF0; command = CMD_VOLUMESLIDE; if (!param) command = CMD_NONE; break;
	case 0x0F:	// Set Tempo / Special
		// F.00 = Pattern Break
		if (!param)	command = CMD_PATTERNBREAK;	else
		// F.01 - F.F0: Set tempo/speed
		if (param <= 0xF0)
		{
			if (pmsh->flags & MMD_FLAG_8CHANNEL)
			{
				param = (param == 0 || param >= 10) ? 99 : bpmvals[param-1];
			} else
			// F.01 - F.0A: Set Speed
			if (param <= 0x0A)
			{
				command = CMD_SPEED;
			} else
			// Old tempo
			if (!(pmsh->flags2 & MMD_FLAG2_BPM))
			{
				param = Util::muldiv(param, 5*715909, 2*474326);
			}
			// F.0B - F.F0: Set Tempo (assumes LPB=4)
			if (param > 0x0A)
			{
				command = CMD_TEMPO;
				if (param < 0x21) param = 0x21;
				if (param > 240) param = 240;
			}
		} else
		switch(param)
		{
		// F.F1: Retrig 2x
		case 0xF1:
			command = CMD_MODCMDEX;
			param = 0x93;
			break;
		// F.F2: Note Delay 2x
		case 0xF2:
			command = CMD_MODCMDEX;
			param = 0xD3;
			break;
		// F.F3: Retrig 3x
		case 0xF3:
			command = CMD_MODCMDEX;
			param = 0x92;
			break;
		// F.F4: Note Delay 1/3
		case 0xF4:
			command = CMD_MODCMDEX;
			param = 0xD2;
			break;
		// F.F5: Note Delay 2/3
		case 0xF5:
			command = CMD_MODCMDEX;
			param = 0xD4;
			break;
		// F.F8: Filter Off
		case 0xF8:
			command = CMD_MODCMDEX;
			param = 0x00;
			break;
		// F.F9: Filter On
		case 0xF9:
			command = CMD_MODCMDEX;
			param = 0x01;
			break;
		// F.FD: Very fast tone-portamento
		case 0xFD:
			command = CMD_TONEPORTAMENTO;
			param = 0xFF;
			break;
		// F.FE: End Song
		case 0xFE:
			command = CMD_SPEED;
			param = 0;
			break;
		// F.FF: Note Cut
		case 0xFF:
			command = CMD_MODCMDEX;
			param = 0xC0;
			break;
		default:
#ifdef MED_LOG
			Log("Unknown Fxx command: cmd=0x%02X param=0x%02X\n", command, param);
#endif
			command = CMD_NONE;
			param = 0;
		}
		break;
	// 11.0x: Fine Slide Up
	case 0x11:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0x10;
		break;
	// 12.0x: Fine Slide Down
	case 0x12:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0x20;
		break;
	// 14.xx: Vibrato
	case 0x14:
		command = CMD_VIBRATO;
		break;
	// 15.xx: FineTune
	case 0x15:
		command = CMD_MODCMDEX;
		param &= 0x0F;
		param |= 0x50;
		break;
	// 16.xx: Pattern Loop
	case 0x16:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0x60;
		break;
	// 18.xx: Note Cut
	case 0x18:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0xC0;
		break;
	// 19.xx: Sample Offset
	case 0x19:
		command = CMD_OFFSET;
		break;
	// 1A.0x: Fine Volume Up
	case 0x1A:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0xA0;
		break;
	// 1B.0x: Fine Volume Down
	case 0x1B:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0xB0;
		break;
	// 1D.xx: Pattern Break
	case 0x1D:
		command = CMD_PATTERNBREAK;
		break;
	// 1E.0x: Pattern Delay
	case 0x1E:
		command = CMD_MODCMDEX;
		if (param > 0x0F) param = 0x0F;
		param |= 0xE0;
		break;
	// 1F.xy: Retrig
	case 0x1F:
		command = CMD_RETRIG;
		param &= 0x0F;
		break;
	// 2E.xx: set panning
	case 0x2E:
		command = CMD_MODCMDEX;
		param = ((param + 0x10) & 0xFF) >> 1;
		if (param > 0x0F) param = 0x0F;
		param |= 0x80;
		break;
	default:
#ifdef MED_LOG
		// 0x2E ?
		Log("Unknown command: cmd=0x%02X param=0x%02X\n", command, param);
#endif
		command = CMD_NONE;
		param = 0;
	}
	p.command = command;
	p.param = static_cast<ModCommand::PARAM>(param);
}


static bool ValidateHeader(const MEDMODULEHEADER &pmmh)
{
	if(std::memcmp(pmmh.id, "MMD", 3)
		|| pmmh.id[3] < '0' || pmmh.id[3] > '3'
		|| pmmh.song == 0
		)
	{
		return false;
	}
	return true;
}


static uint64 GetHeaderMinimumAdditionalSize(const MEDMODULEHEADER &pmmh)
{
	MPT_UNREFERENCED_PARAMETER(pmmh);
	return sizeof(MMD0SONGHEADER);
}


CSoundFile::ProbeResult CSoundFile::ProbeFileHeaderMED(MemoryFileReader file, const uint64 *pfilesize)
{
	MEDMODULEHEADER pmmh;
	if(!file.ReadStruct(pmmh))
	{
		return ProbeWantMoreData;
	}
	if(!ValidateHeader(pmmh))
	{
		return ProbeFailure;
	}
	return ProbeAdditionalSize(file, pfilesize, GetHeaderMinimumAdditionalSize(pmmh));
}


bool CSoundFile::ReadMed(FileReader &file, ModLoadingFlags loadFlags)
{
	file.Rewind();
	MEDMODULEHEADER pmmh;
	if(!file.ReadStruct(pmmh))
	{
		return false;
	}
	if(!ValidateHeader(pmmh))
	{
		return false;
	}
	if(!file.CanRead(mpt::saturate_cast<FileReader::off_t>(GetHeaderMinimumAdditionalSize(pmmh))))
	{
		return false;
	}
	const uint32 dwSong = pmmh.song;
	if(!file.LengthIsAtLeast(dwSong + sizeof(MMD0SONGHEADER)))
	{
		return false;
	}
	if(loadFlags == onlyVerifyHeader)
	{
		return true;
	}

	file.Rewind();
	const FileReader::off_t dwMemLength = file.GetLength();
	const uint8 *lpStream = file.GetRawData<uint8>();
	const MMD0SONGHEADER *pmsh;
	const MMD2SONGHEADER *pmsh2;
	const MMD0EXP *pmex;
	uint32 dwBlockArr, dwSmplArr, dwExpData;
	const_unaligned_ptr_be<uint32> pdwTable;
	int8 version = pmmh.id[3];
	uint32 deftempo;
	int playtransp = 0;

	InitializeGlobals(MOD_TYPE_MED);
	InitializeChannels();
	// Setup channel pan positions and volume
	SetupMODPanning(true);
	m_madeWithTracker = mpt::format(MPT_USTRING("OctaMED (MMD%1)"))(mpt::ToUnicode(mpt::CharsetISO8859_1, std::string(1, version)));

	m_nSamplePreAmp = 32;
	dwBlockArr = pmmh.blockarr;
	dwSmplArr = pmmh.smplarr;
	dwExpData = pmmh.expdata;
	if ((dwExpData) && (dwExpData < dwMemLength - sizeof(MMD0EXP)))
		pmex = (const MMD0EXP *)(lpStream+dwExpData);
	else
		pmex = NULL;
	pmsh = (const MMD0SONGHEADER *)(lpStream + dwSong);
	pmsh2 = (const MMD2SONGHEADER *)pmsh;

	uint16 wNumBlocks = pmsh->numblocks;
	m_nChannels = 4;
	m_nSamples = pmsh->numsamples;
	if (m_nSamples > 63) m_nSamples = 63;
	// Tempo
	m_nDefaultTempo.Set(125);
	deftempo = pmsh->deftempo;
	if (!deftempo) deftempo = 125;
	if (pmsh->flags2 & MMD_FLAG2_BPM)
	{
		uint32 tempo_tpl = (pmsh->flags2 & MMD_FLAG2_BMASK) + 1;
		if (!tempo_tpl) tempo_tpl = 4;
		deftempo *= tempo_tpl;
		deftempo /= 4;
	#ifdef MED_LOG
		Log("newtempo: %3d bpm (bpm=%3d lpb=%2d)\n", deftempo, pmsh->deftempo, (pmsh->flags2 & MMD_FLAG2_BMASK)+1);
	#endif
	} else
	{
		if((pmsh->flags & MMD_FLAG_8CHANNEL) && deftempo > 0 && deftempo <= 9)
			deftempo = bpmvals[deftempo-1];
		else
			deftempo = Util::muldiv(deftempo, 5 * 715909, 2 * 474326);
	#ifdef MED_LOG
		Log("oldtempo: %3d bpm (bpm=%3d)\n", deftempo, pmsh->deftempo);
	#endif
	}
	// Speed
	m_nDefaultSpeed = pmsh->tempo2;
	if (!m_nDefaultSpeed) m_nDefaultSpeed = 6;
	if (deftempo < 0x21) deftempo = 0x21;
	m_nDefaultTempo.Set(deftempo);
	// Reading Samples
	for (uint32 iSHdr=0; iSHdr<m_nSamples; iSHdr++)
	{
		ModSample &sample = Samples[iSHdr + 1];
		sample.nLoopStart = pmsh->sample[iSHdr].rep * 2u;
		sample.nLoopEnd = sample.nLoopStart + (pmsh->sample[iSHdr].replen * 2u);
		sample.nVolume = (pmsh->sample[iSHdr].svol << 2);
		sample.nGlobalVol = 64;
		if (sample.nVolume > 256) sample.nVolume = 256;
		// Was: sample.RelativeTone = -12 * pmsh->sample[iSHdr].strans;
		// But that breaks MMD1 modules (e.g. "94' summer.mmd1" from Modland) - "automatic terminated to.mmd0" still sounds broken, probably "play transpose" is broken there.
		sample.RelativeTone = pmsh->sample[iSHdr].strans;
		sample.nPan = 128;
		if (sample.nLoopEnd <= 2) sample.nLoopEnd = 0;
		if (sample.nLoopEnd) sample.uFlags.set(CHN_LOOP);
	}
	// Common Flags
	m_SongFlags.set(SONG_FASTVOLSLIDES, !(pmsh->flags & 0x20));

	// Reading play sequence
	if (version < '2')
	{
		uint32 nbo = pmsh->songlen;
		if (!nbo) nbo = 1;
		ReadOrderFromArray(Order(), pmsh->playseq, nbo);
		playtransp = pmsh->playtransp;
	} else
	{
		uint32 nSections;
		ORDERINDEX nOrders = 0;
		uint16 nTrks = pmsh2->numtracks;
		if ((nTrks >= 4) && (nTrks <= 32)) m_nChannels = nTrks;
		uint32 playseqtable = pmsh2->playseqtable;
		uint32 numplayseqs = pmsh2->numpseqs;
		if (!numplayseqs) numplayseqs = 1;
		nSections = pmsh2->numsections;
		uint32 sectiontable = pmsh2->sectiontable;
		if ((!nSections) || (!sectiontable) || (sectiontable >= dwMemLength-2)) nSections = 1;
		nOrders = 0;
		Order().clear();
		for (uint32 iSection=0; iSection<nSections; iSection++)
		{
			uint32 nplayseq = 0;
			if (sectiontable && sectiontable < dwMemLength && 2 >= dwMemLength - sectiontable)
			{
				nplayseq = *const_unaligned_ptr_be<uint16>(lpStream + sectiontable);
				sectiontable += 2;
			} else
			{
				nSections = 0;
			}
			uint32 pseq = 0;

			if ((playseqtable) && (playseqtable < dwMemLength) && (nplayseq * 4 <= dwMemLength - playseqtable))
			{
				pseq = (const_unaligned_ptr_be<uint32>(lpStream+playseqtable))[nplayseq];
			}
			if (pseq && pseq < dwMemLength && sizeof(MMD2PLAYSEQ) <= dwMemLength - pseq)
			{
				const MMD2PLAYSEQ *pmps = (const MMD2PLAYSEQ *)(lpStream + pseq);
				if(m_songName.empty()) mpt::String::Read<mpt::String::maybeNullTerminated>(m_songName, pmps->name);
				ORDERINDEX n = std::min<ORDERINDEX>(pmps->length, MAX_ORDERS - nOrders);
				if (n <= (dwMemLength - pseq + 42) / 2u && n < MPT_ARRAY_COUNT(pmps->seq))
				{
					Order().resize(nOrders + n);
					for (uint32 i=0; i<n; i++)
					{
						uint16 seqval = pmps->seq[i];
						if (seqval < wNumBlocks)
						{
							Order()[nOrders++] = static_cast<ORDERINDEX>(seqval);
						}
					}
				}
			}
		}
		playtransp = pmsh2->playtransp;
	}
	// Reading Expansion structure
	if (pmex)
	{
		// Channel Split
		if ((m_nChannels == 4) && (pmsh->flags & MMD_FLAG_8CHANNEL))
		{
			for (uint32 i8ch=0; i8ch<4; i8ch++)
			{
				if (pmex->channelsplit[i8ch]) m_nChannels++;
			}
		}
		// Song Comments (null-terminated)
		uint32 annotxt = pmex->annotxt;
		uint32 annolen = pmex->annolen;
		annolen = std::min<uint32>(annolen, MED_MAX_COMMENT_LENGTH); //Thanks to Luigi Auriemma for pointing out an overflow risk
		if ((annotxt) && (annolen) && (annolen <= dwMemLength) && (annotxt <= dwMemLength - annolen) )
		{
			m_songMessage.Read(lpStream + annotxt, annolen - 1, SongMessage::leAutodetect);
		}
		// Song Name
		uint32 songname = pmex->songname;
		uint32 songnamelen = pmex->songnamelen;
		if ((songname) && (songnamelen) && (songname <= dwMemLength) && (songnamelen <= dwMemLength-songname))
		{
			mpt::String::Read<mpt::String::maybeNullTerminated>(m_songName, lpStream + songname, songnamelen);
		}
		// Sample Names
		uint32 smpinfoex = pmex->iinfo;
		if (smpinfoex)
		{
			uint32 iinfoptr = pmex->iinfo;
			uint32 ientries = pmex->i_ext_entries;
			uint32 ientrysz = pmex->i_ext_entrsz;

			if ((iinfoptr) && (ientrysz < 256) && (ientries*ientrysz < dwMemLength) && (iinfoptr < dwMemLength - ientries*ientrysz))
			{
				const char *psznames = (const char *)(lpStream + iinfoptr);
				for (uint32 i=0; i<ientries; i++) if (i < m_nSamples)
				{
					mpt::String::Read<mpt::String::maybeNullTerminated>(m_szNames[i + 1], (psznames + i * ientrysz), ientrysz);
				}
			}
		}
		// Track Names
		uint32 trackinfo_ofs = pmex->trackinfo_ofs;
		if ((trackinfo_ofs) && (trackinfo_ofs < dwMemLength) && (m_nChannels * 4u < dwMemLength - trackinfo_ofs))
		{
			const_unaligned_ptr_be<uint32> ptrktags = const_unaligned_ptr_be<uint32>(lpStream + trackinfo_ofs);
			for (uint32 i=0; i<m_nChannels; i++)
			{
				uint32 trknameofs = 0, trknamelen = 0;
				uint32 trktagofs = ptrktags[i];
				if (trktagofs && (trktagofs <= dwMemLength - 8) )
				{
					while (trktagofs < dwMemLength - 8)
					{
						uint32 ntag = *const_unaligned_ptr_be<uint32>(lpStream + trktagofs);
						if (ntag == MMDTAG_END) break;
						uint32 tagdata = *const_unaligned_ptr_be<uint32>(lpStream + trktagofs + 4);
						switch(ntag)
						{
						case MMDTAG_TRK_NAMELEN:	trknamelen = tagdata; break;
						case MMDTAG_TRK_NAME:		trknameofs = tagdata; break;
						}
						trktagofs += 8;
					}
					if ((trknameofs) && (trknameofs < dwMemLength - trknamelen) && trknamelen < dwMemLength)
					{
						mpt::String::Read<mpt::String::maybeNullTerminated>(ChnSettings[i].szName, lpStream + trknameofs, trknamelen);
					}
				}
			}
		}
	}
	// Reading samples
	if (dwSmplArr > dwMemLength - 4*m_nSamples) return true;
	pdwTable = const_unaligned_ptr_be<uint32>(lpStream + dwSmplArr);
	for (uint32 iSmp=0; iSmp<m_nSamples; iSmp++) if (pdwTable[iSmp])
	{
		uint32 dwPos = pdwTable[iSmp];
		if ((dwPos >= dwMemLength) || (dwPos + sizeof(MMDSAMPLEHEADER) >= dwMemLength)) continue;
		const MMDSAMPLEHEADER *psdh = (const MMDSAMPLEHEADER *)(lpStream + dwPos);
		uint32 len = psdh->length;
	#ifdef MED_LOG
		Log("SampleData %d: stype=0x%02X len=%d\n", iSmp, psdh->type, len);
	#endif
		if(dwPos + len + 6 > dwMemLength) len = 0;
		uint32 stype = psdh->type;
		const char *psdata = (const char *)(lpStream + dwPos + 6);

		SampleIO sampleIO(
			SampleIO::_8bit,
			SampleIO::mono,
			SampleIO::bigEndian,
			SampleIO::signedPCM);

		if (stype & 0x80)
		{
			psdata += (stype & 0x20) ? 14 : 6;
		} else
		{
			if(stype & 0x10)
			{
				sampleIO |= SampleIO::_16bit;
				len /= 2;
			}
			if(stype & 0x20)
			{
				sampleIO |= SampleIO::stereoSplit;
				len /= 2;
			}
		}
		Samples[iSmp + 1].nLength = len;
		if(loadFlags & loadSampleData)
		{
			FileReader chunk(mpt::byte_cast<mpt::const_byte_span>(mpt::as_span(psdata, dwMemLength - dwPos - 6)));
			sampleIO.ReadSample(Samples[iSmp + 1], chunk);
		}
	}
	// Reading patterns (blocks)
	if(!(loadFlags & loadPatternData))
	{
		return true;
	}
	if (wNumBlocks > MAX_PATTERNS) wNumBlocks = MAX_PATTERNS;
	if ((!dwBlockArr) || (dwBlockArr > dwMemLength - 4u*wNumBlocks) || (4u*wNumBlocks > dwMemLength)) return true;
	pdwTable = const_unaligned_ptr_be<uint32>(lpStream + dwBlockArr);
	playtransp += (version == '3') ? 24 : 48;
	Patterns.ResizeArray(wNumBlocks);
	for (PATTERNINDEX iBlk=0; iBlk<wNumBlocks; iBlk++)
	{
		uint32 dwPos = pdwTable[iBlk];
		if ((!dwPos) || (dwPos >= dwMemLength) || (dwPos >= dwMemLength - 8)) continue;
		uint32 lines = 64, tracks = 4;
		if (version == '0')
		{
			const MMD0BLOCK *pmb = (const MMD0BLOCK *)(lpStream + dwPos);
			lines = pmb->lines + 1;
			tracks = pmb->numtracks;
			if (!tracks) tracks = m_nChannels;
			if(!Patterns.Insert(iBlk, lines)) continue;
			auto p = Patterns[iBlk].begin();
			const uint8 * s = (const uint8 *)(lpStream + dwPos + 2);
			uint32 maxlen = tracks*lines*3;
			if (maxlen + dwPos > dwMemLength - 2) break;
			for (uint32 y=0; y<lines; y++)
			{
				for (uint32 x=0; x<tracks; x++, s+=3) if (x < m_nChannels)
				{
					uint8 note = s[0] & 0x3F;
					uint8 instr = s[1] >> 4;
					if (s[0] & 0x80) instr |= 0x10;
					if (s[0] & 0x40) instr |= 0x20;
					if ((note) && (note <= 132)) p->note = static_cast<uint8>(note + playtransp);
					p->instr = instr;
					p->command = s[1] & 0x0F;
					p->param = s[2];
					// if (!iBlk) Log("%02X.%02X.%02X | ", s[0], s[1], s[2]);
					MedConvert(*p, pmsh);
					p++;
				}
				//if (!iBlk) Log("\n");
			}
		} else
		{
			const MMD1BLOCK *pmb = (const MMD1BLOCK *)(lpStream + dwPos);
		#ifdef MED_LOG
			Log("MMD1BLOCK:   lines=%2d, tracks=%2d, offset=0x%04X\n",
				pmb->lines, pmb->numtracks, pmb->info);
		#endif
			const MMD1BLOCKINFO *pbi = NULL;
			const uint8 *pcmdext = NULL;
			lines = pmb->lines + 1;
			tracks = pmb->numtracks;
			if (!tracks) tracks = m_nChannels;
			Patterns.Insert(iBlk, lines);
			uint32 dwBlockInfo = pmb->info;
			if ((dwBlockInfo) && (dwBlockInfo < dwMemLength - sizeof(MMD1BLOCKINFO)))
			{
				pbi = (const MMD1BLOCKINFO *)(lpStream + dwBlockInfo);
			#ifdef MED_LOG
				Log("  BLOCKINFO: blockname=0x%04X namelen=%d pagetable=0x%04X &cmdexttable=0x%04X\n",
					pbi->blockname, pbi->blocknamelen, pbi->pagetable, pbi->cmdexttable);
			#endif
				if ((pbi->blockname) && (pbi->blocknamelen))
				{
					uint32 nameofs = pbi->blockname;
					uint32 namelen = pbi->blocknamelen;
					if ((nameofs < dwMemLength) && (namelen < dwMemLength - nameofs))
					{
						Patterns[iBlk].SetName((const char *)(lpStream + nameofs), namelen);
					}
				}
				if (pbi->cmdexttable)
				{
					uint32 cmdexttable = pbi->cmdexttable;
					if (cmdexttable < dwMemLength - 4)
					{
						cmdexttable = *const_unaligned_ptr_be<uint32>(lpStream + cmdexttable);
						if ((cmdexttable) && (cmdexttable <= dwMemLength - lines*tracks))
						{
							pcmdext = (const uint8 *)(lpStream + cmdexttable);
						}
					}
				}
			}
			const uint8 * s = (const uint8 *)(lpStream + dwPos + 8);
			uint32 maxlen = tracks*lines*4;
			if (maxlen + dwPos > dwMemLength - 8 || !Patterns.IsValidPat(iBlk)) break;
			auto p = Patterns[iBlk].begin();
			for (uint32 y=0; y<lines; y++)
			{
				for (uint32 x=0; x<tracks; x++, s+=4) if (x < m_nChannels)
				{
					uint8 note = s[0];
					if ((note) && (note <= 132))
					{
						int rnote = note + playtransp;
						if (rnote < 1) rnote = 1;
						if (rnote > NOTE_MAX) rnote = NOTE_MAX;
						p->note = (uint8)rnote;
					}
					p->instr = s[1];
					p->command = s[2];
					p->param = s[3];
					if (pcmdext) p->vol = pcmdext[x];
					MedConvert(*p, pmsh);
					p++;
				}
				if (pcmdext) pcmdext += tracks;
			}
		}
	}
	return true;
}


OPENMPT_NAMESPACE_END
