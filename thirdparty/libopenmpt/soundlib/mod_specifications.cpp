/*
 * mod_specifications.cpp
 * ----------------------
 * Purpose: Mod specifications characterise the features of every editable module format in OpenMPT, such as the number of supported channels, samples, effects, etc...
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mod_specifications.h"
#include "../common/misc_util.h"
#include <algorithm>

OPENMPT_NAMESPACE_BEGIN

namespace ModSpecs
{


// Force built-in integer operations.
// C++11 constexpr operations on the enum value_type would also solve this.
#define SongFlag FlagSet<SongFlags>::store_type


MPT_CONSTEXPR11_VAR CModSpecifications mptm_ =
{
	/*
	TODO: Proper, less arbitrarily chosen values here.
	NOTE: If changing limits, see whether:
			-savefile format and GUI methods can handle new values(might not be a small task :).
	 */
	MOD_TYPE_MPT,								// Internal MODTYPE value
	"mptm",										// File extension
	NOTE_MIN,									// Minimum note index
	NOTE_MAX,									// Maximum note index
	4000,										// Pattern max.
	4000,										// Order max.
	MAX_SEQUENCES,								// Sequences max
	1,											// Channel min
	127,										// Channel max
	32,											// Min tempo
	512,										// Max tempo
	1,											// Min Speed
	255,										// Max Speed
	1,											// Min pattern rows
	1024,										// Max pattern rows
	25,											// Max mod name length
	25,											// Max sample name length
	12,											// Max sample filename length
	25,											// Max instrument name length
	12,											// Max instrument filename length
	0,											// Max comment line length
	3999,										// SamplesMax
	255,										// instrumentMax
	mixLevels1_17RC3,							// defaultMixLevels
	SongFlag(0) | SONG_LINEARSLIDES | SONG_EXFILTERRANGE | SONG_ITOLDEFFECTS | SONG_ITCOMPATGXX,	// Supported song flags
	200,										// Max MIDI mapping directives
	MAX_ENVPOINTS,								// Envelope point count
	true,										// Has notecut.
	true,										// Has noteoff.
	true,										// Has notefade.
	true,										// Has envelope release node
	true,										// Has song comments
	true,										// Has "+++" pattern
	true,										// Has "---" pattern
	true,										// Has restart position (order)
	true,										// Supports plugins
	true,										// Custom pattern time signatures
	true,										// Pattern names
	true,										// Has artist name
	true,										// Has default resampling
	true,										// Fixed point tempo
	" JFEGHLKRXODB?CQATI?SMNVW?UY?P?Z\\:#???????",	// Supported Effects
	" vpcdabuh??gfe?o",							// Supported Volume Column commands
};




MPT_CONSTEXPR11_VAR CModSpecifications mod_ =
{
	MOD_TYPE_MOD,								// Internal MODTYPE value
	"mod",										// File extension
	37,											// Minimum note index
	108,										// Maximum note index
	128,										// Pattern max.
	128,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	99,											// Channel max
	32,											// Min tempo
	255,										// Max tempo
	1,											// Min Speed
	31,											// Max Speed
	64,											// Min pattern rows
	64,											// Max pattern rows
	20,											// Max mod name length
	22,											// Max sample name length
	0,											// Max sample filename length
	0,											// Max instrument name length
	0,											// Max instrument filename length
	0,											// Max comment line length
	31,											// SamplesMax
	0,											// instrumentMax
	mixLevelsCompatible,						// defaultMixLevels
	SongFlag(0) | SONG_PT_MODE | SONG_AMIGALIMITS | SONG_ISAMIGA,	// Supported song flags
	0,											// Max MIDI mapping directives
	0,											// No instrument envelopes
	false,										// No notecut.
	false,										// No noteoff.
	false,										// No notefade.
	false,										// No envelope release node
	false,										// No song comments
	false,										// Doesn't have "+++" pattern
	false,										// Doesn't have "---" pattern
	true,										// Has restart position (order)
	false,										// Doesn't support plugins
	false,										// No custom pattern time signatures
	false,										// No pattern names
	false,										// Doesn't have artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" 0123456789ABCD?FF?E??????????????????????",	// Supported Effects
	" ???????????????",							// Supported Volume Column commands
};


MPT_CONSTEXPR11_VAR CModSpecifications xm_ =
{
	MOD_TYPE_XM,								// Internal MODTYPE value
	"xm",										// File extension
	13,											// Minimum note index
	108,										// Maximum note index
	256,										// Pattern max.
	255,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	32,											// Channel max
	32,											// Min tempo
	512,										// Max tempo
	1,											// Min Speed
	31,											// Max Speed
	1,											// Min pattern rows
	256,										// Max pattern rows
	20,											// Max mod name length
	22,											// Max sample name length
	0,											// Max sample filename length
	22,											// Max instrument name length
	0,											// Max instrument filename length
	0,											// Max comment line length
	128 * 16,									// SamplesMax (actually 16 per instrument)
	128,										// instrumentMax
	mixLevelsCompatibleFT2,						// defaultMixLevels
	SongFlag(0) | SONG_LINEARSLIDES,			// Supported song flags
	0,											// Max MIDI mapping directives
	12,											// Envelope point count
	false,										// No notecut.
	true,										// Has noteoff.
	false,										// No notefade.
	false,										// No envelope release node
	false,										// No song comments
	false,										// Doesn't have "+++" pattern
	false,										// Doesn't have "---" pattern
	true,										// Has restart position (order)
	false,										// Doesn't support plugins
	false,										// No custom pattern time signatures
	false,										// No pattern names
	false,										// Doesn't have artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" 0123456789ABCDRFFTE???GHK??XPL???????????",	// Supported Effects
	" vpcdabuhlrg????",							// Supported Volume Column commands
};

// XM with MPT extensions
MPT_CONSTEXPR11_VAR CModSpecifications xmEx_ =
{
	MOD_TYPE_XM,								// Internal MODTYPE value
	"xm",										// File extension
	13,											// Minimum note index
	108,										// Maximum note index
	256,										// Pattern max.
	255,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	127,										// Channel max
	32,											// Min tempo
	512,										// Max tempo
	1,											// Min Speed
	31,											// Max Speed
	1,											// Min pattern rows
	1024,										// Max pattern rows
	20,											// Max mod name length
	22,											// Max sample name length
	0,											// Max sample filename length
	22,											// Max instrument name length
	0,											// Max instrument filename length
	0,											// Max comment line length
	MAX_SAMPLES - 1,							// SamplesMax (actually 32 per instrument(256 * 32 = 8192), but limited to MAX_SAMPLES = 4000)
	255,										// instrumentMax
	mixLevelsCompatibleFT2,						// defaultMixLevels
	SongFlag(0) | SONG_LINEARSLIDES | SONG_EXFILTERRANGE,	// Supported song flags
	200,										// Max MIDI mapping directives
	12,											// Envelope point count
	false,										// No notecut.
	true,										// Has noteoff.
	false,										// No notefade.
	false,										// No envelope release node
	true,										// Has song comments
	false,										// Doesn't have "+++" pattern
	false,										// Doesn't have "---" pattern
	true,										// Has restart position (order)
	true,										// Supports plugins
	false,										// No custom pattern time signatures
	true,										// Pattern names
	true,										// Has artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" 0123456789ABCDRFFTE???GHK?YXPLZ\\?#???????",	// Supported Effects
	" vpcdabuhlrg????",							// Supported Volume Column commands
};

MPT_CONSTEXPR11_VAR CModSpecifications s3m_ =
{
	MOD_TYPE_S3M,								// Internal MODTYPE value
	"s3m",										// File extension
	13,											// Minimum note index
	108,										// Maximum note index
	100,										// Pattern max.
	255,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	32,											// Channel max
	33,											// Min tempo
	255,										// Max tempo
	1,											// Min Speed
	255,										// Max Speed
	64,											// Min pattern rows
	64,											// Max pattern rows
	27,											// Max mod name length
	27,											// Max sample name length
	12,											// Max sample filename length
	0,											// Max instrument name length
	0,											// Max instrument filename length
	0,											// Max comment line length
	99,											// SamplesMax
	0,											// instrumentMax
	mixLevelsCompatible,						// defaultMixLevels
	SongFlag(0) | SONG_FASTVOLSLIDES | SONG_AMIGALIMITS | SONG_S3MOLDVIBRATO,	// Supported song flags
	0,											// Max MIDI mapping directives
	0,											// No instrument envelopes
	true,										// Has notecut.
	false,										// No noteoff.
	false,										// No notefade.
	false,										// No envelope release node
	false,										// No song comments
	true,										// Has "+++" pattern
	true,										// Has "---" pattern
	false,										// Doesn't have restart position (order)
	false,										// Doesn't support plugins
	false,										// No custom pattern time signatures
	false,										// No pattern names
	false,										// Doesn't have artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" JFEGHLKRXODB?CQATI?SMNVW?U???????????????",	// Supported Effects
	" vp?????????????",							// Supported Volume Column commands
};

// S3M with MPT extensions
MPT_CONSTEXPR11_VAR CModSpecifications s3mEx_ =
{
	MOD_TYPE_S3M,								// Internal MODTYPE value
	"s3m",										// File extension
	13,											// Minimum note index
	108,										// Maximum note index
	100,										// Pattern max.
	255,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	32,											// Channel max
	33,											// Min tempo
	255,										// Max tempo
	1,											// Min Speed
	255,										// Max Speed
	64,											// Min pattern rows
	64,											// Max pattern rows
	27,											// Max mod name length
	27,											// Max sample name length
	12,											// Max sample filename length
	0,											// Max instrument name length
	0,											// Max instrument filename length
	0,											// Max comment line length
	99,											// SamplesMax
	0,											// instrumentMax
	mixLevelsCompatible,						// defaultMixLevels
	SongFlag(0) | SONG_FASTVOLSLIDES | SONG_AMIGALIMITS,	// Supported song flags
	0,											// Max MIDI mapping directives
	0,											// No instrument envelopes
	true,										// Has notecut.
	false,										// No noteoff.
	false,										// No notefade.
	false,										// No envelope release node
	false,										// No song comments
	true,										// Has "+++" pattern
	true,										// Has "---" pattern
	false,										// Doesn't have restart position (order)
	false,										// Doesn't support plugins
	false,										// No custom pattern time signatures
	false,										// No pattern names
	false,										// Doesn't have artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" JFEGHLKRXODB?CQATI?SMNVW?UY?P?Z??????????",	// Supported Effects
	" vp?????????????",							// Supported Volume Column commands
};

MPT_CONSTEXPR11_VAR CModSpecifications it_ =
{
	MOD_TYPE_IT,								// Internal MODTYPE value
	"it",										// File extension
	1,											// Minimum note index
	120,										// Maximum note index
	200,										// Pattern max.
	256,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	64,											// Channel max
	32,											// Min tempo
	255,										// Max tempo
	1,											// Min Speed
	255,										// Max Speed
	1,											// Min pattern rows
	200,										// Max pattern rows
	25,											// Max mod name length
	25,											// Max sample name length
	12,											// Max sample filename length
	25,											// Max instrument name length
	12,											// Max instrument filename length
	75,											// Max comment line length
	99,											// SamplesMax
	99,											// instrumentMax
	mixLevelsCompatible,						// defaultMixLevels
	SongFlag(0) | SONG_LINEARSLIDES | SONG_ITOLDEFFECTS | SONG_ITCOMPATGXX,	// Supported song flags
	0,											// Max MIDI mapping directives
	25,											// Envelope point count
	true,										// Has notecut.
	true,										// Has noteoff.
	true,										// Has notefade.
	false,										// No envelope release node
	true,										// Has song comments
	true,										// Has "+++" pattern
	true,										// Has "--" pattern
	false,										// Doesn't have restart position (order)
	false,										// Doesn't support plugins
	false,										// No custom pattern time signatures
	false,										// No pattern names
	false,										// Doesn't have artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" JFEGHLKRXODB?CQATI?SMNVW?UY?P?Z??????????",	// Supported Effects
	" vpcdab?h??gfe??",							// Supported Volume Column commands
};

MPT_CONSTEXPR11_VAR CModSpecifications itEx_ =
{
	MOD_TYPE_IT,								// Internal MODTYPE value
	"it",										// File extension
	1,											// Minimum note index
	120,										// Maximum note index
	240,										// Pattern max.
	256,										// Order max.
	1,											// Only one order list
	1,											// Channel min
	127,										// Channel max
	32,											// Min tempo
	512,										// Max tempo
	1,											// Min Speed
	255,										// Max Speed
	1,											// Min pattern rows
	1024,										// Max pattern rows
	25,											// Max mod name length
	25,											// Max sample name length
	12,											// Max sample filename length
	25,											// Max instrument name length
	12,											// Max instrument filename length
	75,											// Max comment line length
	3999,										// SamplesMax
	255,										// instrumentMax
	mixLevelsCompatible,						// defaultMixLevels
	SongFlag(0) | SONG_LINEARSLIDES | SONG_EXFILTERRANGE | SONG_ITOLDEFFECTS | SONG_ITCOMPATGXX,	// Supported song flags
	200,										// Max MIDI mapping directives
	25,											// Envelope point count
	true,										// Has notecut.
	true,										// Has noteoff.
	true,										// Has notefade.
	false,										// No envelope release node
	true,										// Has song comments
	true,										// Has "+++" pattern
	true,										// Has "---" pattern
	false,										// Doesn't have restart position (order)
	true,										// Supports plugins
	false,										// No custom pattern time signatures
	true,										// Pattern names
	true,										// Has artist name
	false,										// Doesn't have default resampling
	false,										// Integer tempo
	" JFEGHLKRXODB?CQATI?SMNVW?UY?P?Z\\?#???????",	// Supported Effects
	" vpcdab?h??gfe??",							// Supported Volume Column commands
};

const CModSpecifications *Collection[8] = { &mptm_, &mod_, &s3m_, &s3mEx_, &xm_, &xmEx_, &it_, &itEx_ };

const CModSpecifications & mptm = mptm_;
const CModSpecifications & mod = mod_;
const CModSpecifications & s3m = s3m_;
const CModSpecifications & s3mEx = s3mEx_;
const CModSpecifications & xm = xm_;
const CModSpecifications & xmEx = xmEx_;
const CModSpecifications & it = it_;
const CModSpecifications & itEx = itEx_;

} // namespace ModSpecs


MODTYPE CModSpecifications::ExtensionToType(std::string ext)
{
	if(ext == "")
	{
		return MOD_TYPE_NONE;
	}
	if(ext.length() > 0 && ext[0] == '.')
	{
		ext.erase(0, 1);
	}
	ext = mpt::ToLowerCaseAscii(ext);
	for(std::size_t i = 0; i < CountOf(ModSpecs::Collection); i++)
	{
		if(ext == ModSpecs::Collection[i]->fileExtension)
		{
			return ModSpecs::Collection[i]->internalType;
		}
	}
	return MOD_TYPE_NONE;
}


bool CModSpecifications::HasNote(ModCommand::NOTE note) const
{
	if(note >= noteMin && note <= noteMax)
		return true;
	else if(ModCommand::IsSpecialNote(note))
	{
		if(note == NOTE_NOTECUT)
			return hasNoteCut;
		else if(note == NOTE_KEYOFF)
			return hasNoteOff;
		else if(note == NOTE_FADE)
			return hasNoteFade;
		else
			return (internalType == MOD_TYPE_MPT);
	} else if(note == NOTE_NONE)
		return true;
	return false;
}


bool CModSpecifications::HasVolCommand(ModCommand::VOLCMD volcmd) const
{
	if(volcmd >= MAX_VOLCMDS) return false;
	if(volcommands[volcmd] == '?') return false;
	return true;
}


bool CModSpecifications::HasCommand(ModCommand::COMMAND cmd) const
{
	if(cmd >= MAX_EFFECTS) return false;
	if(commands[cmd] == '?') return false;
	return true;
}


char CModSpecifications::GetVolEffectLetter(ModCommand::VOLCMD volcmd) const
{
	if(volcmd >= MAX_VOLCMDS) return '?';
	return volcommands[volcmd];
}


char CModSpecifications::GetEffectLetter(ModCommand::COMMAND cmd) const
{
	if(cmd >= MAX_EFFECTS) return '?';
	return commands[cmd];
}


OPENMPT_NAMESPACE_END
