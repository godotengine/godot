/*
 * MIDIMacros.cpp
 * --------------
 * Purpose: Helper functions / classes for MIDI Macro functionality.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "../soundlib/MIDIEvents.h"
#include "MIDIMacros.h"
#include "../common/StringFixer.h"
#include "../common/misc_util.h"

#ifdef MODPLUG_TRACKER
#include "Sndfile.h"
#include "plugins/PlugInterface.h"
#endif // MODPLUG_TRACKER

OPENMPT_NAMESPACE_BEGIN

parameteredMacroType MIDIMacroConfig::GetParameteredMacroType(uint32 macroIndex) const
{
	const std::string macro = GetSafeMacro(szMidiSFXExt[macroIndex]);

	for(uint32 i = 0; i < sfx_max; i++)
	{
		parameteredMacroType sfx = static_cast<parameteredMacroType>(i);
		if(sfx != sfx_custom)
		{
			if(macro.compare(CreateParameteredMacro(sfx)) == 0) return sfx;
		}
	}

	// Special macros with additional "parameter":
	if (macro.compare(CreateParameteredMacro(sfx_cc, MIDIEvents::MIDICC_start)) >= 0 && macro.compare(CreateParameteredMacro(sfx_cc, MIDIEvents::MIDICC_end)) <= 0 && macro.size() == 5)
		return sfx_cc;
	if (macro.compare(CreateParameteredMacro(sfx_plug, 0)) >= 0 && macro.compare(CreateParameteredMacro(sfx_plug, 0x17F)) <= 0 && macro.size() == 7)
		return sfx_plug; 

	return sfx_custom;	// custom / unknown
}


// Retrieve Zxx (Z80-ZFF) type from current macro configuration
fixedMacroType MIDIMacroConfig::GetFixedMacroType() const
{
	// Compare with all possible preset patterns
	for(uint32 i = 0; i < zxx_max; i++)
	{
		fixedMacroType zxx = static_cast<fixedMacroType>(i);
		if(zxx != zxx_custom)
		{
			// Prepare macro pattern to compare
			char macros[128][MACRO_LENGTH];
			CreateFixedMacro(macros, zxx);

			bool found = true;
			for(uint32 j = 0; j < 128; j++)
			{
				if(strncmp(macros[j], szMidiZXXExt[j], MACRO_LENGTH))
				{
					found = false;
					break;
				}
			}
			if(found) return zxx;
		}
	}
	return zxx_custom; // Custom setup
}


void MIDIMacroConfig::CreateParameteredMacro(char (&parameteredMacro)[MACRO_LENGTH], parameteredMacroType macroType, int subType) const
{
	switch(macroType)
	{
	case sfx_unused:
		strcpy(parameteredMacro, "");
		break;
	case sfx_cutoff:
		strcpy(parameteredMacro, "F0F000z");
		break;
	case sfx_reso:
		strcpy(parameteredMacro, "F0F001z");
		break;
	case sfx_mode:
		strcpy(parameteredMacro, "F0F002z");
		break;
	case sfx_drywet:
		strcpy(parameteredMacro, "F0F003z");
		break;
	case sfx_cc:
		sprintf(parameteredMacro, "Bc%02Xz", (subType & 0x7F));
		break;
	case sfx_plug:
		sprintf(parameteredMacro, "F0F%03Xz", std::min(subType, 0x17F) + 0x80);
		break;
	case sfx_channelAT:
		strcpy(parameteredMacro, "Dcz");
		break;
	case sfx_polyAT:
		strcpy(parameteredMacro, "Acnz");
		break;
	case sfx_pitch:
		strcpy(parameteredMacro, "Ec00z");
		break;
	case sfx_custom:
	default:
		MPT_ASSERT_NOTREACHED();
		break;
	}
}


// Create Zxx (Z80 - ZFF) from one out of five presets
void MIDIMacroConfig::CreateFixedMacro(char (&fixedMacros)[128][MACRO_LENGTH], fixedMacroType macroType) const
{
	for(uint32 i = 0; i < 128; i++)
	{
		switch(macroType)
		{
		case zxx_unused:
			strcpy(fixedMacros[i], "");
			break;

		case zxx_reso4Bit:
			// Type 1 - Z80 - Z8F controls resonance
			if (i < 16) sprintf(fixedMacros[i], "F0F001%02X", i * 8);
			else strcpy(fixedMacros[i], "");
			break;

		case zxx_reso7Bit:
			// Type 2 - Z80 - ZFF controls resonance
			sprintf(fixedMacros[i], "F0F001%02X", i);
			break;

		case zxx_cutoff:
			// Type 3 - Z80 - ZFF controls cutoff
			sprintf(fixedMacros[i], "F0F000%02X", i);
			break;

		case zxx_mode:
			// Type 4 - Z80 - ZFF controls filter mode
			sprintf(fixedMacros[i], "F0F002%02X", i);
			break;

		case zxx_resomode:
			// Type 5 - Z80 - Z9F controls resonance + filter mode
			if (i < 16) sprintf(fixedMacros[i], "F0F001%02X", i * 8);
			else if (i < 32) sprintf(fixedMacros[i], "F0F002%02X", (i - 16) * 8);
			else strcpy(fixedMacros[i], "");
			break;

		case zxx_channelAT:
			// Type 6 - Z80 - ZFF controls Channel Aftertouch
			sprintf(fixedMacros[i], "Dc%02X", i);
			break;

		case zxx_polyAT:
			// Type 7 - Z80 - ZFF controls Poly Aftertouch
			sprintf(fixedMacros[i], "Acn%02X", i);
			break;

		case zxx_pitch:
			// Type 7 - Z80 - ZFF controls Pitch Bend
			sprintf(fixedMacros[i], "Ec00%02X", i);
			break;

		case zxx_custom:
		default:
			MPT_ASSERT_NOTREACHED();
			break;
		}
	}
}


#ifdef MODPLUG_TRACKER

bool MIDIMacroConfig::operator== (const MIDIMacroConfig &other) const
{
	for(uint32 i = 0; i < CountOf(szMidiGlb); i++)
	{
		if(strncmp(szMidiGlb[i], other.szMidiGlb[i], MACRO_LENGTH))
			return false;
	}
	for(uint32 i = 0; i < CountOf(szMidiSFXExt); i++)
	{
		if(strncmp(szMidiSFXExt[i], other.szMidiSFXExt[i], MACRO_LENGTH))
			return false;
	}
	for(uint32 i = 0; i < CountOf(szMidiZXXExt); i++)
	{
		if(strncmp(szMidiZXXExt[i], other.szMidiZXXExt[i], MACRO_LENGTH))
			return false;
	}
	return true;
}


// Returns macro description including plugin parameter / MIDI CC information
CString MIDIMacroConfig::GetParameteredMacroName(uint32 macroIndex, IMixPlugin *plugin) const
{
	const parameteredMacroType macroType = GetParameteredMacroType(macroIndex);

	switch(macroType)
	{
	case sfx_plug:
		{
			const int param = MacroToPlugParam(macroIndex);
			CString formattedName;
			formattedName.Format(_T("Param %u"), param);
#ifndef NO_PLUGINS
			if(plugin != nullptr)
			{
				CString paramName = plugin->GetParamName(param);
				if(!paramName.IsEmpty())
				{
					formattedName += _T(" (") + paramName + _T(")");
				}
			} else
#else
			MPT_UNREFERENCED_PARAMETER(plugin);
#endif // NO_PLUGINS
			{
				formattedName += _T(" (N/A)");
			}
			return formattedName;
		}

	case sfx_cc:
		{
			CString formattedCC;
			formattedCC.Format(_T("MIDI CC %u"), MacroToMidiCC(macroIndex));
			return formattedCC;
		}

	default:
		return GetParameteredMacroName(macroType);
	}
}


// Returns generic macro description.
CString MIDIMacroConfig::GetParameteredMacroName(parameteredMacroType macroType) const
{
	switch(macroType)
	{
	case sfx_unused:
		return _T("Unused");
	case sfx_cutoff:
		return _T("Set Filter Cutoff");
	case sfx_reso:
		return _T("Set Filter Resonance");
	case sfx_mode:
		return _T("Set Filter Mode");
	case sfx_drywet:
		return _T("Set Plugin Dry/Wet Ratio");
	case sfx_plug:
		return _T("Control Plugin Parameter...");
	case sfx_cc:
		return _T("MIDI CC...");
	case sfx_channelAT:
		return _T("Channel Aftertouch");
	case sfx_polyAT:
		return _T("Polyphonic Aftertouch");
	case sfx_pitch:
		return _T("Pitch Bend");
	case sfx_custom:
	default:
		return _T("Custom");
	}
}


// Returns generic macro description.
CString MIDIMacroConfig::GetFixedMacroName(fixedMacroType macroType) const
{
	switch(macroType)
	{
	case zxx_unused:
		return _T("Unused");
	case zxx_reso4Bit:
		return _T("Z80 - Z8F controls Resonant Filter Resonance");
	case zxx_reso7Bit:
		return _T("Z80 - ZFF controls Resonant Filter Resonance");
	case zxx_cutoff:
		return _T("Z80 - ZFF controls Resonant Filter Cutoff");
	case zxx_mode:
		return _T("Z80 - ZFF controls Resonant Filter Mode");
	case zxx_resomode:
		return _T("Z80 - Z9F controls Resonance + Filter Mode");
	case zxx_channelAT:
		return _T("Z80 - ZFF controls Channel Aftertouch");
	case zxx_polyAT:
		return _T("Z80 - ZFF controls Polyphonic Aftertouch");
	case zxx_pitch:
		return _T("Z80 - ZFF controls Pitch Bend");
	case zxx_custom:
	default:
		return _T("Custom");
	}
}


int MIDIMacroConfig::MacroToPlugParam(uint32 macroIndex) const
{
	const std::string macro = GetSafeMacro(szMidiSFXExt[macroIndex]);

	int code = 0;
	const char *param = macro.c_str();
	param += 4;
	if ((param[0] >= '0') && (param[0] <= '9')) code = (param[0] - '0') << 4; else
		if ((param[0] >= 'A') && (param[0] <= 'F')) code = (param[0] - 'A' + 0x0A) << 4;
	if ((param[1] >= '0') && (param[1] <= '9')) code += (param[1] - '0'); else
		if ((param[1] >= 'A') && (param[1] <= 'F')) code += (param[1] - 'A' + 0x0A);

	if (macro.size() >= 4 && macro.at(3) == '0')
		return (code - 128);
	else
		return (code + 128);
}


int MIDIMacroConfig::MacroToMidiCC(uint32 macroIndex) const
{
	const std::string macro = GetSafeMacro(szMidiSFXExt[macroIndex]);

	int code = 0;
	const char *param = macro.c_str();
	param += 2;
	if ((param[0] >= '0') && (param[0] <= '9')) code = (param[0] - '0') << 4; else
		if ((param[0] >= 'A') && (param[0] <= 'F')) code = (param[0] - 'A' + 0x0A) << 4;
	if ((param[1] >= '0') && (param[1] <= '9')) code += (param[1] - '0'); else
		if ((param[1] >= 'A') && (param[1] <= 'F')) code += (param[1] - 'A' + 0x0A);

	return code;
}


int MIDIMacroConfig::FindMacroForParam(PlugParamIndex param) const
{
	for(int macroIndex = 0; macroIndex < NUM_MACROS; macroIndex++)
	{
		if(GetParameteredMacroType(macroIndex) == sfx_plug && MacroToPlugParam(macroIndex) == param)
		{
			return macroIndex;
		}
	}

	return -1;
}

#endif // MODPLUG_TRACKER


// Check if the MIDI Macro configuration used is the default one,
// i.e. the configuration that is assumed when loading a file that has no macros embedded.
bool MIDIMacroConfig::IsMacroDefaultSetupUsed() const
{
	const MIDIMacroConfig defaultConfig;

	// TODO - Global macros (currently not checked because they are not editable)

	// SF0: Z00-Z7F controls cutoff, all other parametered macros are unused
	for(uint32 i = 0; i < NUM_MACROS; i++)
	{
		if(GetParameteredMacroType(i) != defaultConfig.GetParameteredMacroType(i))
		{
			return false;
		}
	}

	// Z80-Z8F controls resonance
	if(GetFixedMacroType() != defaultConfig.GetFixedMacroType())
	{
		return false;
	}

	return true;
}


// Reset MIDI macro config to default values.
void MIDIMacroConfig::Reset()
{
	MemsetZero(szMidiGlb);
	MemsetZero(szMidiSFXExt);
	MemsetZero(szMidiZXXExt);

	strcpy(szMidiGlb[MIDIOUT_START], "FF");
	strcpy(szMidiGlb[MIDIOUT_STOP], "FC");
	strcpy(szMidiGlb[MIDIOUT_NOTEON], "9c n v");
	strcpy(szMidiGlb[MIDIOUT_NOTEOFF], "9c n 0");
	strcpy(szMidiGlb[MIDIOUT_PROGRAM], "Cc p");
	// SF0: Z00-Z7F controls cutoff
	CreateParameteredMacro(0, sfx_cutoff);
	// Z80-Z8F controls resonance
	CreateFixedMacro(zxx_reso4Bit);
}


// Clear all Zxx macros so that they do nothing.
void MIDIMacroConfig::ClearZxxMacros()
{
	MemsetZero(szMidiSFXExt);
	MemsetZero(szMidiZXXExt);
}


// Sanitize all macro config strings.
void MIDIMacroConfig::Sanitize()
{
	for(uint32 i = 0; i < CountOf(szMidiGlb); i++)
	{
		mpt::String::FixNullString(szMidiGlb[i]);
	}
	for(uint32 i = 0; i < CountOf(szMidiSFXExt); i++)
	{
		mpt::String::FixNullString(szMidiSFXExt[i]);
	}
	for(uint32 i = 0; i < CountOf(szMidiZXXExt); i++)
	{
		mpt::String::FixNullString(szMidiZXXExt[i]);
	}
}


// Helper function for UpgradeMacros()
void MIDIMacroConfig::UpgradeMacroString(char *macro) const
{
	for(uint32 i = 0; i < MACRO_LENGTH; i++)
	{
		if(macro[i] >= 'a' && macro[i] <= 'f')		// both A-F and a-f were treated as hex constants
		{
			macro[i] = macro[i] - 'a' + 'A';
		} else if(macro[i] == 'K' || macro[i] == 'k')	// channel was K or k
		{
			macro[i] = 'c';
		} else if(macro[i] == 'X' || macro[i] == 'x' || macro[i] == 'Y' || macro[i] == 'y')	// those were pointless
		{
			macro[i] = 'z';
		}
	}
}


// Fix old-format (not conforming to IT's MIDI macro definitions) MIDI config strings.
void MIDIMacroConfig::UpgradeMacros()
{
	for(uint32 i = 0; i < CountOf(szMidiSFXExt); i++)
	{
		UpgradeMacroString(szMidiSFXExt[i]);
	}
	for(uint32 i = 0; i < CountOf(szMidiZXXExt); i++)
	{
		UpgradeMacroString(szMidiZXXExt[i]);
	}
}


// Normalize by removing blanks and other unwanted characters from macro strings for internal usage.
std::string MIDIMacroConfig::GetSafeMacro(const char *macro) const
{
	std::string sanitizedMacro = macro;

	std::string::size_type pos;
	while((pos = sanitizedMacro.find_first_not_of("0123456789ABCDEFabpcnuvxyz")) != std::string::npos)
	{
		sanitizedMacro.erase(pos, 1);
	}

	return sanitizedMacro;
}


OPENMPT_NAMESPACE_END
