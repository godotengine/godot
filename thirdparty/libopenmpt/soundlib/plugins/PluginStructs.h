/*
 * PluginStructs.h
 * ---------------
 * Purpose: Basic plugin structs for CSoundFile.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../Snd_defs.h"
#ifndef NO_PLUGINS
#include "../../common/Endianness.h"
#endif // NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

////////////////////////////////////////////////////////////////////
// Mix Plugins

typedef int32 PlugParamIndex;
typedef float PlugParamValue;

struct SNDMIXPLUGINSTATE;
struct SNDMIXPLUGIN;
class IMixPlugin;
class CSoundFile;

#ifndef NO_PLUGINS

struct SNDMIXPLUGININFO
{
	// dwInputRouting flags
	enum RoutingFlags
	{
		irApplyToMaster	= 0x01,	// Apply to master mix
		irBypass		= 0x02,	// Bypass effect
		irWetMix		= 0x04,	// Wet Mix (dry added)
		irExpandMix		= 0x08,	// [0%,100%] -> [-200%,200%]
		irAutoSuspend	= 0x10,	// Plugin will automatically suspend on silence
	};

	int32le dwPluginId1;			// Plugin type (kEffectMagic, kDmoMagic, kBuzzMagic)
	int32le dwPluginId2;			// Plugin unique ID
	uint8le routingFlags;			// See RoutingFlags
	uint8le mixMode;
	uint8le gain;					// Divide by 10 to get real gain
	uint8le reserved;
	uint32le dwOutputRouting;		// 0 = send to master 0x80 + x = send to plugin x
	uint32le dwReserved[4];			// Reserved for routing info
	char    szName[32];				// User-chosen plugin display name - this is locale ANSI!
	char    szLibraryName[64];		// original DLL name - this is UTF-8!

	// Should only be called from SNDMIXPLUGIN::SetBypass() and IMixPlugin::Bypass()
	void SetBypass(bool bypass = true) { if(bypass) routingFlags |= irBypass; else routingFlags &= uint8(~irBypass); }
};

MPT_BINARY_STRUCT(SNDMIXPLUGININFO, 128)	// this is directly written to files, so the size must be correct!


struct SNDMIXPLUGIN
{
	IMixPlugin *pMixPlugin;
	std::vector<mpt::byte> pluginData;
	SNDMIXPLUGININFO Info;
	float fDryRatio;
	int32 defaultProgram;
	int32 editorX, editorY;

	SNDMIXPLUGIN()
		: pMixPlugin(nullptr)
		, fDryRatio(0.0f)
		, defaultProgram(0)
		, editorX(0), editorY(0)
	{
		MemsetZero(Info);
	}

	const char *GetName() const
		{ return Info.szName; }
	const char *GetLibraryName() const
		{ return Info.szLibraryName; }

	// Check if a plugin is loaded into this slot (also returns true if the plugin in this slot has not been found)
	bool IsValidPlugin() const { return (Info.dwPluginId1 | Info.dwPluginId2) != 0; };

	// Input routing getters
	uint8 GetGain() const
		{ return Info.gain; }
	uint8 GetMixMode() const
		{ return Info.mixMode; }
	bool IsMasterEffect() const
		{ return (Info.routingFlags & SNDMIXPLUGININFO::irApplyToMaster) != 0; }
	bool IsWetMix() const
		{ return (Info.routingFlags & SNDMIXPLUGININFO::irWetMix) != 0; }
	bool IsExpandedMix() const
		{ return (Info.routingFlags & SNDMIXPLUGININFO::irExpandMix) != 0; }
	bool IsBypassed() const
		{ return (Info.routingFlags & SNDMIXPLUGININFO::irBypass) != 0; }
	bool IsAutoSuspendable() const
		{ return (Info.routingFlags & SNDMIXPLUGININFO::irAutoSuspend) != 0; }

	// Input routing setters
	void SetGain(uint8 gain);
	void SetMixMode(uint8 mixMode)
		{ Info.mixMode = mixMode; }
	void SetMasterEffect(bool master = true)
		{ if(master) Info.routingFlags |= SNDMIXPLUGININFO::irApplyToMaster; else Info.routingFlags &= uint8(~SNDMIXPLUGININFO::irApplyToMaster); }
	void SetWetMix(bool wetMix = true)
		{ if(wetMix) Info.routingFlags |= SNDMIXPLUGININFO::irWetMix; else Info.routingFlags &= uint8(~SNDMIXPLUGININFO::irWetMix); }
	void SetExpandedMix(bool expanded = true)
		{ if(expanded) Info.routingFlags |= SNDMIXPLUGININFO::irExpandMix; else Info.routingFlags &= uint8(~SNDMIXPLUGININFO::irExpandMix); }
	void SetBypass(bool bypass = true);
	void SetAutoSuspend(bool suspend = true)
		{ if(suspend) Info.routingFlags |= SNDMIXPLUGININFO::irAutoSuspend; else Info.routingFlags &= uint8(~SNDMIXPLUGININFO::irAutoSuspend); }

	// Output routing getters
	bool IsOutputToMaster() const
		{ return Info.dwOutputRouting == 0; }
	PLUGINDEX GetOutputPlugin() const
		{ return Info.dwOutputRouting >= 0x80 ? static_cast<PLUGINDEX>(Info.dwOutputRouting - 0x80) : PLUGINDEX_INVALID; }

	// Output routing setters
	void SetOutputToMaster()
		{ Info.dwOutputRouting = 0; }
	void SetOutputPlugin(PLUGINDEX plugin)
		{ if(plugin < MAX_MIXPLUGINS) Info.dwOutputRouting = plugin + 0x80; else Info.dwOutputRouting = 0; }

	void Destroy();
};

bool CreateMixPluginProc(SNDMIXPLUGIN &mixPlugin, CSoundFile &sndFile);

#endif // NO_PLUGINS

OPENMPT_NAMESPACE_END
