/*
 * PluginManager.cpp
 * -----------------
 * Purpose: Implementation of the plugin manager, which keeps a list of known plugins and instantiates them.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS

#include "../../common/version.h"
#include "PluginManager.h"
#include "PlugInterface.h"

#include "../../common/mptUUID.h"

// Built-in plugins
#include "DigiBoosterEcho.h"
#include "LFOPlugin.h"
#include "dmo/DMOPlugin.h"
#include "dmo/Chorus.h"
#include "dmo/Compressor.h"
#include "dmo/Distortion.h"
#include "dmo/Echo.h"
#include "dmo/Flanger.h"
#include "dmo/Gargle.h"
#include "dmo/I3DL2Reverb.h"
#include "dmo/ParamEq.h"
#include "dmo/WavesReverb.h"
#ifdef MODPLUG_TRACKER
#include "../../mptrack/plugins/MidiInOut.h"
#endif // MODPLUG_TRACKER

#include "../../common/StringFixer.h"
#include "../Sndfile.h"
#include "../Loaders.h"

#ifndef NO_VST
#include "../../mptrack/Vstplug.h"
#include "../../pluginBridge/BridgeWrapper.h"
#endif // NO_VST

#ifndef NO_DMO
#include <winreg.h>
#include <strmif.h>
#include <tchar.h>
#endif // NO_DMO

#ifdef MODPLUG_TRACKER
#include "../../mptrack/Mptrack.h"
#include "../../mptrack/TrackerSettings.h"
#include "../../mptrack/AbstractVstEditor.h"
#include "../../soundlib/AudioCriticalSection.h"
#include "../common/mptCRC.h"
#endif // MODPLUG_TRACKER

OPENMPT_NAMESPACE_BEGIN

//#define VST_LOG
//#define DMO_LOG

#ifdef MODPLUG_TRACKER
static const MPT_UCHAR_TYPE *const cacheSection = MPT_ULITERAL("PluginCache");
#endif // MODPLUG_TRACKER


uint8 VSTPluginLib::GetDllBits(bool fromCache) const
{
	// Built-in plugins are always native.
	if(dllPath.empty())
		return sizeof(void *) * CHAR_BIT;
#ifndef NO_VST
	if(!dllBits || !fromCache)
	{
		dllBits = static_cast<uint8>(BridgeWrapper::GetPluginBinaryType(dllPath));
	}
#else
	MPT_UNREFERENCED_PARAMETER(fromCache);
#endif // NO_VST
	return dllBits;
}


// PluginCache format:
// FullDllPath = <ID1><ID2><CRC32> (hex-encoded)
// <ID1><ID2><CRC32>.Flags = Plugin Flags (see VSTPluginLib::DecodeCacheFlags).
// <ID1><ID2><CRC32>.Vendor = Plugin Vendor String.

#ifdef MODPLUG_TRACKER
void VSTPluginLib::WriteToCache() const
{
	SettingsContainer &cacheFile = theApp.GetPluginCache();

	const std::string crcName = dllPath.ToUTF8();
	const uint32 crc = mpt::crc32(crcName);
	const mpt::ustring IDs = mpt::ufmt::HEX0<8>(SwapBytesLE(pluginId1)) + mpt::ufmt::HEX0<8>(SwapBytesLE(pluginId2)) + mpt::ufmt::HEX0<8>(SwapBytesLE(crc));

	mpt::PathString writePath = dllPath;
	if(theApp.IsPortableMode())
	{
		writePath = theApp.AbsolutePathToRelative(writePath);
	}

	cacheFile.Write<mpt::ustring>(cacheSection, writePath.ToUnicode(), IDs);
	cacheFile.Write<CString>(cacheSection, IDs + MPT_USTRING(".Vendor"), vendor);
	cacheFile.Write<int32>(cacheSection, IDs + MPT_USTRING(".Flags"), EncodeCacheFlags());
}
#endif // MODPLUG_TRACKER


bool CreateMixPluginProc(SNDMIXPLUGIN &mixPlugin, CSoundFile &sndFile)
{
#ifdef MODPLUG_TRACKER
	CVstPluginManager *that = theApp.GetPluginManager();
	if(that)
	{
		return that->CreateMixPlugin(mixPlugin, sndFile);
	}
	return false;
#else
	if(!sndFile.m_PluginManager)
	{
		sndFile.m_PluginManager = mpt::make_unique<CVstPluginManager>();
	}
	return sndFile.m_PluginManager->CreateMixPlugin(mixPlugin, sndFile);
#endif // MODPLUG_TRACKER
}


CVstPluginManager::CVstPluginManager()
#ifndef NO_DMO
	: MustUnInitilizeCOM(false)
#endif
{

	#ifndef NO_DMO
		HRESULT COMinit = CoInitializeEx(NULL, COINIT_MULTITHREADED);
		if(COMinit == S_OK || COMinit == S_FALSE)
		{
			MustUnInitilizeCOM = true;
		}
	#endif

	// Hard-coded "plugins"
	static constexpr struct
	{
		VSTPluginLib::CreateProc createProc;
		const char *filename, *name;
		uint32 pluginId1, pluginId2;
		VSTPluginLib::PluginCategory category;
		bool isInstrument, isOurs;
	} BuiltInPlugins[] =
	{
		// DirectX Media Objects Emulation
		{ DMO::Chorus::Create,		"{EFE6629C-81F7-4281-BD91-C9D604A95AF6}", "Chorus",			kDmoMagic, 0xEFE6629C, VSTPluginLib::catDMO, false, false },
		{ DMO::Compressor::Create,	"{EF011F79-4000-406D-87AF-BFFB3FC39D57}", "Compressor",		kDmoMagic, 0xEF011F79, VSTPluginLib::catDMO, false, false },
		{ DMO::Distortion::Create,	"{EF114C90-CD1D-484E-96E5-09CFAF912A21}", "Distortion",		kDmoMagic, 0xEF114C90, VSTPluginLib::catDMO, false, false },
		{ DMO::Echo::Create,		"{EF3E932C-D40B-4F51-8CCF-3F98F1B29D5D}", "Echo",			kDmoMagic, 0xEF3E932C, VSTPluginLib::catDMO, false, false },
		{ DMO::Flanger::Create,		"{EFCA3D92-DFD8-4672-A603-7420894BAD98}", "Flanger",		kDmoMagic, 0xEFCA3D92, VSTPluginLib::catDMO, false, false },
		{ DMO::Gargle::Create,		"{DAFD8210-5711-4B91-9FE3-F75B7AE279BF}", "Gargle",			kDmoMagic, 0xDAFD8210, VSTPluginLib::catDMO, false, false },
		{ DMO::I3DL2Reverb::Create,	"{EF985E71-D5C7-42D4-BA4D-2D073E2E96F4}", "I3DL2Reverb",	kDmoMagic, 0xEF985E71, VSTPluginLib::catDMO, false, false },
		{ DMO::ParamEq::Create,		"{120CED89-3BF4-4173-A132-3CB406CF3231}", "ParamEq",		kDmoMagic, 0x120CED89, VSTPluginLib::catDMO, false, false },
		{ DMO::WavesReverb::Create,	"{87FC0268-9A55-4360-95AA-004A1D9DE26C}", "WavesReverb",	kDmoMagic, 0x87FC0268, VSTPluginLib::catDMO, false, false },
		// DigiBooster Pro Echo DSP
		{ DigiBoosterEcho::Create, "", "DigiBooster Pro Echo", MAGIC4LE('D','B','M','0'), MAGIC4LE('E','c','h','o'), VSTPluginLib::catRoomFx, false, true },
		// LFO
		{ LFOPlugin::Create, "", "LFO", MAGIC4LE('O','M','P','T'), MAGIC4LE('L','F','O',' '), VSTPluginLib::catGenerator, false, true },
#ifdef MODPLUG_TRACKER
		{ MidiInOut::Create, "", "MIDI Input Output", PLUGMAGIC('V','s','t','P'), PLUGMAGIC('M','M','I','D'), VSTPluginLib::catSynth, true, true },
#endif // MODPLUG_TRACKER
	};

	pluginList.reserve(mpt::size(BuiltInPlugins));
	for(const auto &plugin : BuiltInPlugins)
	{
		VSTPluginLib *plug = new (std::nothrow) VSTPluginLib(plugin.createProc, true, mpt::PathString::FromUTF8(plugin.filename), mpt::PathString::FromUTF8(plugin.name));
		if(plug != nullptr)
		{
			pluginList.push_back(plug);
			plug->pluginId1 = plugin.pluginId1;
			plug->pluginId2 = plugin.pluginId2;
			plug->category = plugin.category;
			plug->isInstrument = plugin.isInstrument;
#ifdef MODPLUG_TRACKER
			if(plugin.isOurs)
				plug->vendor = _T("OpenMPT Project");
#endif // MODPLUG_TRACKER
		}
	}

#ifdef MODPLUG_TRACKER
	// For security reasons, we do not load untrusted DMO plugins in libopenmpt.
	EnumerateDirectXDMOs();
#endif
}


CVstPluginManager::~CVstPluginManager()
{
	for(auto &plug : pluginList)
	{
		while(plug->pPluginsList != nullptr)
		{
			plug->pPluginsList->Release();
		}
		delete plug;
	}
	#ifndef NO_DMO
		if(MustUnInitilizeCOM)
		{
			CoUninitialize();
			MustUnInitilizeCOM = false;
		}
	#endif
}


bool CVstPluginManager::IsValidPlugin(const VSTPluginLib *pLib) const
{
	return std::find(pluginList.begin(), pluginList.end(), pLib) != pluginList.end();
}


void CVstPluginManager::EnumerateDirectXDMOs()
{
#ifndef NO_DMO
	const mpt::UUID knownDMOs[] =
	{
		MPT_UUID(745057C7,F353,4F2D,A7EE,58434477730E), // AEC (Acoustic echo cancellation, not usable)
		MPT_UUID(EFE6629C,81F7,4281,BD91,C9D604A95AF6), // Chorus
		MPT_UUID(EF011F79,4000,406D,87AF,BFFB3FC39D57), // Compressor
		MPT_UUID(EF114C90,CD1D,484E,96E5,09CFAF912A21), // Distortion
		MPT_UUID(EF3E932C,D40B,4F51,8CCF,3F98F1B29D5D), // Echo
		MPT_UUID(EFCA3D92,DFD8,4672,A603,7420894BAD98), // Flanger
		MPT_UUID(DAFD8210,5711,4B91,9FE3,F75B7AE279BF), // Gargle
		MPT_UUID(EF985E71,D5C7,42D4,BA4D,2D073E2E96F4), // I3DL2Reverb
		MPT_UUID(120CED89,3BF4,4173,A132,3CB406CF3231), // ParamEq
		MPT_UUID(87FC0268,9A55,4360,95AA,004A1D9DE26C), // WavesReverb
		MPT_UUID(F447B69E,1884,4A7E,8055,346F74D6EDB3), // Resampler DMO (not usable)
	};

	HKEY hkEnum;
	WCHAR keyname[128];

	LONG cr = RegOpenKeyEx(HKEY_LOCAL_MACHINE, _T("software\\classes\\DirectShow\\MediaObjects\\Categories\\f3602b3f-0592-48df-a4cd-674721e7ebeb"), 0, KEY_READ, &hkEnum);
	DWORD index = 0;
	while (cr == ERROR_SUCCESS)
	{
		if ((cr = RegEnumKeyW(hkEnum, index, keyname, CountOf(keyname))) == ERROR_SUCCESS)
		{
			CLSID clsid;
			std::wstring formattedKey = std::wstring(L"{") + std::wstring(keyname) + std::wstring(L"}");
			if(Util::VerifyStringToCLSID(formattedKey, clsid))
			{
				if(std::find(std::begin(knownDMOs), std::end(knownDMOs), clsid) == std::end(knownDMOs))
				{
					HKEY hksub;
					formattedKey = std::wstring(L"software\\classes\\DirectShow\\MediaObjects\\") + std::wstring(keyname);
					if (RegOpenKeyW(HKEY_LOCAL_MACHINE, formattedKey.c_str(), &hksub) == ERROR_SUCCESS)
					{
						WCHAR name[64];
						DWORD datatype = REG_SZ;
						DWORD datasize = sizeof(name);

						if(ERROR_SUCCESS == RegQueryValueExW(hksub, nullptr, 0, &datatype, (LPBYTE)name, &datasize))
						{
							mpt::String::SetNullTerminator(name);

							VSTPluginLib *plug = new (std::nothrow) VSTPluginLib(DMOPlugin::Create, true, mpt::PathString::FromNative(Util::GUIDToString(clsid)), mpt::PathString::FromNative(name));
							if(plug != nullptr)
							{
								try
								{
									pluginList.push_back(plug);
									plug->pluginId1 = kDmoMagic;
									plug->pluginId2 = clsid.Data1;
									plug->category = VSTPluginLib::catDMO;
								} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
								{
									MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
									delete plug;
								}
#ifdef DMO_LOG
								Log(mpt::format(L"Found \"%1\" clsid=%2\n")(plug->libraryName, plug->dllPath));
#endif
							}
						}
						RegCloseKey(hksub);
					}
				}
			}
		}
		index++;
	}
	if (hkEnum) RegCloseKey(hkEnum);
#endif // NO_DMO
}


// Extract instrument and category information from plugin.
#ifndef NO_VST
static void GetPluginInformation(AEffect *effect, VSTPluginLib &library)
{
	unsigned long exception = 0;
	library.category = static_cast<VSTPluginLib::PluginCategory>(CVstPlugin::DispatchSEH(effect, effGetPlugCategory, 0, 0, nullptr, 0, exception));
	library.isInstrument = ((effect->flags & effFlagsIsSynth) || !effect->numInputs);

	if(library.isInstrument)
	{
		library.category = VSTPluginLib::catSynth;
	} else if(library.category >= VSTPluginLib::numCategories)
	{
		library.category = VSTPluginLib::catUnknown;
	}

#ifdef MODPLUG_TRACKER
	std::vector<char> s(256, 0);
	CVstPlugin::DispatchSEH(effect, effGetVendorString, 0, 0, s.data(), 0, exception);
	library.vendor = mpt::ToCString(mpt::CharsetLocale, s.data());
#endif // MODPLUG_TRACKER
}
#endif // NO_VST


#ifdef MODPLUG_TRACKER
// Add a plugin to the list of known plugins.
VSTPluginLib *CVstPluginManager::AddPlugin(const mpt::PathString &dllPath, const mpt::ustring &tags, bool fromCache, const bool checkFileExistence, std::wstring *const errStr)
{
	const mpt::PathString fileName = dllPath.GetFileName();

	// Check if this is already a known plugin.
	for(const auto &dupePlug : pluginList)
	{
		if(!dllPath.CompareNoCase(dllPath, dupePlug->dllPath)) return dupePlug;
	}

	if(checkFileExistence && errStr != nullptr && !dllPath.IsFile())
	{
		*errStr += L"\nUnable to find " + dllPath.ToWide();
	}

	// Look if the plugin info is stored in the PluginCache
	if(fromCache)
	{
		SettingsContainer & cacheFile = theApp.GetPluginCache();
		// First try finding the full path
		mpt::ustring IDs = cacheFile.Read<mpt::ustring>(cacheSection, dllPath.ToUnicode(), MPT_USTRING(""));
		if(IDs.length() < 16)
		{
			// If that didn't work out, find relative path
			mpt::PathString relPath = theApp.AbsolutePathToRelative(dllPath);
			IDs = cacheFile.Read<mpt::ustring>(cacheSection, relPath.ToUnicode(), MPT_USTRING(""));
		}

		if(IDs.length() >= 16)
		{
			VSTPluginLib *plug = new (std::nothrow) VSTPluginLib(nullptr, false, dllPath, fileName, tags);
			if(plug == nullptr)
			{
				return nullptr;
			}
			pluginList.push_back(plug);

			// Extract plugin IDs
			for (int i = 0; i < 16; i++)
			{
				int32 n = IDs[i] - '0';
				if (n > 9) n = IDs[i] + 10 - 'A';
				n &= 0x0f;
				if (i < 8)
				{
					plug->pluginId1 = (plug->pluginId1 << 4) | n;
				} else
				{
					plug->pluginId2 = (plug->pluginId2 << 4) | n;
				}
			}

			const mpt::ustring flagKey = IDs + MPT_USTRING(".Flags");
			plug->DecodeCacheFlags(cacheFile.Read<int32>(cacheSection, flagKey, 0));
			plug->vendor = cacheFile.Read<CString>(cacheSection, IDs + MPT_USTRING(".Vendor"), CString());

#ifdef VST_LOG
			Log("Plugin \"%s\" found in PluginCache\n", plug->libraryName.ToLocale().c_str());
#endif // VST_LOG
			return plug;
		} else
		{
#ifdef VST_LOG
			Log("Plugin \"%s\" mismatch in PluginCache: \"%s\" [%s]=\"%s\"\n", s, dllPath, (LPCTSTR)IDs, (LPCTSTR)strFullPath);
#endif // VST_LOG
		}
	}

	// If this key contains a file name on program launch, a plugin previously crashed OpenMPT.
	theApp.GetSettings().Write<mpt::PathString>(MPT_USTRING("VST Plugins"), MPT_USTRING("FailedPlugin"), dllPath, SettingWriteThrough);

	bool validPlug = false;

	VSTPluginLib *plug = new (std::nothrow) VSTPluginLib(nullptr, false, dllPath, fileName, tags);
	if(plug == nullptr)
	{
		return nullptr;
	}

#ifndef NO_VST
	unsigned long exception = 0;
	// Always scan plugins in a separate process
	HINSTANCE hLib = NULL;
	AEffect *pEffect = CVstPlugin::LoadPlugin(*plug, hLib, true);

	if(pEffect != nullptr && pEffect->magic == kEffectMagic && pEffect->dispatcher != nullptr)
	{
		CVstPlugin::DispatchSEH(pEffect, effOpen, 0, 0, 0, 0, exception);

		plug->pluginId1 = pEffect->magic;
		plug->pluginId2 = pEffect->uniqueID;

		GetPluginInformation(pEffect, *plug);

#ifdef VST_LOG
		int nver = CVstPlugin::DispatchSEH(pEffect, effGetVstVersion, 0,0, nullptr, 0, exception);
		if (!nver) nver = pEffect->version;
		Log("%-20s: v%d.0, %d in, %d out, %2d programs, %2d params, flags=0x%04X realQ=%d offQ=%d\n",
			plug->libraryName.ToLocale().c_str(), nver,
			pEffect->numInputs, pEffect->numOutputs,
			pEffect->numPrograms, pEffect->numParams,
			pEffect->flags, pEffect->realQualities, pEffect->offQualities);
#endif // VST_LOG

		CVstPlugin::DispatchSEH(pEffect, effClose, 0, 0, 0, 0, exception);

		validPlug = true;
	}

	FreeLibrary(hLib);
	if(exception != 0)
	{
		CVstPluginManager::ReportPlugException(mpt::format(L"Exception %1 while trying to load plugin \"%2\"!\n")(mpt::wfmt::HEX<8>(exception), plug->libraryName));
	}
#endif // NO_VST

	// Now it should be safe to assume that this plugin loaded properly. :)
	theApp.GetSettings().Remove(MPT_USTRING("VST Plugins"), MPT_USTRING("FailedPlugin"));

	// If OK, write the information in PluginCache
	if(validPlug)
	{
		pluginList.push_back(plug);
		plug->WriteToCache();
	} else
	{
		delete plug;
	}

	return (validPlug ? plug : nullptr);
}


// Remove a plugin from the list of known plugins and release any remaining instances of it.
bool CVstPluginManager::RemovePlugin(VSTPluginLib *pFactory)
{
	for(const_iterator p = begin(); p != end(); p++)
	{
		VSTPluginLib *plug = *p;
		if(plug == pFactory)
		{
			// Kill all instances of this plugin
			CriticalSection cs;

			while(plug->pPluginsList != nullptr)
			{
				plug->pPluginsList->Release();
			}
			pluginList.erase(p);
			delete plug;
			return true;
		}
	}
	return false;
}
#endif // MODPLUG_TRACKER


// Create an instance of a plugin.
bool CVstPluginManager::CreateMixPlugin(SNDMIXPLUGIN &mixPlugin, CSoundFile &sndFile)
{
	VSTPluginLib *pFound = nullptr;
#ifdef MODPLUG_TRACKER
	mixPlugin.SetAutoSuspend(TrackerSettings::Instance().enableAutoSuspend);
#endif // MODPLUG_TRACKER

	// Find plugin in library
	int8 match = 0;	// "Match quality" of found plugin. Higher value = better match.
#if MPT_OS_WINDOWS && !MPT_OS_WINDOWS_WINRT
	const mpt::PathString libraryName = mpt::PathString::FromUTF8(mixPlugin.GetLibraryName());
#else
	const std::string libraryName = mpt::ToLowerCaseAscii(mixPlugin.GetLibraryName());
#endif
	for(const auto &plug : pluginList)
	{
		const bool matchID = (plug->pluginId1 == mixPlugin.Info.dwPluginId1)
			&& (plug->pluginId2 == mixPlugin.Info.dwPluginId2);
#if MPT_OS_WINDOWS && !MPT_OS_WINDOWS_WINRT
		const bool matchName = !mpt::PathString::CompareNoCase(plug->libraryName, libraryName);
#else
		const bool matchName = (mpt::ToLowerCaseAscii(plug->libraryName.ToUTF8()) == libraryName);
#endif

		if(matchID && matchName)
		{
			pFound = plug;
			if(plug->IsNative(false))
			{
				break;
			}
			// If the plugin isn't native, first check if a native version can be found.
			match = 3;
		} else if(matchID && match < 2)
		{
			pFound = plug;
			match = 2;
		} else if(matchName && match < 1)
		{
			pFound = plug;
			match = 1;
		}
	}

	if(pFound != nullptr && pFound->Create != nullptr)
	{
		IMixPlugin *plugin = pFound->Create(*pFound, sndFile, &mixPlugin);
		return plugin != nullptr;
	}

#ifdef MODPLUG_TRACKER
	if(!pFound && strcmp(mixPlugin.GetLibraryName(), ""))
	{
		// Try finding the plugin DLL in the plugin directory or plugin cache instead.
		mpt::PathString fullPath = TrackerSettings::Instance().PathPlugins.GetDefaultDir();
		if(fullPath.empty())
		{
			fullPath = theApp.GetAppDirPath() + MPT_PATHSTRING("Plugins\\");
		}
		fullPath += mpt::PathString::FromUTF8(mixPlugin.GetLibraryName()) + MPT_PATHSTRING(".dll");

		pFound = AddPlugin(fullPath);
		if(!pFound)
		{
			// Try plugin cache (search for library name)
			SettingsContainer &cacheFile = theApp.GetPluginCache();
			mpt::ustring IDs = cacheFile.Read<mpt::ustring>(cacheSection, mpt::ToUnicode(mpt::CharsetUTF8, mixPlugin.GetLibraryName()), MPT_USTRING(""));
			if(IDs.length() >= 16)
			{
				fullPath = cacheFile.Read<mpt::PathString>(cacheSection, IDs, MPT_PATHSTRING(""));
				if(!fullPath.empty())
				{
					fullPath = theApp.RelativePathToAbsolute(fullPath);
					if(fullPath.IsFile())
					{
						pFound = AddPlugin(fullPath);
					}
				}
			}
		}
	}

#ifndef NO_VST
	if(pFound && mixPlugin.Info.dwPluginId1 == kEffectMagic)
	{
		AEffect *pEffect = nullptr;
		HINSTANCE hLibrary = nullptr;
		bool validPlugin = false;

		pEffect = CVstPlugin::LoadPlugin(*pFound, hLibrary, TrackerSettings::Instance().bridgeAllPlugins);

		if(pEffect != nullptr && pEffect->dispatcher != nullptr && pEffect->magic == kEffectMagic)
		{
			validPlugin = true;

			GetPluginInformation(pEffect, *pFound);

			// Update cached information
			pFound->WriteToCache();

			CVstPlugin *pVstPlug = new (std::nothrow) CVstPlugin(hLibrary, *pFound, mixPlugin, *pEffect, sndFile);
			if(pVstPlug == nullptr)
			{
				validPlugin = false;
			}
		}

		if(!validPlugin)
		{
			FreeLibrary(hLibrary);
			CVstPluginManager::ReportPlugException(mpt::format(L"Unable to create plugin \"%1\"!\n")(pFound->libraryName));
		}
		return validPlugin;
	} else
	{
		// "plug not found" notification code MOVED to CSoundFile::Create
#ifdef VST_LOG
		Log("Unknown plugin\n");
#endif
	}
#endif // NO_VST

#endif // MODPLUG_TRACKER
	return false;
}


#ifdef MODPLUG_TRACKER
void CVstPluginManager::OnIdle()
{
	for(auto &factory : pluginList)
	{
		// Note: bridged plugins won't receive these messages and generate their own idle messages.
		IMixPlugin *p = factory->pPluginsList;
		while (p)
		{
			//rewbs. VSTCompliance: A specific plug has requested indefinite periodic processing time.
			p->Idle();
			//We need to update all open editors
			CAbstractVstEditor *editor = p->GetEditor();
			if (editor && editor->m_hWnd)
			{
				editor->UpdateParamDisplays();
			}
			//end rewbs. VSTCompliance:

			p = p->GetNextInstance();
		}
	}
}


void CVstPluginManager::ReportPlugException(const std::string &msg)
{
	Reporting::Notification(msg.c_str());
#ifdef VST_LOG
	Log("%s", msg.c_str());
#endif
}

void CVstPluginManager::ReportPlugException(const std::wstring &msg)
{
	Reporting::Notification(msg);
#ifdef VST_LOG
	Log(mpt::ToUnicode(msg));
#endif
}
#endif // MODPLUG_TRACKER

OPENMPT_NAMESPACE_END

#endif // NO_PLUGINS
