/*
 * mptOS.cpp
 * ---------
 * Purpose: Operating system version information.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "mptOS.h"

#if MPT_OS_WINDOWS
#include <windows.h>
#endif


OPENMPT_NAMESPACE_BEGIN


namespace mpt
{
namespace Windows
{


#if MPT_OS_WINDOWS


#if !MPT_OS_WINDOWS_WINRT

static uint32 VersionDecimalTo_WIN32_WINNT(uint32 major, uint32 minor)
{
	// GetVersionEx returns decimal.
	// _WIN32_WINNT macro uses BCD for the minor byte (see Windows 98 / ME).
	// We use what _WIN32_WINNT does.
	uint32 result = 0;
	minor = mpt::clamp<uint32>(minor, 0, 99);
	result |= major;
	result <<= 8;
	result |= minor/10*0x10 + minor%10;
	return result;
}

#endif // !MPT_OS_WINDOWS_WINRT


static void GatherWindowsVersion(uint32 & SystemVersion)
{
	// Initialize to used SDK version
	SystemVersion =
		#if NTDDI_VERSION >= 0x0A000000 // NTDDI_WIN10
			mpt::Windows::Version::Win10
		#elif NTDDI_VERSION >= 0x06030000 // NTDDI_WINBLUE
			mpt::Windows::Version::Win81
		#elif NTDDI_VERSION >= 0x06020000 // NTDDI_WIN8
			mpt::Windows::Version::Win8
		#elif NTDDI_VERSION >= 0x06010000 // NTDDI_WIN7
			mpt::Windows::Version::Win7
		#elif NTDDI_VERSION >= 0x06000000 // NTDDI_VISTA
			mpt::Windows::Version::WinVista
		#elif NTDDI_VERSION >= 0x05020000 // NTDDI_WS03
			mpt::Windows::Version::WinXP64
		#elif NTDDI_VERSION >= NTDDI_WINXP
			mpt::Windows::Version::WinXP
		#elif NTDDI_VERSION >= NTDDI_WIN2K
			mpt::Windows::Version::Win2000
		#else
			mpt::Windows::Version::WinNT4
		#endif
		;
#if !MPT_OS_WINDOWS_WINRT
	OSVERSIONINFOEXW versioninfoex;
	MemsetZero(versioninfoex);
	versioninfoex.dwOSVersionInfoSize = sizeof(versioninfoex);
#if MPT_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable:4996) // 'GetVersionExW': was declared deprecated
#endif // MPT_COMPILER_MSVC
#if MPT_COMPILER_CLANG || MPT_COMPILER_MSVCCLANGC2
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif // MPT_COMPILER_CLANG
	GetVersionExW((LPOSVERSIONINFOW)&versioninfoex);
#if MPT_COMPILER_MSVC
#pragma warning(pop)
#endif // MPT_COMPILER_MSVC
#if MPT_COMPILER_CLANG || MPT_COMPILER_MSVCCLANGC2
#pragma clang diagnostic pop
#endif // MPT_COMPILER_CLANG
	SystemVersion = VersionDecimalTo_WIN32_WINNT(versioninfoex.dwMajorVersion, versioninfoex.dwMinorVersion);
#endif // !MPT_OS_WINDOWS_WINRT
}


#ifdef MODPLUG_TRACKER

namespace {
struct WindowsVersionCache
{
	uint32 SystemVersion;
	WindowsVersionCache()
		: SystemVersion(mpt::Windows::Version::WinNT4)
	{
		GatherWindowsVersion(SystemVersion);
	}
};
}

static void GatherWindowsVersionFromCache(uint32 & SystemVersion)
{
	static WindowsVersionCache gs_WindowsVersionCache;
	SystemVersion = gs_WindowsVersionCache.SystemVersion;
}

#endif // MODPLUG_TRACKER


#endif // MPT_OS_WINDOWS


Version::Version()
	: SystemIsWindows(false)
	, SystemVersion(mpt::Windows::Version::WinNT4)
{
	return;
}


mpt::Windows::Version Version::Current()
{
	mpt::Windows::Version result;
	#if MPT_OS_WINDOWS
		result.SystemIsWindows = true;
		#ifdef MODPLUG_TRACKER
			GatherWindowsVersionFromCache(result.SystemVersion);
		#else // !MODPLUG_TRACKER
			GatherWindowsVersion(result.SystemVersion);
		#endif // MODPLUG_TRACKER
	#endif // MPT_OS_WINDOWS
	return result;
}


bool Version::IsWindows() const
{
	return SystemIsWindows;
}


bool Version::IsBefore(mpt::Windows::Version::Number version) const
{
	if(!SystemIsWindows)
	{
		return false;
	}
	return (SystemVersion < static_cast<uint32>(version));
}


bool Version::IsAtLeast(mpt::Windows::Version::Number version) const
{
	if(!SystemIsWindows)
	{
		return false;
	}
	return (SystemVersion >= static_cast<uint32>(version));
}


static MPT_CONSTEXPR11_VAR struct { Version::Number version; const MPT_UCHAR_TYPE * name; } versionMap[] =
{
	{ mpt::Windows::Version::WinNewer, MPT_ULITERAL("Windows 10 (or newer)") },
	{ mpt::Windows::Version::Win10, MPT_ULITERAL("Windows 10") },
	{ mpt::Windows::Version::Win81, MPT_ULITERAL("Windows 8.1") },
	{ mpt::Windows::Version::Win8, MPT_ULITERAL("Windows 8") },
	{ mpt::Windows::Version::Win7, MPT_ULITERAL("Windows 7") },
	{ mpt::Windows::Version::WinVista, MPT_ULITERAL("Windows Vista") },
	{ mpt::Windows::Version::WinXP64, MPT_ULITERAL("Windows XP x64 / Windows Server 2003") },
	{ mpt::Windows::Version::WinXP, MPT_ULITERAL("Windows XP") },
	{ mpt::Windows::Version::Win2000, MPT_ULITERAL("Windows 2000") },
	{ mpt::Windows::Version::WinME, MPT_ULITERAL("Windows ME") },
	{ mpt::Windows::Version::Win98, MPT_ULITERAL("Windows 98") },
	{ mpt::Windows::Version::WinNT4, MPT_ULITERAL("Windows NT4") }
};


mpt::ustring Version::VersionToString(uint16 version)
{
	mpt::ustring result;
	for(const auto &v : versionMap)
	{
		if(version > v.version)
		{
			result = MPT_USTRING("> ") + v.name;
			break;
		} else if(version == v.version)
		{
			result = v.name;
			break;
		}
	}
	if(result.empty())
	{
		result = mpt::format(MPT_USTRING("0x%1"))(mpt::ufmt::dec0<4>(version));
	}
	return result;
}



mpt::ustring Version::VersionToString(Number version)
{
	return VersionToString(static_cast<uint16>(version));
}


mpt::ustring Version::GetName() const
{
	mpt::ustring name = MPT_USTRING("Generic Windows NT");
	for(const auto &v : versionMap)
	{
		if(mpt::Windows::Version::IsAtLeast(v.version))
		{
			name = v.name;
			break;
		}
	}
	mpt::ustring result = name;
	#if defined(MODPLUG_TRACKER) && MPT_OS_WINDOWS
		if(mpt::Windows::IsWine())
		{
			mpt::Wine::VersionContext v;
			if(v.Version().IsValid())
			{
				result = mpt::format(MPT_USTRING("Wine %1 (%2)"))(
					  v.Version().AsString()
					, name
					);
			} else
			{
				result = mpt::format(MPT_USTRING("Wine (unknown version: '%1') (%2)"))(
					  mpt::ToUnicode(mpt::CharsetUTF8, v.RawVersion())
					, name
					);
			}
		}
	#endif // MODPLUG_TRACKER && MPT_OS_WINDOWS
	return result;
}


#ifdef MODPLUG_TRACKER
mpt::ustring Version::GetNameShort() const
{
	mpt::ustring name;
	if(mpt::Windows::IsWine())
	{
		mpt::Wine::VersionContext v;
		if(v.Version().IsValid())
		{
			name = mpt::format(MPT_USTRING("wine-%1"))(v.Version().AsString());
		} else if(v.RawVersion().length() > 0)
		{
			name = MPT_USTRING("wine-") + Util::BinToHex(mpt::as_span(v.RawVersion()));
		} else
		{
			name = MPT_USTRING("wine-");
		}
		name += MPT_USTRING("-") + Util::BinToHex(mpt::as_span(v.RawHostSysName()));
	} else
	{
		name = mpt::format(MPT_USTRING("%1.%2"))(mpt::ufmt::dec(SystemVersion >> 8), mpt::ufmt::HEX0<2>(SystemVersion & 0xFF));
	}
	return name;
}
#endif // MODPLUG_TRACKER


mpt::Windows::Version::Number Version::GetMinimumKernelLevel()
{
	uint16 minimumKernelVersion = 0;
	#if MPT_OS_WINDOWS && MPT_COMPILER_MSVC
		#if !defined(MPT_BUILD_TARGET_XP)
			minimumKernelVersion = std::max<uint16>(minimumKernelVersion, mpt::Windows::Version::WinVista);
		#else
			minimumKernelVersion = std::max<uint16>(minimumKernelVersion, mpt::Windows::Version::WinXP);
		#endif
	#endif
	return static_cast<mpt::Windows::Version::Number>(minimumKernelVersion);
}


mpt::Windows::Version::Number Version::GetMinimumAPILevel()
{
	uint16 minimumApiVersion = 0;
	#if MPT_OS_WINDOWS && defined(_WIN32_WINNT)
		minimumApiVersion = std::max<uint16>(minimumApiVersion, _WIN32_WINNT);
	#endif
	return static_cast<mpt::Windows::Version::Number>(minimumApiVersion);
}


#if defined(MODPLUG_TRACKER)


#if MPT_OS_WINDOWS

static bool GatherSystemIsWine()
{
	bool SystemIsWine = false;
	HMODULE hNTDLL = LoadLibraryW(L"ntdll.dll");
	if(hNTDLL)
	{
		SystemIsWine = (GetProcAddress(hNTDLL, "wine_get_version") != NULL);
		FreeLibrary(hNTDLL);
		hNTDLL = NULL;
	}
	return SystemIsWine;
}

#endif // MPT_OS_WINDOWS

#if defined(MODPLUG_TRACKER) && MPT_OS_WINDOWS

namespace {
struct SystemIsWineCache
{
	bool SystemIsWine;
	SystemIsWineCache()
		: SystemIsWine(GatherSystemIsWine())
	{
		return;
	}
	SystemIsWineCache(bool isWine)
		: SystemIsWine(isWine)
	{
		return;
	}
};
}

#endif // MODPLUG_TRACKER && MPT_OS_WINDOWS

static bool SystemIsWine(bool allowDetection = true)
{
	#if defined(MODPLUG_TRACKER) && MPT_OS_WINDOWS
		static SystemIsWineCache gs_SystemIsWineCache = allowDetection ? SystemIsWineCache() : SystemIsWineCache(false);
		if(!allowDetection)
		{ // catch too late calls of PreventWineDetection
			MPT_ASSERT(!gs_SystemIsWineCache.SystemIsWine);
		}
		return gs_SystemIsWineCache.SystemIsWine;
	#elif MPT_OS_WINDOWS
		MPT_UNREFERENCED_PARAMETER(allowDetection);
		return GatherSystemIsWine();
	#else
		MPT_UNREFERENCED_PARAMETER(allowDetection);
		return false;
	#endif
}

void PreventWineDetection()
{
	SystemIsWine(false);
}

bool IsOriginal()
{
	return mpt::Windows::Version::Current().IsWindows() && !SystemIsWine();
}

bool IsWine()
{
	return mpt::Windows::Version::Current().IsWindows() && SystemIsWine();
}


#endif // MODPLUG_TRACKER


} // namespace Windows
} // namespace mpt



#if defined(MODPLUG_TRACKER)

namespace mpt
{
namespace Wine
{


Version::Version()
	: valid(false)
	, vmajor(0)
	, vminor(0)
	, vupdate(0)
{
	return;
}


Version::Version(const mpt::ustring &rawVersion)
	: valid(false)
	, vmajor(0)
	, vminor(0)
	, vupdate(0)
{
	if(rawVersion.empty())
	{
		return;
	}
	std::vector<uint8> version = mpt::String::Split<uint8>(rawVersion, MPT_USTRING("."));
	if(version.size() < 2)
	{
		return;
	}
	mpt::ustring parsedVersion = mpt::String::Combine(version, MPT_USTRING("."));
	std::size_t len = std::min(parsedVersion.length(), rawVersion.length());
	if(len == 0)
	{
		return;
	}
	if(parsedVersion.substr(0, len) != rawVersion.substr(0, len))
	{
		return;
	}
	valid = true;
	vmajor = version[0];
	vminor = version[1];
	vupdate = (version.size() >= 3) ? version[2] : 0;
}


Version::Version(uint8 vmajor, uint8 vminor, uint8 vupdate)
	: valid((vmajor > 0) || (vminor > 0) || (vupdate > 0)) 
	, vmajor(vmajor)
	, vminor(vminor)
	, vupdate(vupdate)
{
	return;
}


mpt::Wine::Version Version::FromInteger(uint32 version)
{
	mpt::Wine::Version result;
	result.valid = (version <= 0xffffff);
	result.vmajor = static_cast<uint8>(version >> 16);
	result.vminor = static_cast<uint8>(version >> 8);
	result.vupdate = static_cast<uint8>(version >> 0);
	return result;
}


bool Version::IsValid() const
{
	return valid;
}


mpt::ustring Version::AsString() const
{
	return mpt::ufmt::dec(vmajor) + MPT_USTRING(".") + mpt::ufmt::dec(vminor) + MPT_USTRING(".") + mpt::ufmt::dec(vupdate);
}


uint32 Version::AsInteger() const
{
	uint32 version = 0;
	version |= static_cast<uint32>(vmajor) << 16;
	version |= static_cast<uint32>(vminor) << 8;
	version |= static_cast<uint32>(vupdate) << 0;
	return version;
}


bool Version::IsBefore(mpt::Wine::Version other) const
{
	if(!IsValid())
	{
		return false;
	}
	return (AsInteger() < other.AsInteger());
}


bool Version::IsAtLeast(mpt::Wine::Version other) const
{
	if(!IsValid())
	{
		return false;
	}
	return (AsInteger() >= other.AsInteger());
}


mpt::Wine::Version GetMinimumWineVersion()
{
	mpt::Wine::Version minimumWineVersion = mpt::Wine::Version(0,0,0);
	#if MPT_OS_WINDOWS && MPT_COMPILER_MSVC
		#if !defined(MPT_BUILD_TARGET_XP)
			minimumWineVersion = mpt::Wine::Version(1,8,0);
		#else
			minimumWineVersion = mpt::Wine::Version(1,6,0);
		#endif
	#endif
	return minimumWineVersion;
}


VersionContext::VersionContext()
	: m_IsWine(false)
	, m_HostIsLinux(false)
	, m_HostIsBSD(false)
{
	#if MPT_OS_WINDOWS
		m_IsWine = mpt::Windows::IsWine();
		if(!m_IsWine)
		{
			return;
		}
		m_NTDLL = mpt::Library(mpt::LibraryPath::FullPath(MPT_PATHSTRING("ntdll.dll")));
		if(m_NTDLL.IsValid())
		{
			const char * (__cdecl * wine_get_version)(void) = nullptr;
			const char * (__cdecl * wine_get_build_id)(void) = nullptr;
			void (__cdecl * wine_get_host_version)(const char * *, const char * *) = nullptr;
			m_NTDLL.Bind(wine_get_version, "wine_get_version");
			m_NTDLL.Bind(wine_get_build_id, "wine_get_build_id");
			m_NTDLL.Bind(wine_get_host_version, "wine_get_host_version");
			const char * wine_version = nullptr;
			const char * wine_build_id = nullptr;
			const char * wine_host_sysname = nullptr;
			const char * wine_host_release = nullptr;
			wine_version = wine_get_version ? wine_get_version() : "";
			wine_build_id = wine_get_build_id ? wine_get_build_id() : "";
			if(wine_get_host_version)
			{
				wine_get_host_version(&wine_host_sysname, &wine_host_release);
			}
			m_RawVersion = wine_version ? wine_version : "";
			m_RawBuildID = wine_build_id ? wine_build_id : "";
			m_RawHostSysName = wine_host_sysname ? wine_host_sysname : "";
			m_RawHostRelease = wine_host_release ? wine_host_release : "";
		}
		m_Version = mpt::Wine::Version(mpt::ToUnicode(mpt::CharsetUTF8, m_RawVersion));
		m_HostIsLinux = (m_RawHostSysName == "Linux");
		m_HostIsBSD = (m_RawHostSysName == "FreeBSD" || m_RawHostSysName == "DragonFly" || m_RawHostSysName == "NetBSD" || m_RawHostSysName == "OpenBSD");
	#endif // MPT_OS_WINDOWS
}


} // namespace Wine
} // namespace mpt

#endif // MODPLUG_TRACKER



OPENMPT_NAMESPACE_END
