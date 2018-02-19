/*
 * mptOS.h
 * -------
 * Purpose: Operating system version information.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "mptLibrary.h"


OPENMPT_NAMESPACE_BEGIN


namespace mpt
{
namespace Windows
{

class Version
{

public:

	enum Number
	{

		WinNT4   = 0x0400,
		Win98    = 0x0410,
		WinME    = 0x0490,
		Win2000  = 0x0500,
		WinXP    = 0x0501,
		WinXP64  = 0x0502,
		WinVista = 0x0600,
		Win7     = 0x0601,
		Win8     = 0x0602,
		Win81    = 0x0603,
		Win10    = 0x0a00,

		WinNewer = Win10 + 1

	};

	static mpt::ustring VersionToString(uint16 version);
	static mpt::ustring VersionToString(Number version);

private:

	bool SystemIsWindows;

	uint32 SystemVersion;

private:

	Version();

public:

	static mpt::Windows::Version Current();

public:

	bool IsWindows() const;

	bool IsBefore(mpt::Windows::Version::Number version) const;
	bool IsAtLeast(mpt::Windows::Version::Number version) const;

	mpt::ustring GetName() const;
#ifdef MODPLUG_TRACKER
	mpt::ustring GetNameShort() const;
#endif // MODPLUG_TRACKER

public:

	static mpt::Windows::Version::Number GetMinimumKernelLevel();
	static mpt::Windows::Version::Number GetMinimumAPILevel();

}; // class Version

#if defined(MODPLUG_TRACKER)

void PreventWineDetection();

bool IsOriginal();
bool IsWine();

#endif // MODPLUG_TRACKER

} // namespace Windows
} // namespace mpt


#if defined(MODPLUG_TRACKER)

namespace mpt
{

namespace Wine
{

class Version
{
private:
	bool valid;
	uint8 vmajor;
	uint8 vminor;
	uint8 vupdate;
public:
	Version();
	Version(uint8 vmajor, uint8 vminor, uint8 vupdate);
	explicit Version(const mpt::ustring &version);
public:
	bool IsValid() const;
	mpt::ustring AsString() const;
private:
	static mpt::Wine::Version FromInteger(uint32 version);
	uint32 AsInteger() const;
public:
	bool IsBefore(mpt::Wine::Version other) const;
	bool IsAtLeast(mpt::Wine::Version other) const;
};

mpt::Wine::Version GetMinimumWineVersion();

class VersionContext
{
protected:
	bool m_IsWine;
	mpt::Library m_NTDLL;
	std::string m_RawVersion;
	std::string m_RawBuildID;
	std::string m_RawHostSysName;
	std::string m_RawHostRelease;
	mpt::Wine::Version m_Version;
	bool m_HostIsLinux;
	bool m_HostIsBSD;
public:
	VersionContext();
public:
	bool IsWine() const { return m_IsWine; }
	mpt::Library NTDLL() const { return m_NTDLL; }
	std::string RawVersion() const { return m_RawVersion; }
	std::string RawBuildID() const { return m_RawBuildID; }
	std::string RawHostSysName() const { return m_RawHostSysName; }
	std::string RawHostRelease() const { return m_RawHostRelease; }
	mpt::Wine::Version Version() const { return m_Version; }
	bool HostIsLinux() const { return m_HostIsLinux; }
	bool HostIsBSD() const { return m_HostIsBSD; }
};

} // namespace Wine

} // namespace mpt

#endif // MODPLUG_TRACKER


OPENMPT_NAMESPACE_END
