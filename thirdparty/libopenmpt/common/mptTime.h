/*
 * mptTime.h
 * ---------
 * Purpose: Various time utility functions.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <string>

#include <time.h>


OPENMPT_NAMESPACE_BEGIN



namespace mpt
{
namespace Date
{

#if defined(MODPLUG_TRACKER)

#if MPT_OS_WINDOWS

namespace ANSI
{
// uint64 counts 100ns since 1601-01-01T00:00Z

uint64 Now();

mpt::ustring ToString(uint64 time100ns); // i.e. 2015-01-15 18:32:01.718

} // namespacee ANSI

#endif // MPT_OS_WINDOWS

#endif // MODPLUG_TRACKER

class Unix
{
// int64 counts 1s since 1970-01-01T00:00Z
private:
	int64 Value;
public:
	Unix();
	explicit Unix(int64 unixtime);
	operator int64 () const;
public:
	static mpt::Date::Unix FromUTC(tm timeUtc);
	tm AsUTC() const;
};

mpt::ustring ToShortenedISO8601(tm date); // i.e. 2015-01-15T18:32:01Z

} // namespace Date
} // namespace mpt



#ifdef MODPLUG_TRACKER

namespace Util
{

#if MPT_OS_WINDOWS

// RAII wrapper around timeBeginPeriod/timeEndPeriod/timeGetTime (on Windows).
// This clock is monotonic, even across changing its resolution.
// This is needed to synchronize time in Steinberg APIs (ASIO and VST).
class MultimediaClock
{
private:
	uint32 m_CurrentPeriod;
private:
	void Init();
	void SetPeriod(uint32 ms);
	void Cleanup();
public:
	MultimediaClock();
	MultimediaClock(uint32 ms);
	~MultimediaClock();
public:
	// Sets the desired resolution in milliseconds, returns the obtained resolution in milliseconds.
	// A parameter of 0 causes the resolution to be reset to system defaults.
	// A return value of 0 means the resolution is unknown, but timestamps will still be valid.
	uint32 SetResolution(uint32 ms);
	// Returns obtained resolution in milliseconds.
	// A return value of 0 means the resolution is unknown, but timestamps will still be valid.
	uint32 GetResolution() const; 
	// Returns current instantaneous timestamp in milliseconds.
	// The epoch (offset) of the timestamps is undefined but constant until the next system reboot.
	// The resolution is the value returned from GetResolution().
	uint32 Now() const;
	// Returns current instantaneous timestamp in nanoseconds.
	// The epoch (offset) of the timestamps is undefined but constant until the next system reboot.
	// The resolution is the value returned from GetResolution() in milliseconds.
	uint64 NowNanoseconds() const;
};

#endif // MPT_OS_WINDOWS

} // namespace Util

#endif // MODPLUG_TRACKER



OPENMPT_NAMESPACE_END
