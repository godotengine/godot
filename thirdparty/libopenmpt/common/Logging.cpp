/*
 * Logging.cpp
 * -----------
 * Purpose: General logging
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#include "Logging.h"
#include "mptFileIO.h"
#if defined(MODPLUG_TRACKER)
#include <atomic>
#endif
#include "version.h"

#include <iostream>

#include <cstdarg>
#include <cstring>

#include <stdarg.h>


OPENMPT_NAMESPACE_BEGIN


namespace mpt
{
namespace log
{
	

#ifndef NO_LOGGING



#if !defined(MPT_LOG_GLOBAL_LEVEL_STATIC)
#if defined(MPT_LOG_GLOBAL_LEVEL)
int GlobalLogLevel = static_cast<int>(MPT_LOG_GLOBAL_LEVEL);
#else
int GlobalLogLevel = static_cast<int>(LogDebug);
#endif
#endif



#if defined(MODPLUG_TRACKER) && !defined(MPT_LOG_IS_DISABLED)

bool FileEnabled = false;
bool DebuggerEnabled = true;
bool ConsoleEnabled = false;

static char g_FacilitySolo[1024] = {0};
static char g_FacilityBlocked[1024] = {0};

void SetFacilities(const std::string &solo, const std::string &blocked)
{
	std::strcpy(g_FacilitySolo, solo.c_str());
	std::strcpy(g_FacilityBlocked, blocked.c_str());
}

bool IsFacilityActive(const char *facility)
{
	if(facility)
	{
		if(std::strlen(g_FacilitySolo) > 0)
		{
			if(std::strcmp(facility, g_FacilitySolo) != 0)
			{
				return false;
			}
		}
		if(std::strlen(g_FacilityBlocked) > 0)
		{
			if(std::strcmp(facility, g_FacilitySolo) == 0)
			{
				return false;
			}
		}
	}
	return true;
}

#endif


void Logger::SendLogMessage(const Context &context, LogLevel level, const char *facility, const mpt::ustring &text)
{
#ifdef MPT_LOG_IS_DISABLED
	MPT_UNREFERENCED_PARAMETER(context);
	MPT_UNREFERENCED_PARAMETER(level);
	MPT_UNREFERENCED_PARAMETER(facility);
	MPT_UNREFERENCED_PARAMETER(text);
#else // !MPT_LOG_IS_DISABLED
	MPT_MAYBE_CONSTANT_IF(mpt::log::GlobalLogLevel < level)
	{
		return;
	}
	#if defined(MODPLUG_TRACKER)
		if(!IsFacilityActive(facility))
		{
			return;
		}
	#else // !MODPLUG_TRACKER
		MPT_UNREFERENCED_PARAMETER(facility);
	#endif // MODPLUG_TRACKER
	// remove eol if already present and add log level prefix
	const mpt::ustring message = LogLevelToString(level) + MPT_USTRING(": ") + mpt::String::RTrim(text, MPT_USTRING("\r\n"));
	const mpt::ustring file = mpt::ToUnicode(mpt::CharsetASCII, context.file);
	const mpt::ustring function = mpt::ToUnicode(mpt::CharsetASCII, context.function);
	const mpt::ustring line = mpt::ufmt::dec(context.line);
	#if defined(MODPLUG_TRACKER) && !defined(MPT_BUILD_WINESUPPORT)
#if MPT_OS_WINDOWS
		static uint64 s_lastlogtime = 0;
		uint64 cur = mpt::Date::ANSI::Now();
		uint64 diff = cur/10000 - s_lastlogtime;
		s_lastlogtime = cur/10000;
#else
		uint64 cur = 0;
		uint64 diff = 0;
#endif
		if(mpt::log::FileEnabled)
		{
			static FILE * s_logfile = nullptr;
			if(!s_logfile)
			{
				s_logfile = mpt_fopen(MPT_PATHSTRING("mptrack.log"), "a");
			}
			if(s_logfile)
			{
				fprintf(s_logfile, mpt::ToCharset(mpt::CharsetUTF8, mpt::format(MPT_USTRING("%1+%2 %3(%4): %5 [%6]\n"))
					( mpt::Date::ANSI::ToString(cur)
					, mpt::ufmt::dec<6>(diff)
					, file
					, line
					, message
					, function
					)).c_str());
				fflush(s_logfile);
			}
		}
		if(mpt::log::DebuggerEnabled)
		{
			OutputDebugStringW(mpt::ToWide(mpt::format(MPT_USTRING("%1(%2): +%3 %4 [%5]\n"))
				( file
				, line
				, mpt::ufmt::dec<6>(diff)
				, message
				, function
				)).c_str());
		}
		if(mpt::log::ConsoleEnabled)
		{
			static bool consoleInited = false;
			if(!consoleInited)
			{
				AllocConsole();
				consoleInited = true;
			}
			std::wstring consoletext = mpt::ToWide(message) + L"\r\n";
			DWORD dummy = 0;
			WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE), consoletext.c_str(), consoletext.length(), &dummy, NULL);
		}
	#elif defined(MODPLUG_TRACKER) && defined(MPT_BUILD_WINESUPPORT)
		std::clog
			<< "NativeSupport: "
			<< mpt::ToCharset(mpt::CharsetLocaleOrUTF8, file) << "(" << mpt::ToCharset(mpt::CharsetLocaleOrUTF8, line) << ")" << ": "
			<< mpt::ToCharset(mpt::CharsetLocaleOrUTF8, message)
			<< " [" << mpt::ToCharset(mpt::CharsetLocaleOrUTF8, function) << "]"
			<< std::endl;
	#else // !MODPLUG_TRACKER
		std::clog
			<< "libopenmpt: "
			<< mpt::ToCharset(mpt::CharsetLocaleOrUTF8, file) << "(" << mpt::ToCharset(mpt::CharsetLocaleOrUTF8, line) << ")" << ": "
			<< mpt::ToCharset(mpt::CharsetLocaleOrUTF8, message)
			<< " [" << mpt::ToCharset(mpt::CharsetLocaleOrUTF8, function) << "]"
			<< std::endl;
	#endif // MODPLUG_TRACKER
#endif // MPT_LOG_IS_DISABLED
}

void LegacyLogger::operator () (const AnyStringLocale &text)
{
	SendLogMessage(context, MPT_LEGACY_LOGLEVEL, "", text);
}

void LegacyLogger::operator () (const char *format, ...)
{
	static const std::size_t LOGBUF_SIZE = 1024;
	char message[LOGBUF_SIZE];
	va_list va;
	va_start(va, format);
	vsnprintf(message, LOGBUF_SIZE, format, va);
	va_end(va);
	message[LOGBUF_SIZE - 1] = '\0';
	SendLogMessage(context, MPT_LEGACY_LOGLEVEL, "", mpt::ToUnicode(mpt::CharsetLocaleOrUTF8, message));
}

void LegacyLogger::operator () (LogLevel level, const mpt::ustring &text)
{
	SendLogMessage(context, level, "", text);
}



#endif // !NO_LOGGING



#if defined(MODPLUG_TRACKER)

namespace Trace {

#if MPT_OS_WINDOWS

// Debugging functionality will use simple globals.

bool volatile g_Enabled = false;

static bool g_Sealed = false;

struct Entry {
	uint32       Index;
	uint32       ThreadId;
	uint64       Timestamp;
	const char * Function;
	const char * File;
	int          Line;
};

inline bool operator < (const Entry &a, const Entry &b)
{
/*
	return false
		|| (a.Timestamp < b.Timestamp)
		|| (a.ThreadID < b.ThreadID)
		|| (a.File < b.File)
		|| (a.Line < b.Line)
		|| (a.Function < b.Function)
		;
*/
	return false
		|| (a.Index < b.Index)
		;
}

static std::vector<mpt::log::Trace::Entry> Entries;

static std::atomic<uint32> NextIndex(0);

static uint32 ThreadIdGUI = 0;
static uint32 ThreadIdAudio = 0;
static uint32 ThreadIdNotify = 0;

void Enable(std::size_t numEntries)
{
	if(g_Sealed)
	{
		return;
	}
	Entries.clear();
	Entries.resize(numEntries);
	NextIndex.store(0);
	g_Enabled = true;
}

void Disable()
{
	if(g_Sealed)
	{
		return;
	}
	g_Enabled = false;
}

MPT_NOINLINE void Trace(const mpt::log::Context & context)
{
	// This will get called in realtime contexts and hot paths.
	// No blocking allowed here.
	const uint32 index = NextIndex.fetch_add(1);
#if 1
	LARGE_INTEGER time;
	time.QuadPart = 0;
	QueryPerformanceCounter(&time);
	const uint64 timestamp = time.QuadPart;
#else
	FILETIME time = FILETIME();
	GetSystemTimeAsFileTime(&time);
	const uint64 timestamp = (static_cast<uint64>(time.dwHighDateTime) << 32) | (static_cast<uint64>(time.dwLowDateTime) << 0);
#endif
	const uint32 threadid = static_cast<uint32>(GetCurrentThreadId());
	mpt::log::Trace::Entry & entry = Entries[index % Entries.size()];
	entry.Index = index;
	entry.ThreadId = threadid;
	entry.Timestamp = timestamp;
	entry.Function = context.function;
	entry.File = context.file;
	entry.Line = context.line;
}

void Seal()
{
	if(!g_Enabled)
	{
		return;
	}
	g_Enabled = false;
	g_Sealed = true;
	uint32 count = NextIndex.fetch_add(0);
	if(count < Entries.size())
	{
		Entries.resize(count);
	}
}

bool Dump(const mpt::PathString &filename)
{
	if(!g_Sealed)
	{
		return false;
	}

	LARGE_INTEGER qpcNow;
	qpcNow.QuadPart = 0;
	QueryPerformanceCounter(&qpcNow);
	uint64 ftNow = mpt::Date::ANSI::Now();

	// sort according to index in case of overflows
	std::stable_sort(Entries.begin(), Entries.end());

	mpt::ofstream f(filename, std::ios::out);

	f << "Build: OpenMPT " << MptVersion::GetVersionStringExtended() << std::endl;

	bool qpcValid = false;

	LARGE_INTEGER qpcFreq;
	qpcFreq.QuadPart = 0;
	QueryPerformanceFrequency(&qpcFreq);
	if(qpcFreq.QuadPart > 0)
	{
		qpcValid = true;
	}

	f << "Dump: " << mpt::ToCharset(mpt::CharsetUTF8, mpt::Date::ANSI::ToString(ftNow)) << std::endl;
	f << "Captured events: " << Entries.size() << std::endl;
	if(qpcValid && (Entries.size() > 0))
	{
		double period = static_cast<double>(Entries[Entries.size() - 1].Timestamp - Entries[0].Timestamp) / static_cast<double>(qpcFreq.QuadPart);
		double eventsPerSecond = Entries.size() / period;
		f << "Period [s]: " << mpt::fmt::fix(period) << std::endl;
		f << "Events/second: " << mpt::fmt::fix(eventsPerSecond) << std::endl;
	}

	for(std::size_t i = 0; i < Entries.size(); ++i)
	{
		mpt::log::Trace::Entry & entry = Entries[i];
		if(!entry.Function) entry.Function = "";
		if(!entry.File) entry.File = "";
		std::string time;
		if(qpcValid)
		{
			time = mpt::ToCharset(mpt::CharsetUTF8, mpt::Date::ANSI::ToString( ftNow - static_cast<int64>( static_cast<double>(qpcNow.QuadPart - entry.Timestamp) * (10000000.0 / static_cast<double>(qpcFreq.QuadPart) ) ) ) );
		} else
		{
			time = mpt::format("0x%1")(mpt::fmt::hex0<16>(entry.Timestamp));
		}
		f << time;
		if(entry.ThreadId == ThreadIdGUI)
		{
			f << " -----GUI ";
		} else if(entry.ThreadId == ThreadIdAudio)
		{
			f << " ---Audio ";
		} else if(entry.ThreadId == ThreadIdNotify)
		{
			f << " --Notify ";
		} else
		{
			f << " " << mpt::fmt::hex0<8>(entry.ThreadId) << " ";
		}
		f << entry.File << "(" << entry.Line << "): " << entry.Function;
		f << std::endl;
	}
	return true;
}

void SetThreadId(mpt::log::Trace::ThreadKind kind, uint32 id)
{
	if(id == 0)
	{
		return;
	}
	switch(kind)
	{
		case ThreadKindGUI:
			ThreadIdGUI = id;
			break;
		case ThreadKindAudio:
			ThreadIdAudio = id;
			break;
		case ThreadKindNotify:
			ThreadIdNotify = id;
			break;
	}
}

#endif // MPT_OS_WINDOWS

} // namespace Trace

#endif // MODPLUG_TRACKER


} // namespace log
} // namespace mpt


OPENMPT_NAMESPACE_END
