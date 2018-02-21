/*
 * Logging.h
 * ---------
 * Purpose: General logging
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once


OPENMPT_NAMESPACE_BEGIN


/*


Build time logging configuration (in BuildSettings.h):

 *  #define NO_LOGGING
    Disables all logging completely.
    MPT_LOG calls are not even compiled but instead completely removed via the
    preprocessor.

 *  #define MPT_LOG_GLOBAL_LEVEL_STATIC
    #define MPT_LOG_GLOBAL_LEVEL #
    Define the former (to anything) and the latter (to one of the log levels
    below) in order to statically select the verbosity of logging at build time.
    MPT_LOG calls that exceed the specified logging level will get dead-code
    eliminated at compile time.
    This especially means that, when setting MPT_LOG_GLOBAL_LEVEL to 0, no
    MPT_LOG call (with a constant level parameter) remains in the resulting
    binary, however, they still do get parsed and properly type checked by the
    compiler.


Logging:

If the context is related to a particular CSoundfile instance, use
CSoundfile::AddToLog.

Logging a simple message:
MPT_LOG(LogWarning, "sounddev", "some message");
MPT_LOG(LogWarning, "sounddev", MPT_USTRING("some message"));
Facility is some course grained code section identifier (more coarse grained
than the current file name probably), useful to do some selective logging.

Logging a more complex message:
MPT_LOG(LogWarning, "sounddev", mpt::format(MPT_USTRING("Some message: foo=%1, bar=0x%2"))(foo, mpt::ufmt::hex0<8>(bar)));

Note that even with full enabled logging and a runtime configurable logging
level, the runtime overhead of a MPT_LOG(level, facility, text) call is just a
single conditional in case the verbosity does not require logging the respective
message. Even the expression "text" is not evaluated.


*/


enum LogLevel
{
	LogDebug        = 5,
	LogInformation  = 4,
	LogNotification = 3,
	LogWarning      = 2,
	LogError        = 1
};


inline mpt::ustring LogLevelToString(LogLevel level)
{
	switch(level)
	{
	case LogError:        return MPT_USTRING("error");   break;
	case LogWarning:      return MPT_USTRING("warning"); break;
	case LogNotification: return MPT_USTRING("notify");  break;
	case LogInformation:  return MPT_USTRING("info");    break;
	case LogDebug:        return MPT_USTRING("debug");   break;
	}
	return MPT_USTRING("unknown");
}


class ILog
{
protected:
	virtual ~ILog() { }
public:
	virtual	void AddToLog(LogLevel level, const mpt::ustring &text) const = 0;
};



namespace mpt
{
namespace log
{



#ifndef NO_LOGGING


#if defined(MPT_LOG_GLOBAL_LEVEL_STATIC)
#if (MPT_LOG_GLOBAL_LEVEL <= 0)
// Logging framework is enabled (!NO_LOGGING) but all logging has beeen statically disabled.
// All logging code gets compiled and immediately dead-code eliminated.
#define MPT_LOG_IS_DISABLED
#endif
static const int GlobalLogLevel = MPT_LOG_GLOBAL_LEVEL ;
#else
extern int GlobalLogLevel;
#endif


#if defined(MODPLUG_TRACKER) && !defined(MPT_LOG_IS_DISABLED)
extern bool FileEnabled;
extern bool DebuggerEnabled;
extern bool ConsoleEnabled;
void SetFacilities(const std::string &solo, const std::string &blocked);
bool IsFacilityActive(const char *facility);
#else
static MPT_FORCEINLINE bool IsFacilityActive(const char * /*facility*/ ) { return true; }
#endif


#endif // !NO_LOGGING


struct Context
{
	const char * const file;
	const int line;
	const char * const function;
	MPT_FORCEINLINE Context(const char *file, int line, const char *function)
		: file(file)
		, line(line)
		, function(function)
	{
		return;
	}
	MPT_FORCEINLINE Context(const Context &c)
		: file(c.file)
		, line(c.line)
		, function(c.function)
	{
		return;
	}
}; // class Context

#define MPT_LOG_CURRENTCONTEXT() mpt::log::Context( __FILE__ , __LINE__ , __FUNCTION__ )


#ifndef NO_LOGGING


class Logger
{
public:
	// facility:ASCII
	void SendLogMessage(const Context &context, LogLevel level, const char *facility, const mpt::ustring &text);
public:
	// facility:ASCII, text:ASCII (only string literals)
	template <std::size_t size> MPT_FORCEINLINE void SendLogMessage(const Context &context, LogLevel level, const char *facility, const char (&text)[size])
	{
		SendLogMessage(context, level, facility, mpt::ToUnicode(mpt::CharsetASCII, text));
	}
};

#define MPT_LOG(level, facility, text) \
	MPT_DO \
	{ \
		MPT_MAYBE_CONSTANT_IF(mpt::log::GlobalLogLevel >= ( level )) \
		{ \
			MPT_MAYBE_CONSTANT_IF(mpt::log::IsFacilityActive(( facility ))) \
			{ \
				mpt::log::Logger().SendLogMessage( MPT_LOG_CURRENTCONTEXT() , ( level ), ( facility ), ( text )); \
			} \
		} \
	} MPT_WHILE_0 \
/**/


#define MPT_LEGACY_LOGLEVEL LogDebug

class LegacyLogger : public Logger
{
private:
	const Context context;
public:
	LegacyLogger(const Context &context) : context(context) {}
	/* MPT_DEPRECATED */ void MPT_PRINTF_FUNC(2,3) operator () (const char *format, ...); // migrate to type-safe MPT_LOG
	/* MPT_DEPRECATED */ void operator () (const AnyStringLocale &text); // migrate to properly namespaced MPT_LOG
	/* MPT_DEPRECATED */ void operator () (LogLevel level, const mpt::ustring &text); // migrate to properly namespaced MPT_LOG
};

#define Log MPT_MAYBE_CONSTANT_IF(mpt::log::GlobalLogLevel < MPT_LEGACY_LOGLEVEL) { } else MPT_MAYBE_CONSTANT_IF(!mpt::log::IsFacilityActive("")) { } else mpt::log::LegacyLogger(MPT_LOG_CURRENTCONTEXT())


#else // !NO_LOGGING


#define MPT_LOG(level, facility, text) MPT_DO { } MPT_WHILE_0

struct LegacyLogger
{
	inline void MPT_PRINTF_FUNC(2,3) operator () (const char * /*format*/ , ...) {}
	inline void operator () (const AnyStringLocale & /*text*/ ) {}
	inline void operator () (LogLevel /*level*/ , const mpt::ustring & /*text*/ ) {}
};
#define Log MPT_CONSTANT_IF(true) {} else mpt::log::LegacyLogger() // completely compile out arguments to Log() so that they do not even get evaluated


#endif // NO_LOGGING


#if defined(MODPLUG_TRACKER) && MPT_OS_WINDOWS

namespace Trace {

// This is not strictly thread safe in all corner cases because of missing barriers.
// We do not care in order to not harm the fast path with additional barriers.
// Enabled tracing incurs a runtime overhead with multiple threads as a global atomic variable
//  gets modified.
// This cacheline bouncing does not matter at all
//  if there are not multiple thread adding trace points at high frequency (way greater than 1000Hz),
//  which, in OpenMPT, is only ever the case for just a single thread (the audio thread), if at all.
extern bool volatile g_Enabled;
static inline bool IsEnabled() { return g_Enabled; }

MPT_NOINLINE void Trace(const mpt::log::Context & contexxt);

enum ThreadKind {
	ThreadKindGUI,
	ThreadKindAudio,
	ThreadKindNotify,
};

void Enable(std::size_t numEntries);
void Disable();

void SetThreadId(mpt::log::Trace::ThreadKind kind, uint32 id);

void Seal();
bool Dump(const mpt::PathString &filename);

#define MPT_TRACE() MPT_DO { if(mpt::log::Trace::g_Enabled) { mpt::log::Trace::Trace(MPT_LOG_CURRENTCONTEXT()); } } MPT_WHILE_0

} // namespace Trace

#else // !MODPLUG_TRACKER

#define MPT_TRACE() MPT_DO { } MPT_WHILE_0

#endif // MODPLUG_TRACKER



} // namespace log
} // namespace mpt


OPENMPT_NAMESPACE_END
