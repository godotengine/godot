/* 
	config.h.

	Manual config for Godot attempting to combine different platforms for compiling with Scons.

	Needs to be enhanced for Linux!
*/

/* Default visibility */
#define DEFAULT_VISIBILITY /**/

/* Enable global message logging */
#define ENABLE_LOGGING 1

/* Uncomment to start with debug message logging enabled */
// #define ENABLE_DEBUG_LOGGING 1

/* Uncomment to enabling logging to system log */
// #define USE_SYSTEM_LOGGING_FACILITY

#ifdef OSX_ENABLED

/* Define to 1 if you have the <poll.h> header file. */
#define HAVE_POLL_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Darwin backend */
#define OS_DARWIN 1

/* type of second poll() argument */
#define POLL_NFDS_TYPE nfds_t

/* Use POSIX Threads */
#define THREADS_POSIX 1

/* Use GNU extensions */
#define _GNU_SOURCE 1

#endif

#ifdef X11_ENABLED

/* Define to 1 if you have the <poll.h> header file. */
#define HAVE_POLL_H 1

/* type of second poll() argument */
#define POLL_NFDS_TYPE nfds_t

/* Use POSIX Threads */
#define THREADS_POSIX 1

/* Use GNU extensions */
#define _GNU_SOURCE 1

#endif

#ifdef WINDOWS_ENABLED

#ifndef _MSC_VER
#warn "msvc/config.h shouldn't be included for your development environment."
#error "Please make sure the msvc/ directory is removed from your build path."
#endif

/* Visual Studio 2015 and later defines timespec */
#if defined(_MSC_VER) && (_MSC_VER >= 1900)
#define _TIMESPEC_DEFINED 1
#endif

/* Disable: warning C4200: nonstandard extension used : zero-sized array in struct/union */
#pragma warning(disable:4200)
/* Disable: warning C6258: Using TerminateThread does not allow proper thread clean up */
#pragma warning(disable: 6258)
/* Disable: warning C4996: 'GetVersionA': was declared deprecated */
#pragma warning(disable: 4996)

#if defined(_PREFAST_)
/* Disable "Banned API" errors when using the MS's WDK OACR/Prefast */
#pragma warning(disable:28719)
/* Disable "The function 'InitializeCriticalSection' must be called from within a try/except block" */
#pragma warning(disable:28125)
#endif

/* type of second poll() argument */
#define POLL_NFDS_TYPE unsigned int

/* Windows/WinCE backend */
#if defined(_WIN32_WCE)
#define OS_WINCE 1
#define HAVE_MISSING_H
#else
#define OS_WINDOWS 1
#define HAVE_SYS_TYPES_H 1
#endif

#endif
