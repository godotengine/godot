/* vim: tabstop=4 shiftwidth=4 noexpandtab
 * MiniUPnP project
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * (c) 2020 Pali Rohár
 * This software is subject to the conditions detailed
 * in the LICENCE file provided within the distribution */

#ifndef WIN32_SNPRINTF_H
#define WIN32_SNPRINTF_H

#ifdef _WIN32

#include <stdio.h>

/* snprintf is supported by:
 *  - Visual Studio 2015 or new
 *  - mingw32 with iso c ext
 *  - mingw-w64 with ansi stdio
 *  - mingw-w64 6.0.0 or new with ucrt
 *  - mingw-w64 8.0.0 or new with iso c ext
 */
#if ( \
	(defined(_MSC_VER) && _MSC_VER < 1900) /* Visual Studio older than 2015 */ || \
	(defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR) && defined(__NO_ISOCEXT)) /* mingw32 without iso c ext */ || \
	(defined(__MINGW64_VERSION_MAJOR) && /* mingw-w64 not ... */ !( \
		(defined (__USE_MINGW_ANSI_STDIO) && __USE_MINGW_ANSI_STDIO != 0) /* ... with ansi stdio */ || \
		(__MINGW64_VERSION_MAJOR >= 6 && defined(_UCRT)) /* ... at least 6.0.0 with ucrt */ || \
		(__MINGW64_VERSION_MAJOR >= 8 && !defined(__NO_ISOCEXT))) /* ... at least 8.0.0 with iso c ext */ || \
	0) || \
0)

/* _scprintf is supported by:
 *  - Visual Studio 2002 or new
 *  - msvcr70.dll or new
 *  - msvcrt.dll on Windows XP or new
 */
#if ( \
	(defined(_MSC_VER) && _MSC_VER < 1300) /* Visual Studio older than 2002 */ || \
	(defined(__MSVCRT_VERSION__) && __MSVCRT_VERSION__ < 0x700) /* msvcrt older than 7.0 */ || \
0)
#define CHECK_SCPRINTF 0
#define IF_SCPRINTF(expr) 0
#define ELSE_SCPRINTF(expr) expr
#else
#define CHECK_SCPRINTF 1
#define IF_SCPRINTF(expr) expr
#define ELSE_SCPRINTF(expr) 0
#endif

/* Emulation of snprintf for win32 */
#define snprintf(buf, size, fmt, ...) ( \
	(((size) != 0 && (buf) != NULL) ? ( /* _snprintf does not work with NULL buffer */ \
		_snprintf((buf), (size), (fmt), __VA_ARGS__), /* _snprintf returns -1 on overflow, so ignore its value */ \
		(((char *)buf)[(size_t)(size)-1] = 0), /* _snprintf does not fill nul byte on overflow */ \
	0) : 0), \
	(CHECK_SCPRINTF ? IF_SCPRINTF( \
		_scprintf((fmt), __VA_ARGS__) /* calculate return value for snprintf via _scprintf */ \
	) : ELSE_SCPRINTF( \
		((size) != 0 && (buf) != NULL) ? \
			strlen((buf)) /* return just length of buffer */ \
		: \
			1 /* no buffer, impossible to calculate, return just non-zero number */ \
		) \
	) \
)

#endif

#endif /* _WIN32 */

#endif /* WIN32_SNPRINTF_H */
