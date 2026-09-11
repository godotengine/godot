#ifndef EFSW_BASE
#define EFSW_BASE

#include <efsw/efsw.hpp>
#include <efsw/sophist.h>

namespace efsw {

typedef SOPHIST_int8 Int8;
typedef SOPHIST_uint8 Uint8;
typedef SOPHIST_int16 Int16;
typedef SOPHIST_uint16 Uint16;
typedef SOPHIST_int32 Int32;
typedef SOPHIST_uint32 Uint32;
typedef SOPHIST_int64 Int64;
typedef SOPHIST_uint64 Uint64;

#define EFSW_OS_WIN 1
#define EFSW_OS_LINUX 2
#define EFSW_OS_MACOSX 3
#define EFSW_OS_BSD 4
#define EFSW_OS_SOLARIS 5
#define EFSW_OS_HAIKU 6
#define EFSW_OS_ANDROID 7
#define EFSW_OS_IOS 8

#define EFSW_PLATFORM_WIN32 1
#define EFSW_PLATFORM_INOTIFY 2
#define EFSW_PLATFORM_KQUEUE 3
#define EFSW_PLATFORM_FSEVENTS 4
#define EFSW_PLATFORM_GENERIC 5

#if defined( _WIN32 )
///	Any Windows platform
#define EFSW_OS EFSW_OS_WIN
#define EFSW_PLATFORM EFSW_PLATFORM_WIN32

#if ( defined( _MSCVER ) || defined( _MSC_VER ) )
#define EFSW_COMPILER_MSVC
#endif

/// Force windows target version above or equal to Windows Server 2008 or Windows Vista
#if _WIN32_WINNT < 0x600
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x600
#endif
#elif defined( __FreeBSD__ ) || defined( __OpenBSD__ ) || defined( __NetBSD__ ) || \
	defined( __DragonFly__ )
#define EFSW_OS EFSW_OS_BSD
#define EFSW_PLATFORM EFSW_PLATFORM_KQUEUE

#elif defined( __APPLE_CC__ ) || defined( __APPLE__ )
#include <TargetConditionals.h>

#if defined( __IPHONE__ ) || ( defined( TARGET_OS_IPHONE ) && TARGET_OS_IPHONE ) || \
	( defined( TARGET_IPHONE_SIMULATOR ) && TARGET_IPHONE_SIMULATOR )
#define EFSW_OS EFSW_OS_IOS
#define EFSW_PLATFORM EFSW_PLATFORM_KQUEUE
#else
#define EFSW_OS EFSW_OS_MACOSX

#if defined( EFSW_FSEVENTS_NOT_SUPPORTED )
#define EFSW_PLATFORM EFSW_PLATFORM_KQUEUE
#else
#define EFSW_PLATFORM EFSW_PLATFORM_FSEVENTS
#endif
#endif

#elif defined( __linux__ )
///	This includes Linux and Android
#ifndef EFSW_KQUEUE
#define EFSW_PLATFORM EFSW_PLATFORM_INOTIFY
#else
/// This is for testing libkqueue, sadly it doesnt work
#define EFSW_PLATFORM EFSW_PLATFORM_KQUEUE
#endif

#if defined( __ANDROID__ ) || defined( ANDROID )
#define EFSW_OS EFSW_OS_ANDROID
#else
#define EFSW_OS EFSW_OS_LINUX
#endif

#else
#if defined( __SVR4 )
#define EFSW_OS EFSW_OS_SOLARIS
#elif defined( __HAIKU__ ) || defined( __BEOS__ )
#define EFSW_OS EFSW_OS_HAIKU
#endif

///	Everything else
#define EFSW_PLATFORM EFSW_PLATFORM_GENERIC
#endif

#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
#define EFSW_PLATFORM_POSIX
#endif

#if 1 == SOPHIST_pointer64
#define EFSW_64BIT
#else
#define EFSW_32BIT
#endif

#if defined( arm ) || defined( __arm__ )
#define EFSW_ARM
#endif

#define efCOMMA ,

#define efSAFE_DELETE( p ) \
	{                      \
		if ( p ) {         \
			delete ( p );  \
			( p ) = NULL;  \
		}                  \
	}
#define efSAFE_DELETE_ARRAY( p ) \
	{                            \
		if ( p ) {               \
			delete[] ( p );      \
			( p ) = NULL;        \
		}                        \
	}
#define efARRAY_SIZE( __array ) ( sizeof( __array ) / sizeof( __array[0] ) )

} // namespace efsw

#endif
