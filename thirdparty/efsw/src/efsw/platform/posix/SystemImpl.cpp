#include <efsw/platform/posix/SystemImpl.hpp>

#if defined( EFSW_PLATFORM_POSIX )

#include <cstdio>
#include <limits.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <efsw/Debug.hpp>
#include <efsw/FileSystem.hpp>

#if EFSW_OS == EFSW_OS_MACOSX
#include <CoreFoundation/CoreFoundation.h>
#elif EFSW_OS == EFSW_OS_LINUX || EFSW_OS == EFSW_OS_ANDROID
#include <libgen.h>
#include <unistd.h>
#elif EFSW_OS == EFSW_OS_HAIKU
#include <kernel/OS.h>
#include <kernel/image.h>
#elif EFSW_OS == EFSW_OS_SOLARIS
#include <stdlib.h>
#elif EFSW_OS == EFSW_OS_BSD
#include <sys/sysctl.h>
#endif

namespace efsw { namespace Platform {

void System::sleep( const unsigned long& ms ) {
	// usleep( static_cast<unsigned long>( ms * 1000 ) );

	// usleep is not reliable enough (it might block the
	// whole process instead of just the current thread)
	// so we must use pthread_cond_timedwait instead

	// this implementation is inspired from Qt
	// and taken from SFML

	unsigned long long usecs = ms * 1000;

	// get the current time
	timeval tv;
	gettimeofday( &tv, NULL );

	// construct the time limit (current time + time to wait)
	timespec ti;
	ti.tv_nsec = ( tv.tv_usec + ( usecs % 1000000 ) ) * 1000;
	ti.tv_sec = tv.tv_sec + ( usecs / 1000000 ) + ( ti.tv_nsec / 1000000000 );
	ti.tv_nsec %= 1000000000;

	// create a mutex and thread condition
	pthread_mutex_t mutex;
	pthread_mutex_init( &mutex, 0 );
	pthread_cond_t condition;
	pthread_cond_init( &condition, 0 );

	// wait...
	pthread_mutex_lock( &mutex );
	pthread_cond_timedwait( &condition, &mutex, &ti );
	pthread_mutex_unlock( &mutex );

	// destroy the mutex and condition
	pthread_cond_destroy( &condition );
}

std::string System::getProcessPath() {
#if EFSW_OS == EFSW_OS_MACOSX
	char exe_file[FILENAME_MAX + 1];

	CFBundleRef mainBundle = CFBundleGetMainBundle();

	if ( mainBundle ) {
		CFURLRef mainURL = CFBundleCopyBundleURL( mainBundle );

		if ( mainURL ) {
			int ok = CFURLGetFileSystemRepresentation( mainURL, ( Boolean ) true, (UInt8*)exe_file,
													   FILENAME_MAX );

			if ( ok ) {
				return std::string( exe_file ) + "/";
			}
		}
	}

	return "./";
#elif EFSW_OS == EFSW_OS_LINUX
	char exe_file[FILENAME_MAX + 1];

	int size;

	size = readlink( "/proc/self/exe", exe_file, FILENAME_MAX );

	if ( size < 0 ) {
		return std::string( "./" );
	} else {
		exe_file[size] = '\0';
		return std::string( dirname( exe_file ) ) + "/";
	}

#elif EFSW_OS == EFSW_OS_BSD
	int mib[4];
	mib[0] = CTL_KERN;
	mib[1] = KERN_PROC;
	mib[2] = KERN_PROC_PATHNAME;
	mib[3] = -1;
	char buf[1024];
	size_t cb = sizeof( buf );
	sysctl( mib, 4, buf, &cb, NULL, 0 );

	return FileSystem::pathRemoveFileName( std::string( buf ) );

#elif EFSW_OS == EFSW_OS_SOLARIS
	return FileSystem::pathRemoveFileName( std::string( getexecname() ) );

#elif EFSW_OS == EFSW_OS_HAIKU
	image_info info;
	int32 cookie = 0;

	while ( B_OK == get_next_image_info( 0, &cookie, &info ) ) {
		if ( info.type == B_APP_IMAGE )
			break;
	}

	return FileSystem::pathRemoveFileName( std::string( info.name ) );

#elif EFSW_OS == EFSW_OS_ANDROID
	return "/sdcard/";

#else
#warning getProcessPath() not implemented on this platform. ( will return "./" )
	return "./";

#endif
}

void System::maxFD() {
	static bool maxed = false;

	if ( !maxed ) {
		struct rlimit limit;
		getrlimit( RLIMIT_NOFILE, &limit );
		limit.rlim_cur = limit.rlim_max;
		setrlimit( RLIMIT_NOFILE, &limit );

		getrlimit( RLIMIT_NOFILE, &limit );

		efDEBUG( "File descriptor limit %ld\n", limit.rlim_cur );

		maxed = true;
	}
}

Uint64 System::getMaxFD() {
	static rlim_t max_fd = 0;

	if ( max_fd == 0 ) {
		struct rlimit limit;
		getrlimit( RLIMIT_NOFILE, &limit );
		max_fd = limit.rlim_cur;
	}

	return max_fd;
}

}} // namespace efsw::Platform

#endif
