#include <efsw/FileSystem.hpp>
#include <efsw/FileWatcherGeneric.hpp>
#include <efsw/FileWatcherImpl.hpp>
#include <efsw/efsw.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32
#include <efsw/FileWatcherWin32.hpp>
#define FILEWATCHER_IMPL FileWatcherWin32
#define BACKEND_NAME "Win32"
#elif EFSW_PLATFORM == EFSW_PLATFORM_INOTIFY
#include <efsw/FileWatcherInotify.hpp>
#define FILEWATCHER_IMPL FileWatcherInotify
#define BACKEND_NAME "Inotify"
#elif EFSW_PLATFORM == EFSW_PLATFORM_KQUEUE
#include <efsw/FileWatcherKqueue.hpp>
#define FILEWATCHER_IMPL FileWatcherKqueue
#define BACKEND_NAME "Kqueue"
#elif EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS
#include <efsw/FileWatcherFSEvents.hpp>
#define FILEWATCHER_IMPL FileWatcherFSEvents
#define BACKEND_NAME "FSEvents"
#else
#define FILEWATCHER_IMPL FileWatcherGeneric
#define BACKEND_NAME "Generic"
#endif

#include <efsw/Debug.hpp>

namespace efsw {

FileWatcher::FileWatcher() : mFollowSymlinks( false ), mOutOfScopeLinks( false ) {
	efDEBUG( "Using backend: %s\n", BACKEND_NAME );

	mImpl = new FILEWATCHER_IMPL( this );

	if ( !mImpl->initOK() ) {
		efSAFE_DELETE( mImpl );

		efDEBUG( "Falled back to backend: %s\n", BACKEND_NAME );

		mImpl = new FileWatcherGeneric( this );
	}
}

FileWatcher::FileWatcher( bool useGenericFileWatcher ) :
	mFollowSymlinks( false ), mOutOfScopeLinks( false ) {
	if ( useGenericFileWatcher ) {
		efDEBUG( "Using backend: Generic\n" );

		mImpl = new FileWatcherGeneric( this );
	} else {
		efDEBUG( "Using backend: %s\n", BACKEND_NAME );

		mImpl = new FILEWATCHER_IMPL( this );

		if ( !mImpl->initOK() ) {
			efSAFE_DELETE( mImpl );

			efDEBUG( "Falled back to backend: %s\n", BACKEND_NAME );

			mImpl = new FileWatcherGeneric( this );
		}
	}
}

FileWatcher::~FileWatcher() {
	efSAFE_DELETE( mImpl );
}

WatchID FileWatcher::addWatch( const std::string& directory, FileWatchListener* watcher ) {
	return addWatch( directory, watcher, false, {} );
}

WatchID FileWatcher::addWatch( const std::string& directory, FileWatchListener* watcher,
							   bool recursive ) {
	return addWatch( directory, watcher, recursive, {} );
}

WatchID FileWatcher::addWatch( const std::string& directory, FileWatchListener* watcher,
							   bool recursive, const std::vector<WatcherOption>& options ) {
	if ( mImpl->mIsGeneric || !FileSystem::isRemoteFS( directory ) ) {
		return mImpl->addWatch( directory, watcher, recursive, options );
	} else {
		return Errors::Log::createLastError( Errors::FileRemote, directory );
	}
}

void FileWatcher::removeWatch( const std::string& directory ) {
	mImpl->removeWatch( directory );
}

void FileWatcher::removeWatch( WatchID watchid ) {
	mImpl->removeWatch( watchid );
}

void FileWatcher::watch() {
	mImpl->watch();
}

std::vector<std::string> FileWatcher::directories() {
	return mImpl->directories();
}

void FileWatcher::followSymlinks( bool follow ) {
	mFollowSymlinks = follow;
}

const bool& FileWatcher::followSymlinks() const {
	return mFollowSymlinks;
}

void FileWatcher::allowOutOfScopeLinks( bool allow ) {
	mOutOfScopeLinks = allow;
}

const bool& FileWatcher::allowOutOfScopeLinks() const {
	return mOutOfScopeLinks;
}

} // namespace efsw
