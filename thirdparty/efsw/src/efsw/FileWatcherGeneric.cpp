#include <efsw/FileSystem.hpp>
#include <efsw/FileWatcherGeneric.hpp>
#include <efsw/Lock.hpp>
#include <efsw/System.hpp>

namespace efsw {

FileWatcherGeneric::FileWatcherGeneric( FileWatcher* parent ) :
	FileWatcherImpl( parent ), mThread( NULL ), mLastWatchID( 0 ) {
	mInitOK = true;
	mIsGeneric = true;
}

FileWatcherGeneric::~FileWatcherGeneric() {
	mInitOK = false;

	efSAFE_DELETE( mThread );

	/// Delete the watches
	WatchList::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); ++it ) {
		efSAFE_DELETE( ( *it ) );
	}
}

WatchID FileWatcherGeneric::addWatch( const std::string& directory, FileWatchListener* watcher,
									  bool recursive, const std::vector<WatcherOption>& options ) {
	std::string dir( directory );

	FileSystem::dirAddSlashAtEnd( dir );

	FileInfo fi( dir );

	if ( !fi.isDirectory() ) {
		return Errors::Log::createLastError( Errors::FileNotFound, dir );
	} else if ( !fi.isReadable() ) {
		return Errors::Log::createLastError( Errors::FileNotReadable, dir );
	} else if ( pathInWatches( dir ) ) {
		return Errors::Log::createLastError( Errors::FileRepeated, dir );
	}

	std::string curPath;
	std::string link( FileSystem::getLinkRealPath( dir, curPath ) );

	if ( "" != link ) {
		if ( pathInWatches( link ) ) {
			return Errors::Log::createLastError( Errors::FileRepeated, dir );
		} else if ( !linkAllowed( curPath, link ) ) {
			return Errors::Log::createLastError( Errors::FileOutOfScope, dir );
		} else {
			dir = link;
		}
	}

	mLastWatchID++;

	WatcherGeneric* pWatch = new WatcherGeneric( mLastWatchID, dir, watcher, this, recursive );

	Lock lock( mWatchesLock );
	mWatches.push_back( pWatch );

	return pWatch->ID;
}

void FileWatcherGeneric::removeWatch( const std::string& directory ) {
	std::string dir( directory );
	FileSystem::dirAddSlashAtEnd( dir );

	WatchList::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); ++it ) {
		if ( ( *it )->Directory == dir ) {
			WatcherGeneric* watch = ( *it );

			Lock lock( mWatchesLock );

			mWatches.erase( it );

			efSAFE_DELETE( watch );

			return;
		}
	}
}

void FileWatcherGeneric::removeWatch( WatchID watchid ) {
	WatchList::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); ++it ) {
		if ( ( *it )->ID == watchid ) {
			WatcherGeneric* watch = ( *it );

			Lock lock( mWatchesLock );

			mWatches.erase( it );

			efSAFE_DELETE( watch );

			return;
		}
	}
}

void FileWatcherGeneric::watch() {
	if ( NULL == mThread ) {
		mThread = new Thread([this]{run();});
		mThread->launch();
	}
}

void FileWatcherGeneric::run() {
	do {
		{
			Lock lock( mWatchesLock );

			WatchList::iterator it = mWatches.begin();

			for ( ; it != mWatches.end(); ++it ) {
				( *it )->watch();
			}
		}

		if ( mInitOK )
			System::sleep( 1000 );
	} while ( mInitOK );
}

void FileWatcherGeneric::handleAction( Watcher*, const std::string&, bool, unsigned long,
									   std::string ) {
	/// Not used
}

std::vector<std::string> FileWatcherGeneric::directories() {
	std::vector<std::string> dirs;

	Lock lock( mWatchesLock );

	dirs.reserve( mWatches.size() );

	WatchList::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); ++it ) {
		dirs.push_back( ( *it )->Directory );
	}

	return dirs;
}

bool FileWatcherGeneric::pathInWatches( const std::string& path ) {
	WatchList::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); ++it ) {
		if ( ( *it )->Directory == path || ( *it )->pathInWatches( path ) ) {
			return true;
		}
	}

	return false;
}

} // namespace efsw
