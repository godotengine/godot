#include <efsw/WatcherKqueue.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_KQUEUE || EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <efsw/Debug.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/FileWatcherKqueue.hpp>
#include <efsw/String.hpp>
#include <efsw/System.hpp>
#include <efsw/WatcherGeneric.hpp>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define KEVENT_RESERVE_VALUE ( 10 )

#ifndef O_EVTONLY
#define O_EVTONLY ( O_RDONLY | O_NONBLOCK )
#endif

namespace efsw {

int comparator( const void* ke1, const void* ke2 ) {
	const KEvent* kev1 = reinterpret_cast<const KEvent*>( ke1 );
	const KEvent* kev2 = reinterpret_cast<const KEvent*>( ke2 );

	if ( NULL != kev2->udata ) {
		FileInfo* fi1 = reinterpret_cast<FileInfo*>( kev1->udata );
		FileInfo* fi2 = reinterpret_cast<FileInfo*>( kev2->udata );

		return strcmp( fi1->Filepath.c_str(), fi2->Filepath.c_str() );
	}

	return 1;
}

WatcherKqueue::WatcherKqueue( WatchID watchid, const std::string& dirname,
							  FileWatchListener* listener, bool recursive,
							  FileWatcherKqueue* watcher, WatcherKqueue* parent ) :
	Watcher( watchid, dirname, listener, recursive ),
	mLastWatchID( 0 ),
	mChangeListCount( 0 ),
	mKqueue( kqueue() ),
	mWatcher( watcher ),
	mParent( parent ),
	mInitOK( true ),
	mErrno( 0 ) {
	if ( -1 == mKqueue ) {
		efDEBUG(
			"kqueue() returned invalid descriptor for directory %s. File descriptors count: %ld\n",
			Directory.c_str(), mWatcher->mFileDescriptorCount );

		mInitOK = false;
		mErrno = errno;
	} else {
		mWatcher->addFD();
	}
}

WatcherKqueue::~WatcherKqueue() {
	// Remove the childs watchers ( sub-folders watches )
	removeAll();

	for ( size_t i = 0; i < mChangeListCount; i++ ) {
		if ( NULL != mChangeList[i].udata ) {
			FileInfo* fi = reinterpret_cast<FileInfo*>( mChangeList[i].udata );

			efSAFE_DELETE( fi );
		}
	}

	close( mKqueue );

	mWatcher->removeFD();
}

void WatcherKqueue::addAll() {
	if ( -1 == mKqueue ) {
		return;
	}

	// scan directory and call addFile(name, false) on each file
	FileSystem::dirAddSlashAtEnd( Directory );

	efDEBUG( "addAll(): Added folder: %s\n", Directory.c_str() );

	// add base dir
	int fd = open( Directory.c_str(), O_EVTONLY );

	if ( -1 == fd ) {
		efDEBUG( "addAll(): Couldn't open folder: %s\n", Directory.c_str() );

		if ( EACCES != errno ) {
			mInitOK = false;
		}

		mErrno = errno;

		return;
	}

	mDirSnap.setDirectoryInfo( Directory );
	mDirSnap.scan();

	mChangeList.resize( KEVENT_RESERVE_VALUE );

	// Creates the kevent for the folder
	EV_SET( &mChangeList[0], fd, EVFILT_VNODE, EV_ADD | EV_ENABLE | EV_ONESHOT,
			NOTE_DELETE | NOTE_EXTEND | NOTE_WRITE | NOTE_ATTRIB | NOTE_RENAME, 0, 0 );

	mWatcher->addFD();

	// Get the files and directories from the directory
	FileInfoMap files = FileSystem::filesInfoFromPath( Directory );

	for ( FileInfoMap::iterator it = files.begin(); it != files.end(); it++ ) {
		FileInfo& fi = it->second;

		if ( fi.isRegularFile() ) {
			// Add the regular files kevent
			addFile( fi.Filepath, false );
		} else if ( Recursive && fi.isDirectory() && fi.isReadable() ) {
			// Create another watcher for the subfolders ( if recursive )
			WatchID id = addWatch( fi.Filepath, Listener, Recursive, this );

			// If the watcher is not adding the watcher means that the directory was created
			if ( id > 0 && !mWatcher->isAddingWatcher() ) {
				handleFolderAction( fi.Filepath, Actions::Add );
			}
		}
	}
}

void WatcherKqueue::removeAll() {
	efDEBUG( "removeAll(): Removing all child watchers\n" );

	std::vector<WatchID> erase;

	for ( WatchMap::iterator it = mWatches.begin(); it != mWatches.end(); it++ ) {
		efDEBUG( "removeAll(): Removed child watcher %s\n", it->second->Directory.c_str() );

		erase.push_back( it->second->ID );
	}

	for ( std::vector<WatchID>::iterator eit = erase.begin(); eit != erase.end(); eit++ ) {
		removeWatch( *eit );
	}
}

void WatcherKqueue::addFile( const std::string& name, bool emitEvents ) {
	efDEBUG( "addFile(): Added: %s\n", name.c_str() );

	// Open the file to get the file descriptor
	int fd = open( name.c_str(), O_EVTONLY );

	if ( fd == -1 ) {
		efDEBUG( "addFile(): Could open file descriptor for %s. File descriptor count: %ld\n",
				 name.c_str(), mWatcher->mFileDescriptorCount );

		Errors::Log::createLastError( Errors::FileNotReadable, name );

		if ( EACCES != errno ) {
			mInitOK = false;
		}

		mErrno = errno;

		return;
	}

	mWatcher->addFD();

	// increase the file kevent file count
	mChangeListCount++;

	if ( mChangeListCount + KEVENT_RESERVE_VALUE > mChangeList.size() &&
		 mChangeListCount % KEVENT_RESERVE_VALUE == 0 ) {
		size_t reserve_size = mChangeList.size() + KEVENT_RESERVE_VALUE;
		mChangeList.resize( reserve_size );
		efDEBUG( "addFile(): Reserverd more KEvents space for %s, space reserved %ld, list actual "
				 "size %ld.\n",
				 Directory.c_str(), reserve_size, mChangeListCount );
	}

	// create entry
	FileInfo* entry = new FileInfo( name );

	// set the event data at the end of the list
	EV_SET( &mChangeList[mChangeListCount], fd, EVFILT_VNODE, EV_ADD | EV_ENABLE | EV_ONESHOT,
			NOTE_DELETE | NOTE_EXTEND | NOTE_WRITE | NOTE_ATTRIB | NOTE_RENAME, 0, (void*)entry );

	// qsort sort the list by name
	qsort( &mChangeList[1], mChangeListCount, sizeof( KEvent ), comparator );

	// handle action
	if ( emitEvents ) {
		handleAction( name, false, Actions::Add );
	}
}

void WatcherKqueue::removeFile( const std::string& name, bool emitEvents ) {
	efDEBUG( "removeFile(): Trying to remove file: %s\n", name.c_str() );

	// bsearch
	KEvent target;

	// Create a temporary file info to search the kevent ( searching the directory )
	FileInfo tempEntry( name );

	target.udata = &tempEntry;

	// Search the kevent
	KEvent* ke = (KEvent*)bsearch( &target, &mChangeList[0], mChangeListCount + 1, sizeof( KEvent ),
								   comparator );

	// Trying to remove a non-existing file?
	if ( !ke ) {
		Errors::Log::createLastError( Errors::FileNotFound, name );
		efDEBUG( "File not removed\n" );
		return;
	}

	efDEBUG( "File removed\n" );

	// handle action
	if ( emitEvents ) {
		handleAction( name, false, Actions::Delete );
	}

	// Delete the user data ( FileInfo ) from the kevent closed
	FileInfo* del = reinterpret_cast<FileInfo*>( ke->udata );

	efSAFE_DELETE( del );

	// close the file descriptor from the kevent
	close( ke->ident );

	mWatcher->removeFD();

	memset( ke, 0, sizeof( KEvent ) );

	// move end to current
	memcpy( ke, &mChangeList[mChangeListCount], sizeof( KEvent ) );
	memset( &mChangeList[mChangeListCount], 0, sizeof( KEvent ) );
	--mChangeListCount;
}

void WatcherKqueue::rescan() {
	efDEBUG( "rescan(): Rescanning: %s\n", Directory.c_str() );

	DirectorySnapshotDiff Diff = mDirSnap.scan();

	if ( Diff.DirChanged ) {
		sendDirChanged();
	}

	if ( Diff.changed() ) {
		FileInfoList::iterator it;
		MovedList::iterator mit;

		/// Files
		DiffIterator( FilesCreated ) {
			addFile( ( *it ).Filepath );
		}

		DiffIterator( FilesModified ) {
			handleAction( ( *it ).Filepath, false, Actions::Modified );
		}

		DiffIterator( FilesDeleted ) {
			removeFile( ( *it ).Filepath );
		}

		DiffMovedIterator( FilesMoved ) {
			handleAction( ( *mit ).second.Filepath, false, Actions::Moved, ( *mit ).first );
			removeFile( Directory + ( *mit ).first, false );
			addFile( ( *mit ).second.Filepath, false );
		}

		/// Directories
		DiffIterator( DirsCreated ) {
			handleFolderAction( ( *it ).Filepath, Actions::Add );
			addWatch( ( *it ).Filepath, Listener, Recursive, this );
		}

		DiffIterator( DirsModified ) {
			handleFolderAction( ( *it ).Filepath, Actions::Modified );
		}

		DiffIterator( DirsDeleted ) {
			handleFolderAction( ( *it ).Filepath, Actions::Delete );

			Watcher* watch = findWatcher( ( *it ).Filepath );

			if ( NULL != watch ) {
				removeWatch( watch->ID );
			}
		}

		DiffMovedIterator( DirsMoved ) {
			moveDirectory( Directory + ( *mit ).first, ( *mit ).second.Filepath );
		}
	}
}

WatchID WatcherKqueue::watchingDirectory( std::string dir ) {
	Watcher* watch = findWatcher( dir );

	if ( NULL != watch ) {
		return watch->ID;
	}

	return Errors::FileNotFound;
}

void WatcherKqueue::handleAction( const std::string& filename, bool isDir, efsw::Action action,
								  const std::string& oldFilename ) {
	Listener->handleFileAction( ID, Directory, FileSystem::fileNameFromPath( filename ), isDir,
								action, FileSystem::fileNameFromPath( oldFilename ) );
}

void WatcherKqueue::handleFolderAction( std::string filename, efsw::Action action,
										const std::string& oldFilename ) {
	FileSystem::dirRemoveSlashAtEnd( filename );

	handleAction( filename, true, action, oldFilename );
}

void WatcherKqueue::sendDirChanged() {
	if ( NULL != mParent ) {
		Listener->handleFileAction( mParent->ID, mParent->Directory,
									FileSystem::fileNameFromPath( Directory ), true,
									Actions::Modified );
	}
}

void WatcherKqueue::watch() {
	if ( -1 == mKqueue ) {
		return;
	}

	int nev = 0;
	KEvent event;

	// First iterate the childs, to get the events from the deepest folder, to the watcher childs
	for ( WatchMap::iterator it = mWatches.begin(); it != mWatches.end(); ++it ) {
		it->second->watch();
	}

	bool needScan = false;

	// Then we get the the events of the current folder
	while ( !mChangeList.empty() &&
			( nev = kevent( mKqueue, mChangeList.data(), mChangeListCount + 1, &event, 1,
							&mWatcher->mTimeOut ) ) != 0 ) {
		// An error ocurred?
		if ( nev == -1 ) {
			efDEBUG( "watch(): Error on directory %s\n", Directory.c_str() );
			perror( "kevent" );
			break;
		} else {
			FileInfo* entry = NULL;

			// If udate == NULL means that it is the fisrt element of the change list, the folder.
			// otherwise it is an event of some file inside the folder
			if ( ( entry = reinterpret_cast<FileInfo*>( event.udata ) ) != NULL ) {
				efDEBUG( "watch(): File: %s ", entry->Filepath.c_str() );

				// If the event flag is delete... the file was deleted
				if ( event.fflags & NOTE_DELETE ) {
					efDEBUG( "deleted\n" );

					mDirSnap.removeFile( entry->Filepath );

					removeFile( entry->Filepath );
				} else if ( event.fflags & NOTE_EXTEND || event.fflags & NOTE_WRITE ||
							event.fflags & NOTE_ATTRIB ) {
					// The file was modified
					efDEBUG( "modified\n" );

					FileInfo fi( entry->Filepath );

					if ( fi != *entry ) {
						*entry = fi;

						mDirSnap.updateFile( entry->Filepath );

						handleAction( entry->Filepath, entry->isDirectory(),
									  efsw::Actions::Modified );
					}
				} else if ( event.fflags & NOTE_RENAME ) {
					efDEBUG( "moved\n" );

					needScan = true;
				}
			} else {
				needScan = true;
			}
		}
	}

	if ( needScan ) {
		rescan();
	}
}

Watcher* WatcherKqueue::findWatcher( const std::string path ) {
	WatchMap::iterator it = mWatches.begin();

	for ( ; it != mWatches.end(); it++ ) {
		if ( it->second->Directory == path ) {
			return it->second;
		}
	}

	return NULL;
}

void WatcherKqueue::moveDirectory( std::string oldPath, std::string newPath, bool emitEvents ) {
	// Update the directory path if it's a watcher
	std::string opath2( oldPath );
	FileSystem::dirAddSlashAtEnd( opath2 );

	Watcher* watch = findWatcher( opath2 );

	if ( NULL != watch ) {
		watch->Directory = opath2;
	}

	if ( emitEvents ) {
		handleFolderAction( newPath, efsw::Actions::Moved, oldPath );
	}
}

WatchID WatcherKqueue::addWatch( const std::string& directory, FileWatchListener* watcher,
								 bool recursive, WatcherKqueue* parent ) {
	static bool s_ug = false;

	std::string dir( directory );

	FileSystem::dirAddSlashAtEnd( dir );

	// This should never happen here
	if ( !FileSystem::isDirectory( dir ) ) {
		return Errors::Log::createLastError( Errors::FileNotFound, dir );
	} else if ( pathInWatches( dir ) || pathInParent( dir ) ) {
		return Errors::Log::createLastError( Errors::FileRepeated, directory );
	} else if ( NULL != parent && FileSystem::isRemoteFS( dir ) ) {
		return Errors::Log::createLastError( Errors::FileRemote, dir );
	}

	std::string curPath;
	std::string link( FileSystem::getLinkRealPath( dir, curPath ) );

	if ( "" != link ) {
		/// Avoid adding symlinks directories if it's now enabled
		if ( NULL != parent && !mWatcher->mFileWatcher->followSymlinks() ) {
			return Errors::Log::createLastError( Errors::FileOutOfScope, dir );
		}

		if ( pathInWatches( link ) || pathInParent( link ) ) {
			return Errors::Log::createLastError( Errors::FileRepeated, link );
		} else if ( !mWatcher->linkAllowed( curPath, link ) ) {
			return Errors::Log::createLastError( Errors::FileOutOfScope, link );
		} else {
			dir = link;
		}
	}

	if ( mWatcher->availablesFD() ) {
		WatcherKqueue* watch =
			new WatcherKqueue( ++mLastWatchID, dir, watcher, recursive, mWatcher, parent );

		mWatches.insert( std::make_pair( mLastWatchID, watch ) );

		watch->addAll();

		// if failed to open the directory... erase the watcher
		if ( !watch->initOK() ) {
			int le = watch->lastErrno();

			mWatches.erase( watch->ID );

			efSAFE_DELETE( watch );

			mLastWatchID--;

			// Probably the folder has too many files, create a generic watcher
			if ( EACCES != le ) {
				WatcherGeneric* watch =
					new WatcherGeneric( ++mLastWatchID, dir, watcher, mWatcher, recursive );

				mWatches.insert( std::make_pair( mLastWatchID, watch ) );
			} else {
				return Errors::Log::createLastError( Errors::Unspecified, link );
			}
		}
	} else {
		if ( !s_ug ) {
			efDEBUG( "Started using WatcherGeneric, reached file descriptors limit: %ld.\n",
					 mWatcher->mFileDescriptorCount );
			s_ug = true;
		}

		WatcherGeneric* watch =
			new WatcherGeneric( ++mLastWatchID, dir, watcher, mWatcher, recursive );

		mWatches.insert( std::make_pair( mLastWatchID, watch ) );
	}

	return mLastWatchID;
}

bool WatcherKqueue::initOK() {
	return mInitOK;
}

void WatcherKqueue::removeWatch( WatchID watchid ) {
	WatchMap::iterator iter = mWatches.find( watchid );

	if ( iter == mWatches.end() )
		return;

	Watcher* watch = iter->second;

	mWatches.erase( iter );

	efSAFE_DELETE( watch );
}

bool WatcherKqueue::pathInWatches( const std::string& path ) {
	return NULL != findWatcher( path );
}

bool WatcherKqueue::pathInParent( const std::string& path ) {
	WatcherKqueue* pNext = mParent;

	while ( NULL != pNext ) {
		if ( pNext->pathInWatches( path ) ) {
			return true;
		}

		pNext = pNext->mParent;
	}

	if ( mWatcher->pathInWatches( path ) ) {
		return true;
	}

	if ( path == Directory ) {
		return true;
	}

	return false;
}

int WatcherKqueue::lastErrno() {
	return mErrno;
}

} // namespace efsw

#endif
