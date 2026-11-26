#include <efsw/Debug.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/FileWatcherFSEvents.hpp>
#include <efsw/WatcherFSEvents.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS

namespace efsw {

WatcherFSEvents::WatcherFSEvents() :
	Watcher(), FWatcher( NULL ), FSStream( NULL ), WatcherGen( NULL ) {}

WatcherFSEvents::~WatcherFSEvents() {
	if ( NULL != FSStream ) {
		FSEventStreamStop( FSStream );
		FSEventStreamInvalidate( FSStream );
		FSEventStreamRelease( FSStream );
	}

	efSAFE_DELETE( WatcherGen );
}

void WatcherFSEvents::init() {
	CFStringRef CFDirectory =
		CFStringCreateWithCString( NULL, Directory.c_str(), kCFStringEncodingUTF8 );
	CFArrayRef CFDirectoryArray = CFArrayCreate( NULL, (const void**)&CFDirectory, 1, NULL );

	Uint32 streamFlags = kFSEventStreamCreateFlagNone;

	if ( FileWatcherFSEvents::isGranular() ) {
		streamFlags = efswFSEventStreamCreateFlagFileEvents | efswFSEventStreamCreateFlagNoDefer |
					  efswFSEventStreamCreateFlagUseExtendedData |
					  efswFSEventStreamCreateFlagUseCFTypes;
	} else {
		WatcherGen = new WatcherGeneric( ID, Directory, Listener, FWatcher.load(), Recursive );
	}

	FSEventStreamContext ctx;
	/* Initialize context */
	ctx.version = 0;
	ctx.info = this;
	ctx.retain = NULL;
	ctx.release = NULL;
	ctx.copyDescription = NULL;

	dispatch_queue_t queue = dispatch_queue_create( NULL, NULL );

	FSStream =
		FSEventStreamCreate( kCFAllocatorDefault, &FileWatcherFSEvents::FSEventCallback, &ctx,
							 CFDirectoryArray, kFSEventStreamEventIdSinceNow, 0., streamFlags );

	FSEventStreamSetDispatchQueue( FSStream, queue );

	FSEventStreamStart( FSStream );

	CFRelease( CFDirectoryArray );
	CFRelease( CFDirectory );
}

void WatcherFSEvents::sendFileAction( WatchID watchid, const std::string& dir,
									  const std::string& filename, bool isDir, Action action,
									  std::string oldFilename ) {
	Listener->handleFileAction( watchid, FileSystem::precomposeFileName( dir ),
								FileSystem::precomposeFileName( filename ), isDir, action,
								FileSystem::precomposeFileName( oldFilename ) );
}

void WatcherFSEvents::sendMissedFileActions( WatchID watchid,
											 const std::string& dir) {
	Listener->handleMissedFileActions( watchid,
									   FileSystem::precomposeFileName( dir ) );
}

void WatcherFSEvents::handleAddModDel( const Uint32& flags, const std::string& path,
									   std::string& dirPath, std::string& filePath, Uint64 inode ) {
	if ( ( flags & efswFSEventStreamEventFlagItemCreated ) && FileInfo::exists( path ) &&
		 ( !SanitizeEvents || FilesAdded.find( inode ) != FilesAdded.end() ) ) {
		sendFileAction( ID, dirPath, filePath, efswFSEventStreamEventFlagItemIsDir & flags,
						Actions::Add );

		if ( SanitizeEvents )
			FilesAdded.insert( inode );
	}

	if ( flags & ModifiedFlags ) {
		sendFileAction( ID, dirPath, filePath, efswFSEventStreamEventFlagItemIsDir & flags,
						Actions::Modified );
	}

	if ( ( flags & efswFSEventStreamEventFlagItemRemoved ) && !FileInfo::exists( path ) ) {
		// Since i don't know the order, at least i try to keep the data consistent with the real
		// state
		sendFileAction( ID, dirPath, filePath, efswFSEventStreamEventFlagItemIsDir & flags,
						Actions::Delete );

		if ( SanitizeEvents )
			FilesAdded.erase( inode );
	}
}

void WatcherFSEvents::handleActions( std::vector<FSEvent>& events ) {
	size_t esize = events.size();

	for ( size_t i = 0; i < esize; i++ ) {
		FSEvent& event = events[i];

		if ( event.Flags &
			 ( kFSEventStreamEventFlagUserDropped | kFSEventStreamEventFlagKernelDropped |
			   kFSEventStreamEventFlagMustScanSubDirs) ) {
			efDEBUG( "Rescan/Drop event for watch: %s - flags: 0x%x\n", Directory.c_str(), event.Flags );
			std::string dirPath = Directory;
			FileSystem::dirRemoveSlashAtEnd( dirPath );
			sendMissedFileActions(ID, dirPath );
			continue;
		}

		if ( event.Flags &
			 ( kFSEventStreamEventFlagEventIdsWrapped | kFSEventStreamEventFlagHistoryDone |
			   kFSEventStreamEventFlagMount | kFSEventStreamEventFlagUnmount |
			   kFSEventStreamEventFlagRootChanged ) ) {
			continue;
		}

		if ( !Recursive ) {
			/** In case that is not recursive the watcher, ignore the events from subfolders */
			if ( event.Path.find_last_of( FileSystem::getOSSlash() ) != Directory.size() - 1 ) {
				continue;
			}
		}

		if ( FileWatcherFSEvents::isGranular() ) {
			std::string dirPath( FileSystem::pathRemoveFileName( event.Path ) );
			std::string filePath( FileSystem::fileNameFromPath( event.Path ) );

			if ( event.Flags &
				 ( efswFSEventStreamEventFlagItemCreated | efswFSEventStreamEventFlagItemRemoved |
				   efswFSEventStreamEventFlagItemRenamed ) ) {
				if ( dirPath != Directory ) {
					DirsChanged.insert( dirPath );
				}
			}

			// This is a mess. But it's FSEvents faults, because shrinks events from the same file
			// in one single event ( so there's no order for them ) For example a file could have
			// been added modified and erased, but i can't know if first was erased and then added
			// and modified, or added, then modified and then erased. I don't know what they were
			// thinking by doing this...
			efDEBUG( "Event in: %s - flags: 0x%x\n", event.Path.c_str(), event.Flags );

			if ( event.Flags & efswFSEventStreamEventFlagItemRenamed ) {
				if ( ( i + 1 < esize ) &&
					 ( events[i + 1].Flags & efswFSEventStreamEventFlagItemRenamed ) &&
					 ( events[i + 1].inode == event.inode ) ) {
					FSEvent& nEvent = events[i + 1];
					std::string newDir( FileSystem::pathRemoveFileName( nEvent.Path ) );
					std::string newFilepath( FileSystem::fileNameFromPath( nEvent.Path ) );

					if ( event.Path != nEvent.Path ) {
						if ( dirPath == newDir ) {
							if ( !FileInfo::exists( event.Path ) ||
								 0 == strcasecmp( event.Path.c_str(), nEvent.Path.c_str() ) ) {
								sendFileAction( ID, dirPath, newFilepath,
												efswFSEventStreamEventFlagItemIsDir & event.Flags,
												Actions::Moved, filePath );
							} else {
								sendFileAction( ID, dirPath, filePath,
												efswFSEventStreamEventFlagItemIsDir & event.Flags,
												Actions::Moved, newFilepath );
							}
						} else {
							sendFileAction( ID, dirPath, filePath,
											efswFSEventStreamEventFlagItemIsDir & event.Flags,
											Actions::Delete );
							sendFileAction( ID, newDir, newFilepath,
											efswFSEventStreamEventFlagItemIsDir & event.Flags,
											Actions::Add );

							if ( nEvent.Flags & ModifiedFlags ) {
								sendFileAction( ID, newDir, newFilepath,
												efswFSEventStreamEventFlagItemIsDir & event.Flags,
												Actions::Modified );
							}
						}
					} else {
						handleAddModDel( nEvent.Flags, nEvent.Path, dirPath, filePath, event.inode );
					}

					if ( nEvent.Flags & ( efswFSEventStreamEventFlagItemCreated |
										  efswFSEventStreamEventFlagItemRemoved |
										  efswFSEventStreamEventFlagItemRenamed ) ) {
						if ( newDir != Directory ) {
							DirsChanged.insert( newDir );
						}
					}

					// Skip the renamed file
					i++;
				} else if ( FileInfo::exists( event.Path ) ) {
					sendFileAction( ID, dirPath, filePath,
									efswFSEventStreamEventFlagItemIsDir & event.Flags,
									Actions::Add );

					if ( event.Flags & ModifiedFlags ) {
						sendFileAction( ID, dirPath, filePath,
										efswFSEventStreamEventFlagItemIsDir & event.Flags,
										Actions::Modified );
					}
				} else {
					sendFileAction( ID, dirPath, filePath,
									efswFSEventStreamEventFlagItemIsDir & event.Flags,
									Actions::Delete );
				}
			} else {
				handleAddModDel( event.Flags, event.Path, dirPath, filePath, event.inode );
			}
		} else {
			efDEBUG( "Directory: %s changed\n", event.Path.c_str() );
			DirsChanged.insert( event.Path );
		}
	}
}

void WatcherFSEvents::process() {
	std::unordered_set<std::string>::iterator it = DirsChanged.begin();

	for ( ; it != DirsChanged.end(); it++ ) {
		if ( !FileWatcherFSEvents::isGranular() ) {
			WatcherGen->watchDir( ( *it ) );
		} else {
			sendFileAction( ID, FileSystem::pathRemoveFileName( ( *it ) ),
							FileSystem::fileNameFromPath( ( *it ) ),
							FileSystem::isDirectory( ( *it ) ), Actions::Modified );
		}
	}

	DirsChanged.clear();
}

} // namespace efsw

#endif
