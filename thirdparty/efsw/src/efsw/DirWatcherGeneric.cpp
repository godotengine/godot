#include <efsw/Debug.hpp>
#include <efsw/DirWatcherGeneric.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/String.hpp>

namespace efsw {

DirWatcherGeneric::DirWatcherGeneric( DirWatcherGeneric* parent, WatcherGeneric* ws,
									  const std::string& directory, bool recursive,
									  bool reportNewFiles ) :
	Parent( parent ), Watch( ws ), Recursive( recursive ), Deleted( false ) {
	resetDirectory( directory );

	if ( !reportNewFiles ) {
		DirSnap.scan();
	} else {
		DirectorySnapshotDiff Diff = DirSnap.scan();

		if ( Diff.changed() ) {
			FileInfoList::iterator it;

			DiffIterator( FilesCreated ) {
				handleAction( ( *it ).Filepath, false, Actions::Add );
			}
		}
	}
}

DirWatcherGeneric::~DirWatcherGeneric() {
	/// If the directory was deleted mark the files as deleted
	if ( Deleted ) {
		DirectorySnapshotDiff Diff = DirSnap.scan();

		if ( !DirSnap.exists() ) {
			FileInfoList::iterator it;

			DiffIterator( FilesDeleted ) {
				handleAction( ( *it ).Filepath, false, Actions::Delete );
			}

			DiffIterator( DirsDeleted ) {
				handleAction( ( *it ).Filepath, true, Actions::Delete );
			}
		}
	}

	DirWatchMap::iterator it = Directories.begin();

	for ( ; it != Directories.end(); ++it ) {
		if ( Deleted ) {
			/// If the directory was deleted, mark the flag for file deletion
			it->second->Deleted = true;
		}

		efSAFE_DELETE( it->second );
	}
}

void DirWatcherGeneric::resetDirectory( std::string directory ) {
	std::string dir( directory );

	/// Is this a recursive watch?
	if ( Watch->Directory != directory ) {
		if ( !( directory.size() &&
				( directory.at( 0 ) == FileSystem::getOSSlash() ||
				  directory.at( directory.size() - 1 ) == FileSystem::getOSSlash() ) ) ) {
			/// Get the real directory
			if ( NULL != Parent ) {
				std::string parentPath( Parent->DirSnap.DirectoryInfo.Filepath );
				FileSystem::dirAddSlashAtEnd( parentPath );
				FileSystem::dirAddSlashAtEnd( directory );

				dir = parentPath + directory;
			} else {
				efDEBUG( "resetDirectory(): Parent is NULL. Fatal error." );
			}
		}
	}

	DirSnap.setDirectoryInfo( dir );
}

void DirWatcherGeneric::handleAction( const std::string& filename, bool isDir, unsigned long action,
									  std::string oldFilename ) {
	Watch->Listener->handleFileAction( Watch->ID, DirSnap.DirectoryInfo.Filepath,
									   FileSystem::fileNameFromPath( filename ), isDir,
									   (Action)action, oldFilename );
}

void DirWatcherGeneric::addChilds( bool reportNewFiles ) {
	if ( Recursive ) {
		/// Create the subdirectories watchers
		std::string dir;

		for ( FileInfoMap::iterator it = DirSnap.Files.begin(); it != DirSnap.Files.end(); it++ ) {
			if ( it->second.isDirectory() && it->second.isReadable() &&
				 !FileSystem::isRemoteFS( it->second.Filepath ) ) {
				/// Check if the directory is a symbolic link
				std::string curPath;
				std::string link( FileSystem::getLinkRealPath( it->second.Filepath, curPath ) );

				dir = it->first;

				if ( "" != link ) {
					/// Avoid adding symlinks directories if it's now enabled
					if ( !Watch->WatcherImpl->mFileWatcher->followSymlinks() ) {
						continue;
					}

					/// If it's a symlink check if the realpath exists as a watcher, or
					/// if the path is outside the current dir
					if ( Watch->WatcherImpl->pathInWatches( link ) ||
						 Watch->pathInWatches( link ) ||
						 !Watch->WatcherImpl->linkAllowed( curPath, link ) ) {
						continue;
					} else {
						dir = link;
					}
				} else {
					if ( Watch->pathInWatches( dir ) || Watch->WatcherImpl->pathInWatches( dir ) ) {
						continue;
					}
				}

				if ( reportNewFiles ) {
					handleAction( dir, true, Actions::Add );
				}

				Directories[dir] =
					new DirWatcherGeneric( this, Watch, dir, Recursive, reportNewFiles );

				Directories[dir]->addChilds( reportNewFiles );
			}
		}
	}
}

void DirWatcherGeneric::watch( bool reportOwnChange ) {
	DirectorySnapshotDiff Diff = DirSnap.scan();

	if ( reportOwnChange && Diff.DirChanged && NULL != Parent ) {
		Watch->Listener->handleFileAction(
			Watch->ID, FileSystem::pathRemoveFileName( DirSnap.DirectoryInfo.Filepath ),
			FileSystem::fileNameFromPath( DirSnap.DirectoryInfo.Filepath ), true,
			Actions::Modified );
	}

	if ( Diff.changed() ) {
		FileInfoList::iterator it;
		MovedList::iterator mit;

		/// Files
		DiffIterator( FilesCreated ) {
			handleAction( ( *it ).Filepath, false, Actions::Add );
		}

		DiffIterator( FilesModified ) {
			handleAction( ( *it ).Filepath, false, Actions::Modified );
		}

		DiffIterator( FilesDeleted ) {
			handleAction( ( *it ).Filepath, false, Actions::Delete );
		}

		DiffMovedIterator( FilesMoved ) {
			handleAction( ( *mit ).second.Filepath, false, Actions::Moved, ( *mit ).first );
		}

		/// Directories
		DiffIterator( DirsCreated ) {
			createDirectory( ( *it ).Filepath );
		}

		DiffIterator( DirsModified ) {
			handleAction( ( *it ).Filepath, true, Actions::Modified );
		}

		DiffIterator( DirsDeleted ) {
			handleAction( ( *it ).Filepath, true, Actions::Delete );
			removeDirectory( ( *it ).Filepath );
		}

		DiffMovedIterator( DirsMoved ) {
			handleAction( ( *mit ).second.Filepath, true, Actions::Moved, ( *mit ).first );
			moveDirectory( ( *mit ).first, ( *mit ).second.Filepath );
		}
	}

	/// Process the subdirectories looking for changes
	for ( DirWatchMap::iterator dit = Directories.begin(); dit != Directories.end(); ++dit ) {
		/// Just watch
		dit->second->watch();
	}
}

void DirWatcherGeneric::watchDir( std::string& dir ) {
	DirWatcherGeneric* watcher = Watch->WatcherImpl->mFileWatcher->allowOutOfScopeLinks()
									 ? findDirWatcher( dir )
									 : findDirWatcherFast( dir );

	if ( NULL != watcher ) {
		watcher->watch( true );
	}
}

DirWatcherGeneric* DirWatcherGeneric::findDirWatcherFast( std::string dir ) {
	// remove the common base ( dir should always start with the same base as the watcher )
	efASSERT( !dir.empty() );
	efASSERT( dir.size() >= DirSnap.DirectoryInfo.Filepath.size() );
	efASSERT( DirSnap.DirectoryInfo.Filepath ==
			  dir.substr( 0, DirSnap.DirectoryInfo.Filepath.size() ) );

	if ( dir.size() >= DirSnap.DirectoryInfo.Filepath.size() ) {
		dir = dir.substr( DirSnap.DirectoryInfo.Filepath.size() - 1 );
	}

	if ( dir.size() == 1 ) {
		efASSERT( dir[0] == FileSystem::getOSSlash() );
		return this;
	}

	size_t level = 0;
	std::vector<std::string> dirv = String::split( dir, FileSystem::getOSSlash(), false );

	DirWatcherGeneric* watcher = this;

	while ( level < dirv.size() ) {
		// search the dir level in the current watcher
		DirWatchMap::iterator it = watcher->Directories.find( dirv[level] );

		// found? continue with the next level
		if ( it != watcher->Directories.end() ) {
			watcher = it->second;

			level++;
		} else {
			// couldn't found the folder level?
			// directory not watched
			return NULL;
		}
	}

	return watcher;
}

DirWatcherGeneric* DirWatcherGeneric::findDirWatcher( std::string dir ) {
	if ( DirSnap.DirectoryInfo.Filepath == dir ) {
		return this;
	} else {
		DirWatcherGeneric* watcher = NULL;

		for ( DirWatchMap::iterator it = Directories.begin(); it != Directories.end(); ++it ) {
			watcher = it->second->findDirWatcher( dir );

			if ( NULL != watcher ) {
				return watcher;
			}
		}
	}

	return NULL;
}

DirWatcherGeneric* DirWatcherGeneric::createDirectory( std::string newdir ) {
	FileSystem::dirRemoveSlashAtEnd( newdir );
	newdir = FileSystem::fileNameFromPath( newdir );

	DirWatcherGeneric* dw = NULL;

	/// Check if the directory is a symbolic link
	std::string parentPath( DirSnap.DirectoryInfo.Filepath );
	FileSystem::dirAddSlashAtEnd( parentPath );
	std::string dir( parentPath + newdir );

	FileSystem::dirAddSlashAtEnd( dir );

	FileInfo fi( dir );

	if ( !fi.isDirectory() || !fi.isReadable() || FileSystem::isRemoteFS( dir ) ) {
		return NULL;
	}

	std::string curPath;
	std::string link( FileSystem::getLinkRealPath( dir, curPath ) );
	bool skip = false;

	if ( "" != link ) {
		/// Avoid adding symlinks directories if it's now enabled
		if ( !Watch->WatcherImpl->mFileWatcher->followSymlinks() ) {
			skip = true;
		}

		/// If it's a symlink check if the realpath exists as a watcher, or
		/// if the path is outside the current dir
		if ( Watch->WatcherImpl->pathInWatches( link ) || Watch->pathInWatches( link ) ||
			 !Watch->WatcherImpl->linkAllowed( curPath, link ) ) {
			skip = true;
		} else {
			dir = link;
		}
	} else {
		if ( Watch->pathInWatches( dir ) || Watch->WatcherImpl->pathInWatches( dir ) ) {
			skip = true;
		}
	}

	if ( !skip ) {
		handleAction( newdir, true, Actions::Add );

		/// Creates the new directory watcher of the subfolder and check for new files
		dw = new DirWatcherGeneric( this, Watch, dir, Recursive );

		dw->addChilds();

		dw->watch();

		/// Add it to the list of directories
		Directories[newdir] = dw;
	}

	return dw;
}

void DirWatcherGeneric::removeDirectory( std::string dir ) {
	FileSystem::dirRemoveSlashAtEnd( dir );
	dir = FileSystem::fileNameFromPath( dir );

	DirWatcherGeneric* dw = NULL;
	DirWatchMap::iterator dit;

	/// Folder deleted

	/// Search the folder, it should exists
	dit = Directories.find( dir );

	if ( dit != Directories.end() ) {
		dw = dit->second;

		/// Flag it as deleted so it fire the event for every file inside deleted
		dw->Deleted = true;

		/// Delete the DirWatcherGeneric
		efSAFE_DELETE( dw );

		/// Remove the directory from the map
		Directories.erase( dit->first );
	}
}

void DirWatcherGeneric::moveDirectory( std::string oldDir, std::string newDir ) {
	FileSystem::dirRemoveSlashAtEnd( oldDir );
	oldDir = FileSystem::fileNameFromPath( oldDir );

	FileSystem::dirRemoveSlashAtEnd( newDir );
	newDir = FileSystem::fileNameFromPath( newDir );

	DirWatcherGeneric* dw = NULL;
	DirWatchMap::iterator dit;

	/// Directory existed?
	dit = Directories.find( oldDir );

	if ( dit != Directories.end() ) {
		dw = dit->second;

		/// Remove the directory from the map
		Directories.erase( dit->first );

		Directories[newDir] = dw;

		dw->resetDirectory( newDir );
	}
}

bool DirWatcherGeneric::pathInWatches( std::string path ) {
	if ( DirSnap.DirectoryInfo.Filepath == path ) {
		return true;
	}

	for ( DirWatchMap::iterator it = Directories.begin(); it != Directories.end(); ++it ) {
		if ( it->second->pathInWatches( path ) ) {
			return true;
		}
	}

	return false;
}

} // namespace efsw
