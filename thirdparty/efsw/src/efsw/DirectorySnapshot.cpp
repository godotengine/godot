#include <efsw/DirectorySnapshot.hpp>
#include <efsw/FileSystem.hpp>

namespace efsw {

DirectorySnapshot::DirectorySnapshot() {}

DirectorySnapshot::DirectorySnapshot( std::string directory ) {
	init( directory );
}

DirectorySnapshot::~DirectorySnapshot() {}

void DirectorySnapshot::init( std::string directory ) {
	setDirectoryInfo( directory );
	initFiles();
}

bool DirectorySnapshot::exists() {
	return DirectoryInfo.exists();
}

void DirectorySnapshot::deleteAll( DirectorySnapshotDiff& Diff ) {
	FileInfo fi;

	for ( FileInfoMap::iterator it = Files.begin(); it != Files.end(); it++ ) {
		fi = it->second;

		if ( fi.isDirectory() ) {
			Diff.DirsDeleted.push_back( fi );
		} else {
			Diff.FilesDeleted.push_back( fi );
		}
	}

	Files.clear();
}

void DirectorySnapshot::setDirectoryInfo( std::string directory ) {
	DirectoryInfo = FileInfo( directory );
}

void DirectorySnapshot::initFiles() {
	Files = FileSystem::filesInfoFromPath( DirectoryInfo.Filepath );

	FileInfoMap::iterator it = Files.begin();
	std::vector<std::string> eraseFiles;

	/// Remove all non regular files and non directories
	for ( ; it != Files.end(); it++ ) {
		if ( !it->second.isRegularFile() && !it->second.isDirectory() ) {
			eraseFiles.push_back( it->first );
		}
	}

	for ( std::vector<std::string>::iterator eit = eraseFiles.begin(); eit != eraseFiles.end();
		  eit++ ) {
		Files.erase( *eit );
	}
}

DirectorySnapshotDiff DirectorySnapshot::scan() {
	DirectorySnapshotDiff Diff;

	Diff.clear();

	FileInfo curFI( DirectoryInfo.Filepath );

	Diff.DirChanged = DirectoryInfo != curFI;

	if ( Diff.DirChanged ) {
		DirectoryInfo = curFI;
	}

	/// If the directory was erased, create the events for files and directories deletion
	if ( !curFI.exists() ) {
		deleteAll( Diff );

		return Diff;
	}

	FileInfoMap files = FileSystem::filesInfoFromPath( DirectoryInfo.Filepath );

	if ( files.empty() && Files.empty() ) {
		return Diff;
	}

	FileInfo fi;
	FileInfoMap FilesCpy;
	FileInfoMap::iterator it;
	FileInfoMap::iterator fiIt;

	if ( Diff.DirChanged ) {
		FilesCpy = Files;
	}

	for ( it = files.begin(); it != files.end(); it++ ) {
		fi = it->second;

		/// File existed before?
		fiIt = Files.find( it->first );

		if ( fiIt != Files.end() ) {
			/// Erase from the file list copy
			FilesCpy.erase( it->first );

			/// File changed?
			if ( ( *fiIt ).second != fi ) {
				/// Update the new file info
				Files[it->first] = fi;

				/// handle modified event
				if ( fi.isDirectory() ) {
					Diff.DirsModified.push_back( fi );
				} else {
					Diff.FilesModified.push_back( fi );
				}
			}
		}
		/// Only add regular files or directories
		else if ( fi.isRegularFile() || fi.isDirectory() ) {
			/// New file found
			Files[it->first] = fi;

			FileInfoMap::iterator fit;
			std::string oldFile = "";

			/// Check if the same inode already existed
			if ( ( fit = nodeInFiles( fi ) ) != Files.end() ) {
				oldFile = fit->first;

				/// Avoid firing a Delete event
				FilesCpy.erase( fit->first );

				/// Delete the old file name
				Files.erase( fit->first );

				if ( fi.isDirectory() ) {
					Diff.DirsMoved.push_back( std::make_pair( oldFile, fi ) );
				} else {
					Diff.FilesMoved.push_back( std::make_pair( oldFile, fi ) );
				}
			} else {
				if ( fi.isDirectory() ) {
					Diff.DirsCreated.push_back( fi );
				} else {
					Diff.FilesCreated.push_back( fi );
				}
			}
		}
	}

	if ( !Diff.DirChanged ) {
		return Diff;
	}

	/// The files or directories that remains were deleted
	for ( it = FilesCpy.begin(); it != FilesCpy.end(); it++ ) {
		fi = it->second;

		if ( fi.isDirectory() ) {
			Diff.DirsDeleted.push_back( fi );
		} else {
			Diff.FilesDeleted.push_back( fi );
		}

		/// Remove the file or directory from the list of files
		Files.erase( it->first );
	}

	return Diff;
}

FileInfoMap::iterator DirectorySnapshot::nodeInFiles( FileInfo& fi ) {
	FileInfoMap::iterator it;

	if ( FileInfo::inodeSupported() ) {
		for ( it = Files.begin(); it != Files.end(); it++ ) {
			if ( it->second.sameInode( fi ) && it->second.Filepath != fi.Filepath ) {
				return it;
			}
		}
	}

	return Files.end();
}

void DirectorySnapshot::addFile( std::string path ) {
	std::string name( FileSystem::fileNameFromPath( path ) );
	Files[name] = FileInfo( path );
}

void DirectorySnapshot::removeFile( std::string path ) {
	std::string name( FileSystem::fileNameFromPath( path ) );

	FileInfoMap::iterator it = Files.find( name );

	if ( Files.end() != it ) {
		Files.erase( it );
	}
}

void DirectorySnapshot::moveFile( std::string oldPath, std::string newPath ) {
	removeFile( oldPath );
	addFile( newPath );
}

void DirectorySnapshot::updateFile( std::string path ) {
	addFile( path );
}

} // namespace efsw
