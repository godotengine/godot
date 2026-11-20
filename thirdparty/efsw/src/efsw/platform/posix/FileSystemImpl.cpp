#include <efsw/platform/posix/FileSystemImpl.hpp>

#if defined( EFSW_PLATFORM_POSIX )

#include <cstring>
#include <dirent.h>
#include <efsw/FileInfo.hpp>
#include <efsw/FileSystem.hpp>
#include <unistd.h>

#ifndef _DARWIN_FEATURE_64_BIT_INODE
#define _DARWIN_FEATURE_64_BIT_INODE
#endif

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <climits>
#include <cstdlib>
#include <sys/stat.h>

#if EFSW_OS == EFSW_OS_LINUX || EFSW_OS == EFSW_OS_SOLARIS || EFSW_OS == EFSW_OS_ANDROID
#include <sys/vfs.h>
#elif EFSW_OS == EFSW_OS_MACOSX || EFSW_OS == EFSW_OS_BSD || EFSW_OS == EFSW_OS_IOS
#include <sys/mount.h>
#include <sys/param.h>
#endif

/** Remote file systems codes */
#define S_MAGIC_AFS 0x5346414F
#define S_MAGIC_AUFS 0x61756673
#define S_MAGIC_CEPH 0x00C36400
#define S_MAGIC_CIFS 0xFF534D42
#define S_MAGIC_CODA 0x73757245
#define S_MAGIC_FHGFS 0x19830326
#define S_MAGIC_FUSEBLK 0x65735546
#define S_MAGIC_FUSECTL 0x65735543
#define S_MAGIC_GFS 0x01161970
#define S_MAGIC_GPFS 0x47504653
#define S_MAGIC_KAFS 0x6B414653
#define S_MAGIC_LUSTRE 0x0BD00BD0
#define S_MAGIC_NCP 0x564C
#define S_MAGIC_NFS 0x6969
#define S_MAGIC_NFSD 0x6E667364
#define S_MAGIC_OCFS2 0x7461636F
#define S_MAGIC_PANFS 0xAAD7AAEA
#define S_MAGIC_PIPEFS 0x50495045
#define S_MAGIC_SMB 0x517B
#define S_MAGIC_SNFS 0xBEEFDEAD
#define S_MAGIC_VMHGFS 0xBACBACBC
#define S_MAGIC_VXFS 0xA501FCF5

#if EFSW_OS == EFSW_OS_LINUX
#include <cstdio>
#include <mntent.h>
#endif

namespace efsw { namespace Platform {

#if EFSW_OS == EFSW_OS_LINUX

std::string findMountPoint( std::string file ) {
	std::string cwd = FileSystem::getCurrentWorkingDirectory();
	struct stat last_stat;
	struct stat file_stat;

	stat( file.c_str(), &file_stat );

	std::string mp;

	if ( efsw::FileSystem::isDirectory( file ) ) {
		last_stat = file_stat;

		if ( !FileSystem::changeWorkingDirectory( file ) )
			return "";
	} else {
		std::string dir = efsw::FileSystem::pathRemoveFileName( file );

		if ( !FileSystem::changeWorkingDirectory( dir ) )
			return "";

		if ( stat( ".", &last_stat ) < 0 )
			return "";
	}

	while ( true ) {
		struct stat st;

		if ( stat( "..", &st ) < 0 )
			goto done;

		if ( st.st_dev != last_stat.st_dev || st.st_ino == last_stat.st_ino )
			break;

		if ( !FileSystem::changeWorkingDirectory( ".." ) ) {
			goto done;
		}

		last_stat = st;
	}

	/* Finally reached a mount point, see what it's called.  */
	mp = FileSystem::getCurrentWorkingDirectory();

done:
	FileSystem::changeWorkingDirectory( cwd );

	return mp;
}

std::string findDevicePath( const std::string& directory ) {
	struct mntent* ent;
	FILE* aFile;

	aFile = setmntent( "/proc/mounts", "r" );

	if ( aFile == NULL )
		return "";

	while ( NULL != ( ent = getmntent( aFile ) ) ) {
		std::string dirName( ent->mnt_dir );

		if ( dirName == directory ) {
			std::string fsName( ent->mnt_fsname );

			endmntent( aFile );

			return fsName;
		}
	}

	endmntent( aFile );

	return "";
}

bool isLocalFUSEDirectory( std::string directory ) {
	efsw::FileSystem::dirRemoveSlashAtEnd( directory );

	directory = findMountPoint( directory );

	if ( !directory.empty() ) {
		std::string devicePath = findDevicePath( directory );

		return !devicePath.empty();
	}

	return false;
}

#endif

bool FileSystem::changeWorkingDirectory( const std::string& path ) {
	return -1 != chdir( path.c_str() );
}

std::string FileSystem::getCurrentWorkingDirectory() {
	char dir[PATH_MAX + 1];
	char* result = getcwd( dir, PATH_MAX + 1 );
	return result != NULL ? std::string( result ) : std::string();
}

FileInfoMap FileSystem::filesInfoFromPath( const std::string& path ) {
	FileInfoMap files;

	DIR* dp;
	struct dirent* dirp;

	if ( ( dp = opendir( path.c_str() ) ) == NULL )
		return files;

	while ( ( dirp = readdir( dp ) ) != NULL ) {
		if ( strcmp( dirp->d_name, ".." ) != 0 && strcmp( dirp->d_name, "." ) != 0 ) {
			std::string name( dirp->d_name );
			std::string fpath( path + name );

			files[name] = FileInfo( fpath );
		}
	}

	closedir( dp );

	return files;
}

char FileSystem::getOSSlash() {
	return '/';
}

bool FileSystem::isDirectory( const std::string& path ) {
	struct stat st;
	int res = stat( path.c_str(), &st );

	if ( 0 == res ) {
		return static_cast<bool>( S_ISDIR( st.st_mode ) );
	}

	return false;
}

bool FileSystem::isRemoteFS( const std::string& directory ) {
#if EFSW_OS == EFSW_OS_LINUX || EFSW_OS == EFSW_OS_MACOSX || EFSW_OS == EFSW_OS_BSD || \
	EFSW_OS == EFSW_OS_SOLARIS || EFSW_OS == EFSW_OS_ANDROID || EFSW_OS == EFSW_OS_IOS
	struct statfs statfsbuf;

	statfs( directory.c_str(), &statfsbuf );

	switch ( statfsbuf.f_type | 0UL ) {
		case S_MAGIC_FUSEBLK: /* 0x65735546 remote */
		{
#if EFSW_OS == EFSW_OS_LINUX
			return !isLocalFUSEDirectory( directory );
#endif
		}
		case S_MAGIC_AFS:	  /* 0x5346414F remote */
		case S_MAGIC_AUFS:	  /* 0x61756673 remote */
		case S_MAGIC_CEPH:	  /* 0x00C36400 remote */
		case S_MAGIC_CIFS:	  /* 0xFF534D42 remote */
		case S_MAGIC_CODA:	  /* 0x73757245 remote */
		case S_MAGIC_FHGFS:	  /* 0x19830326 remote */
		case S_MAGIC_FUSECTL: /* 0x65735543 remote */
		case S_MAGIC_GFS:	  /* 0x01161970 remote */
		case S_MAGIC_GPFS:	  /* 0x47504653 remote */
		case S_MAGIC_KAFS:	  /* 0x6B414653 remote */
		case S_MAGIC_LUSTRE:  /* 0x0BD00BD0 remote */
		case S_MAGIC_NCP:	  /* 0x564C remote */
		case S_MAGIC_NFS:	  /* 0x6969 remote */
		case S_MAGIC_NFSD:	  /* 0x6E667364 remote */
		case S_MAGIC_OCFS2:	  /* 0x7461636F remote */
		case S_MAGIC_PANFS:	  /* 0xAAD7AAEA remote */
		case S_MAGIC_PIPEFS:  /* 0x50495045 remote */
		case S_MAGIC_SMB:	  /* 0x517B remote */
		case S_MAGIC_SNFS:	  /* 0xBEEFDEAD remote */
		case S_MAGIC_VMHGFS:  /* 0xBACBACBC remote */
		case S_MAGIC_VXFS:	  /* 0xA501FCF5 remote */
		{
			return true;
		}
		default: {
			return false;
		}
	}
#endif

	return false;
}

}} // namespace efsw::Platform

#endif
