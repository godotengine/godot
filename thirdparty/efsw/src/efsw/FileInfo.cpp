#include <efsw/FileInfo.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/String.hpp>

#ifndef _DARWIN_FEATURE_64_BIT_INODE
#define _DARWIN_FEATURE_64_BIT_INODE
#endif

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <sys/stat.h>

#include <limits.h>
#include <stdlib.h>

#ifdef EFSW_COMPILER_MSVC
#ifndef S_ISDIR
#define S_ISDIR( f ) ( (f)&_S_IFDIR )
#endif

#ifndef S_ISREG
#define S_ISREG( f ) ( (f)&_S_IFREG )
#endif

#ifndef S_ISRDBL
#define S_ISRDBL( f ) ( (f)&_S_IREAD )
#endif
#else
#include <unistd.h>

#ifndef S_ISRDBL
#define S_ISRDBL( f ) ( (f)&S_IRUSR )
#endif
#endif

namespace efsw {

bool FileInfo::exists( const std::string& filePath ) {
	FileInfo fi( filePath );
	return fi.exists();
}

bool FileInfo::isLink( const std::string& filePath ) {
	FileInfo fi( filePath, true );
	return fi.isLink();
}

bool FileInfo::inodeSupported() {
#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	return true;
#else
	return false;
#endif
}

FileInfo::FileInfo() :
	ModificationTime( 0 ), OwnerId( 0 ), GroupId( 0 ), Permissions( 0 ), Inode( 0 ) {}

FileInfo::FileInfo( const std::string& filepath ) :
	Filepath( filepath ),
	ModificationTime( 0 ),
	OwnerId( 0 ),
	GroupId( 0 ),
	Permissions( 0 ),
	Inode( 0 ) {
	getInfo();
}

FileInfo::FileInfo( const std::string& filepath, bool linkInfo ) :
	Filepath( filepath ),
	ModificationTime( 0 ),
	OwnerId( 0 ),
	GroupId( 0 ),
	Permissions( 0 ),
	Inode( 0 ) {
	if ( linkInfo ) {
		getRealInfo();
	} else {
		getInfo();
	}
}

void FileInfo::getInfo() {
#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32
	if ( Filepath.size() == 3 && Filepath[1] == ':' && Filepath[2] == FileSystem::getOSSlash() ) {
		Filepath += FileSystem::getOSSlash();
	}
#endif

	/// Why i'm doing this? stat in mingw32 doesn't work for directories if the dir path ends with a
	/// path slash
	bool slashAtEnd = FileSystem::slashAtEnd( Filepath );

	if ( slashAtEnd ) {
		FileSystem::dirRemoveSlashAtEnd( Filepath );
	}

#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	struct stat st;
	int res = stat( Filepath.c_str(), &st );
#else
	struct _stat st;
	int res = _wstat( String::fromUtf8( Filepath ).toWideString().c_str(), &st );
#endif

	if ( 0 == res ) {
		ModificationTime = st.st_mtime;
		Size = st.st_size;
		OwnerId = st.st_uid;
		GroupId = st.st_gid;
		Permissions = st.st_mode;
		Inode = st.st_ino;
	}

	if ( slashAtEnd ) {
		FileSystem::dirAddSlashAtEnd( Filepath );
	}
}

void FileInfo::getRealInfo() {
	bool slashAtEnd = FileSystem::slashAtEnd( Filepath );

	if ( slashAtEnd ) {
		FileSystem::dirRemoveSlashAtEnd( Filepath );
	}

#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	struct stat st;
	int res = lstat( Filepath.c_str(), &st );
#else
	struct _stat st;
	int res = _wstat( String::fromUtf8( Filepath ).toWideString().c_str(), &st );
#endif

	if ( 0 == res ) {
		ModificationTime = st.st_mtime;
		Size = st.st_size;
		OwnerId = st.st_uid;
		GroupId = st.st_gid;
		Permissions = st.st_mode;
		Inode = st.st_ino;
	}

	if ( slashAtEnd ) {
		FileSystem::dirAddSlashAtEnd( Filepath );
	}
}

bool FileInfo::operator==( const FileInfo& Other ) const {
	return ( ModificationTime == Other.ModificationTime && Size == Other.Size &&
			 OwnerId == Other.OwnerId && GroupId == Other.GroupId &&
			 Permissions == Other.Permissions && Inode == Other.Inode );
}

bool FileInfo::isDirectory() const {
	return 0 != S_ISDIR( Permissions );
}

bool FileInfo::isRegularFile() const {
	return 0 != S_ISREG( Permissions );
}

bool FileInfo::isReadable() const {
#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	static bool isRoot = getuid() == 0;
	return isRoot || 0 != S_ISRDBL( Permissions );
#else
	return 0 != S_ISRDBL( Permissions );
#endif
}

bool FileInfo::isLink() const {
#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	return S_ISLNK( Permissions );
#else
	return false;
#endif
}

std::string FileInfo::linksTo() {
#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	if ( isLink() ) {
		char* ch = realpath( Filepath.c_str(), NULL );

		if ( NULL != ch ) {
			std::string tstr( ch );

			free( ch );

			return tstr;
		}
	}
#endif
	return std::string( "" );
}

bool FileInfo::exists() {
	bool slashAtEnd = FileSystem::slashAtEnd( Filepath );

	if ( slashAtEnd ) {
		FileSystem::dirRemoveSlashAtEnd( Filepath );
	}

#if EFSW_PLATFORM != EFSW_PLATFORM_WIN32
	struct stat st;
	int res = stat( Filepath.c_str(), &st );
#else
	struct _stat st;
	int res = _wstat( String::fromUtf8( Filepath ).toWideString().c_str(), &st );
#endif

	if ( slashAtEnd ) {
		FileSystem::dirAddSlashAtEnd( Filepath );
	}

	return 0 == res;
}

FileInfo& FileInfo::operator=( const FileInfo& Other ) {
	this->Filepath = Other.Filepath;
	this->Size = Other.Size;
	this->ModificationTime = Other.ModificationTime;
	this->GroupId = Other.GroupId;
	this->OwnerId = Other.OwnerId;
	this->Permissions = Other.Permissions;
	this->Inode = Other.Inode;
	return *this;
}

bool FileInfo::sameInode( const FileInfo& Other ) const {
	return inodeSupported() && Inode == Other.Inode;
}

bool FileInfo::operator!=( const FileInfo& Other ) const {
	return !( *this == Other );
}

} // namespace efsw
