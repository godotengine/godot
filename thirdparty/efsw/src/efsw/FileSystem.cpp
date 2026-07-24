#include <cstring>
#include <efsw/FileSystem.hpp>
#include <efsw/platform/platformimpl.hpp>
#include <climits>
#include <cstdlib>

#if EFSW_OS == EFSW_OS_MACOSX
#include <CoreFoundation/CoreFoundation.h>
#endif

#if EFSW_OS == EFSW_OS_WIN
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace efsw {

bool FileSystem::isDirectory( const std::string& path ) {
	return Platform::FileSystem::isDirectory( path );
}

FileInfoMap FileSystem::filesInfoFromPath( std::string path ) {
	dirAddSlashAtEnd( path );

	return Platform::FileSystem::filesInfoFromPath( path );
}

char FileSystem::getOSSlash() {
	return Platform::FileSystem::getOSSlash();
}

bool FileSystem::slashAtEnd( std::string& dir ) {
	return ( dir.size() && dir[dir.size() - 1] == getOSSlash() );
}

void FileSystem::dirAddSlashAtEnd( std::string& dir ) {
	if ( dir.size() >= 1 && dir[dir.size() - 1] != getOSSlash() ) {
		dir.push_back( getOSSlash() );
	}
}

void FileSystem::dirRemoveSlashAtEnd( std::string& dir ) {
	if ( dir.size() >= 1 && dir[dir.size() - 1] == getOSSlash() ) {
		dir.erase( dir.size() - 1 );
	}
}

std::string FileSystem::fileNameFromPath( std::string filepath ) {
	dirRemoveSlashAtEnd( filepath );

	size_t pos = filepath.find_last_of( getOSSlash() );

	if ( pos != std::string::npos ) {
		return filepath.substr( pos + 1 );
	}

	return filepath;
}

std::string FileSystem::pathRemoveFileName( std::string filepath ) {
	dirRemoveSlashAtEnd( filepath );

	size_t pos = filepath.find_last_of( getOSSlash() );

	if ( pos != std::string::npos ) {
		return filepath.substr( 0, pos + 1 );
	}

	return filepath;
}

std::string FileSystem::getLinkRealPath( std::string dir, std::string& curPath ) {
	FileSystem::dirRemoveSlashAtEnd( dir );
	FileInfo fi( dir, true );

	/// Check with lstat and see if it's a link
	if ( fi.isLink() ) {
		/// get the real path of the link
		std::string link( fi.linksTo() );

		/// get the current path of the directory without the link dir path
		curPath = FileSystem::pathRemoveFileName( dir );

		/// ensure that ends with the os directory slash
		FileSystem::dirAddSlashAtEnd( link );

		return link;
	}

	/// if it's not a link return nothing
	return "";
}

std::string FileSystem::precomposeFileName( const std::string& name ) {
#if EFSW_OS == EFSW_OS_MACOSX
	CFStringRef cfStringRef =
		CFStringCreateWithCString( kCFAllocatorDefault, name.c_str(), kCFStringEncodingUTF8 );
	CFMutableStringRef cfMutable = CFStringCreateMutableCopy( NULL, 0, cfStringRef );

	CFStringNormalize( cfMutable, kCFStringNormalizationFormC );

	const char* c_str = CFStringGetCStringPtr( cfMutable, kCFStringEncodingUTF8 );
	if ( c_str != NULL ) {
		std::string result( c_str );
		CFRelease( cfStringRef );
		CFRelease( cfMutable );
		return result;
	}
	CFIndex length = CFStringGetLength( cfMutable );
	CFIndex maxSize = CFStringGetMaximumSizeForEncoding( length, kCFStringEncodingUTF8 );
	if ( maxSize == kCFNotFound ) {
		CFRelease( cfStringRef );
		CFRelease( cfMutable );
		return std::string();
	}

	std::string result( maxSize + 1, '\0' );
	if ( CFStringGetCString( cfMutable, &result[0], result.size(), kCFStringEncodingUTF8 ) ) {
		result.resize( std::strlen( result.c_str() ) );
		CFRelease( cfStringRef );
		CFRelease( cfMutable );
	} else {
		result.clear();
	}
	return result;
#else
	return name;
#endif
}

bool FileSystem::isRemoteFS( const std::string& directory ) {
	return Platform::FileSystem::isRemoteFS( directory );
}

bool FileSystem::changeWorkingDirectory( const std::string& directory ) {
	return Platform::FileSystem::changeWorkingDirectory( directory );
}

std::string FileSystem::getCurrentWorkingDirectory() {
	return Platform::FileSystem::getCurrentWorkingDirectory();
}

std::string FileSystem::getRealPath( const std::string& path ) {
	std::string realPath;
#if defined( EFSW_PLATFORM_POSIX )
	char dir[PATH_MAX];
	realpath( path.c_str(), &dir[0] );
	realPath = std::string( dir );
#elif EFSW_OS == EFSW_OS_WIN
	wchar_t dir[_MAX_PATH + 1];
	GetFullPathNameW( String::fromUtf8( path ).toWideString().c_str(), _MAX_PATH, &dir[0],
					  nullptr );
	realPath = String( dir ).toUtf8();
#else
#warning FileSystem::getRealPath() not implemented on this platform.
#endif
	return realPath;
}

} // namespace efsw
