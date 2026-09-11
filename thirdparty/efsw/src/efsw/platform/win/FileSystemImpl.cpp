#include <efsw/platform/win/FileSystemImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

#include <climits>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifndef EFSW_COMPILER_MSVC
#include <dirent.h>
#else
#include <direct.h>
#endif

namespace efsw { namespace Platform {

bool FileSystem::changeWorkingDirectory( const std::string& path ) {
	int res;
#ifdef EFSW_COMPILER_MSVC
#ifdef UNICODE
	res = _wchdir( String::fromUtf8( path.c_str() ).toWideString().c_str() );
#else
	res = _chdir( String::fromUtf8( path.c_str() ).toAnsiString().c_str() );
#endif
#else
	res = chdir( path.c_str() );
#endif
	return -1 != res;
}

std::string FileSystem::getCurrentWorkingDirectory() {
#ifdef EFSW_COMPILER_MSVC
#if defined( UNICODE ) && !defined( EFSW_NO_WIDECHAR )
	wchar_t dir[_MAX_PATH];
	return ( 0 != GetCurrentDirectoryW( _MAX_PATH, dir ) ) ? String( dir ).toUtf8() : std::string();
#else
	char dir[_MAX_PATH];
	return ( 0 != GetCurrentDirectory( _MAX_PATH, dir ) ) ? String( dir, std::locale() ).toUtf8()
														  : std::string();
#endif
#else
	char dir[PATH_MAX + 1];
	getcwd( dir, PATH_MAX + 1 );
	return std::string( dir );
#endif
}

FileInfoMap FileSystem::filesInfoFromPath( const std::string& path ) {
	FileInfoMap files;

	String tpath( path );

	if ( tpath[tpath.size() - 1] == '/' || tpath[tpath.size() - 1] == '\\' ) {
		tpath += "*";
	} else {
		tpath += "\\*";
	}

	WIN32_FIND_DATAW findFileData;
	HANDLE hFind = FindFirstFileW( (LPCWSTR)tpath.toWideString().c_str(), &findFileData );

	if ( hFind != INVALID_HANDLE_VALUE ) {
		std::string name( String( findFileData.cFileName ).toUtf8() );
		std::string fpath( path + name );

		if ( name != "." && name != ".." ) {
			files[name] = FileInfo( fpath );
		}

		while ( FindNextFileW( hFind, &findFileData ) ) {
			name = String( findFileData.cFileName ).toUtf8();
			fpath = path + name;

			if ( name != "." && name != ".." ) {
				files[name] = FileInfo( fpath );
			}
		}

		FindClose( hFind );
	}

	return files;
}

char FileSystem::getOSSlash() {
	return '\\';
}

bool FileSystem::isDirectory( const std::string& path ) {
	DWORD attrs = GetFileAttributesW( String( path ).toWideString().c_str() );
	return attrs != INVALID_FILE_ATTRIBUTES && ( attrs & FILE_ATTRIBUTE_DIRECTORY ) != 0;
}

bool FileSystem::isRemoteFS( const std::string& directory ) {
	if ( ( directory[0] == '\\' || directory[0] == '/' ) &&
		 ( directory[1] == '\\' || directory[1] == '/' ) ) {
		return true;
	}

	if ( directory.size() >= 3 ) {
		return 4 == GetDriveTypeA( directory.substr( 0, 3 ).c_str() );
	}

	return false;
}

}} // namespace efsw::Platform

#endif
