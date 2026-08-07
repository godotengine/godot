#include <efsw/String.hpp>
#include <efsw/platform/win/SystemImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cstdlib>
#include <windows.h>

namespace efsw { namespace Platform {

void System::sleep( const unsigned long& ms ) {
	::Sleep( ms );
}

std::string System::getProcessPath() {
	// Get path to executable:
	WCHAR szDrive[_MAX_DRIVE];
	WCHAR szDir[_MAX_DIR];
	WCHAR szFilename[_MAX_DIR];
	WCHAR szExt[_MAX_DIR];
	std::wstring dllName( _MAX_DIR, 0 );

	GetModuleFileNameW( 0, &dllName[0], _MAX_PATH );

#ifdef EFSW_COMPILER_MSVC
	_wsplitpath_s( dllName.c_str(), szDrive, _MAX_DRIVE, szDir, _MAX_DIR, szFilename, _MAX_DIR,
				   szExt, _MAX_DIR );
#else
	_wsplitpath( dllName.c_str(), szDrive, szDir, szFilename, szExt );
#endif

	return String( szDrive ).toUtf8() + String( szDir ).toUtf8();
}

void System::maxFD() {}

Uint64 System::getMaxFD() { // Number of ReadDirectory per thread
	return 60;
}

}} // namespace efsw::Platform

#endif
