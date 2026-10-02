#ifndef EFSW_FILESYSTEMIMPLWIN_HPP
#define EFSW_FILESYSTEMIMPLWIN_HPP

#include <efsw/FileInfo.hpp>
#include <efsw/String.hpp>
#include <efsw/base.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

namespace efsw { namespace Platform {

class FileSystem {
  public:
	static FileInfoMap filesInfoFromPath( const std::string& path );

	static char getOSSlash();

	static bool isDirectory( const std::string& path );

	static bool isRemoteFS( const std::string& directory );

	static bool changeWorkingDirectory( const std::string& path );

	static std::string getCurrentWorkingDirectory();
};

}} // namespace efsw::Platform

#endif

#endif
