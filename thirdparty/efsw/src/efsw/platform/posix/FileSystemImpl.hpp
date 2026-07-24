#ifndef EFSW_FILESYSTEMIMPLPOSIX_HPP
#define EFSW_FILESYSTEMIMPLPOSIX_HPP

#include <efsw/FileInfo.hpp>
#include <efsw/base.hpp>

#if defined( EFSW_PLATFORM_POSIX )

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
