#ifndef EFSW_FILEINFO_HPP
#define EFSW_FILEINFO_HPP

#include <efsw/base.hpp>
#include <vector>
#include <map>
#include <string>

namespace efsw {

class FileInfo {
  public:
	static bool exists( const std::string& filePath );

	static bool isLink( const std::string& filePath );

	static bool inodeSupported();

	FileInfo();

	FileInfo( const std::string& filepath );

	FileInfo( const std::string& filepath, bool linkInfo );

	bool operator==( const FileInfo& Other ) const;

	bool operator!=( const FileInfo& Other ) const;

	FileInfo& operator=( const FileInfo& Other );

	bool isDirectory() const;

	bool isRegularFile() const;

	bool isReadable() const;

	bool sameInode( const FileInfo& Other ) const;

	bool isLink() const;

	std::string linksTo();

	bool exists();

	void getInfo();

	void getRealInfo();

	std::string Filepath;
	Uint64 ModificationTime;
	Uint64 Size;
	Uint32 OwnerId;
	Uint32 GroupId;
	Uint32 Permissions;
	Uint64 Inode;
};

typedef std::map<std::string, FileInfo> FileInfoMap;
typedef std::vector<FileInfo> FileInfoList;
typedef std::vector<std::pair<std::string, FileInfo>> MovedList;

} // namespace efsw

#endif
