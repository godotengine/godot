#ifndef EFSW_WATCHERINOTIFY_HPP
#define EFSW_WATCHERINOTIFY_HPP

#include <efsw/FileWatcherImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS

#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>
#include <efsw/FileInfo.hpp>
#include <efsw/WatcherGeneric.hpp>
#include <unordered_set>
#include <vector>

namespace efsw {

/* OSX < 10.7 has no file events */
/* So i declare the events constants */
enum FSEventEvents {
	efswFSEventStreamCreateFlagUseCFTypes = 0x00000001,
	efswFSEventStreamCreateFlagNoDefer = 0x00000002,
	efswFSEventStreamCreateFlagFileEvents = 0x00000010,
	efswFSEventStreamCreateFlagUseExtendedData = 0x00000040,
	efswFSEventStreamEventFlagItemCreated = 0x00000100,
	efswFSEventStreamEventFlagItemRemoved = 0x00000200,
	efswFSEventStreamEventFlagItemInodeMetaMod = 0x00000400,
	efswFSEventStreamEventFlagItemRenamed = 0x00000800,
	efswFSEventStreamEventFlagItemModified = 0x00001000,
	efswFSEventStreamEventFlagItemFinderInfoMod = 0x00002000,
	efswFSEventStreamEventFlagItemChangeOwner = 0x00004000,
	efswFSEventStreamEventFlagItemXattrMod = 0x00008000,
	efswFSEventStreamEventFlagItemIsFile = 0x00010000,
	efswFSEventStreamEventFlagItemIsDir = 0x00020000,
	efswFSEventStreamEventFlagItemIsSymlink = 0x00040000,
	efswFSEventsModified = efswFSEventStreamEventFlagItemFinderInfoMod |
						   efswFSEventStreamEventFlagItemModified |
						   efswFSEventStreamEventFlagItemInodeMetaMod
};

class FileWatcherFSEvents;

class FSEvent {
  public:
	FSEvent( std::string path, long flags, Uint64 id, Uint64 inode = 0 ) :
		Path( path ), Flags( flags ), Id( id ), inode( inode ) {}

	std::string Path;
	long Flags{ 0 };
	Uint64 Id{ 0 };
	Uint64 inode{ 0 };
};

class WatcherFSEvents : public Watcher {
  public:
	WatcherFSEvents();

	~WatcherFSEvents();

	void init();

	void handleActions( std::vector<FSEvent>& events );

	void process();

	Atomic<FileWatcherFSEvents*> FWatcher;
	FSEventStreamRef FSStream;
	Uint64 ModifiedFlags{ efswFSEventsModified };
	bool SanitizeEvents{ false };

  protected:
	void handleAddModDel( const Uint32& flags, const std::string& path, std::string& dirPath,
						  std::string& filePath, Uint64 inode );

	WatcherGeneric* WatcherGen;

	std::unordered_set<std::string> DirsChanged;
	std::unordered_set<Uint64> FilesAdded;

	void sendFileAction( WatchID watchid, const std::string& dir, const std::string& filename,
						 bool isDir, Action action, std::string oldFilename = "" );

	void sendMissedFileActions( WatchID watchid, const std::string& dir);
};

} // namespace efsw

#endif

#endif
