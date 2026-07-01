#ifndef EFSW_WATCHEROSX_HPP
#define EFSW_WATCHEROSX_HPP

#include <efsw/FileWatcherImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_KQUEUE || EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS

#include <efsw/DirectorySnapshot.hpp>
#include <map>
#include <sys/event.h>
#include <sys/types.h>
#include <vector>

namespace efsw {

class FileWatcherKqueue;
class WatcherKqueue;

typedef struct kevent KEvent;

/// type for a map from WatchID to WatcherKqueue pointer
typedef std::map<WatchID, Watcher*> WatchMap;

class WatcherKqueue : public Watcher {
  public:
	WatcherKqueue( WatchID watchid, const std::string& dirname, FileWatchListener* listener,
				   bool recursive, FileWatcherKqueue* watcher, WatcherKqueue* parent = NULL );

	virtual ~WatcherKqueue();

	void addFile( const std::string& name, bool emitEvents = true );

	void removeFile( const std::string& name, bool emitEvents = true );

	// called when the directory is actually changed
	// means a file has been added or removed
	// rescans the watched directory adding/removing files and sending notices
	void rescan();

	void handleAction( const std::string& filename, bool isDir, efsw::Action action,
					   const std::string& oldFilename = "" );

	void handleFolderAction( std::string filename, efsw::Action action,
							 const std::string& oldFilename = "" );

	void addAll();

	void removeAll();

	WatchID watchingDirectory( std::string dir );

	void watch() override;

	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive,
					  WatcherKqueue* parent );

	void removeWatch( WatchID watchid );

	bool initOK();

	int lastErrno();

  protected:
	WatchMap mWatches;
	int mLastWatchID;

	// index 0 is always the directory
	std::vector<KEvent> mChangeList;
	size_t mChangeListCount;
	DirectorySnapshot mDirSnap;

	/// The descriptor for the kqueue
	int mKqueue;

	FileWatcherKqueue* mWatcher;

	WatcherKqueue* mParent;

	bool mInitOK;
	int mErrno;

	bool pathInWatches( const std::string& path );

	bool pathInParent( const std::string& path );

	Watcher* findWatcher( const std::string path );

	void moveDirectory( std::string oldPath, std::string newPath, bool emitEvents = true );

	void sendDirChanged();
};

} // namespace efsw

#endif

#endif
