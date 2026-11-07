#ifndef EFSW_FILEWATCHERLINUX_HPP
#define EFSW_FILEWATCHERLINUX_HPP

#include <efsw/FileWatcherImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_INOTIFY

#include <efsw/WatcherInotify.hpp>
#include <map>
#include <unordered_map>
#include <vector>

namespace efsw {

/// Implementation for Linux based on inotify.
/// @class FileWatcherInotify
class FileWatcherInotify : public FileWatcherImpl {
  public:
	/// type for a map from WatchID to WatchStruct pointer
	typedef std::map<WatchID, WatcherInotify*> WatchMap;

	FileWatcherInotify( FileWatcher* parent );

	virtual ~FileWatcherInotify();

	/// Add a directory watch
	/// On error returns WatchID with Error type.
	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive,
					  const std::vector<WatcherOption>& options ) override;

	/// Remove a directory watch. This is a brute force lazy search O(nlogn).
	void removeWatch( const std::string& directory ) override;

	/// Remove a directory watch. This is a map lookup O(logn).
	void removeWatch( WatchID watchid ) override;

	/// Updates the watcher. Must be called often.
	void watch() override;

	/// Handles the action
	void handleAction( Watcher* watch, const std::string& filename, bool isDir,
					   unsigned long action, std::string oldFilename = "" ) override;

	/// @return Returns a list of the directories that are being watched
	std::vector<std::string> directories() override;

  protected:
	/// Map of WatchID to WatchStruct pointers
	WatchMap mWatches;

	/// User added watches
	WatchMap mRealWatches;

	std::unordered_map<std::string, WatchID> mWatchesRef;

	/// inotify file descriptor
	int mFD;

	Thread* mThread;

	Mutex mWatchesLock;
	Mutex mRealWatchesLock;
	Mutex mInitLock;
	bool mIsTakingAction;
	std::vector<std::pair<WatcherInotify*, std::string>> mMovedOutsideWatches;

	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive,
					  bool syntheticEvents, WatcherInotify* parent = NULL,
					  bool fromInternalEvent = false );

	bool pathInWatches( const std::string& path ) override;

  private:
	void run();

	void removeWatchLocked( WatchID watchid, bool skipInotifyRemove = false );

	void checkForNewWatcher( Watcher* watch, std::string fpath );

	Watcher* watcherContainsDirectory( std::string dir );
};

} // namespace efsw

#endif

#endif
