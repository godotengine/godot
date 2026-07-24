#ifndef EFSW_FILEWATCHEROSX_HPP
#define EFSW_FILEWATCHEROSX_HPP

#include <efsw/FileWatcherImpl.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_KQUEUE || EFSW_PLATFORM == EFSW_PLATFORM_FSEVENTS

#include <efsw/WatcherKqueue.hpp>

namespace efsw {

/// Implementation for OSX based on kqueue.
/// @class FileWatcherKqueue
class FileWatcherKqueue : public FileWatcherImpl {
	friend class WatcherKqueue;

  public:
	FileWatcherKqueue( FileWatcher* parent );

	virtual ~FileWatcherKqueue();

	/// Add a directory watch
	/// On error returns WatchID with Error type.
	WatchID addWatch( const std::string& directory, FileWatchListener* watcher, bool recursive,
					  const std::vector<WatcherOption> &options ) override;

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

	/// time out data
	struct timespec mTimeOut;

	/// WatchID allocator
	int mLastWatchID;

	Thread* mThread;

	Mutex mWatchesLock;

	std::vector<WatchID> mRemoveList;

	long mFileDescriptorCount;

	bool mAddingWatcher;

	bool isAddingWatcher() const;

	bool pathInWatches( const std::string& path ) override;

	void addFD();

	void removeFD();

	bool availablesFD();

  private:
	void run();
};

} // namespace efsw

#endif

#endif
