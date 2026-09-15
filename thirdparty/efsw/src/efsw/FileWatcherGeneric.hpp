#ifndef EFSW_FILEWATCHERGENERIC_HPP
#define EFSW_FILEWATCHERGENERIC_HPP

#include <efsw/DirWatcherGeneric.hpp>
#include <efsw/FileWatcherImpl.hpp>
#include <efsw/WatcherGeneric.hpp>
#include <vector>

namespace efsw {

/// Implementation for Generic File Watcher.
/// @class FileWatcherGeneric
class FileWatcherGeneric : public FileWatcherImpl {
  public:
	typedef std::vector<WatcherGeneric*> WatchList;

	FileWatcherGeneric( FileWatcher* parent );

	virtual ~FileWatcherGeneric();

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
	Thread* mThread;

	/// The last watchid
	WatchID mLastWatchID;

	/// Map of WatchID to WatchStruct pointers
	WatchList mWatches;

	Mutex mWatchesLock;

	bool pathInWatches( const std::string& path ) override;

  private:
	void run();
};

} // namespace efsw

#endif
