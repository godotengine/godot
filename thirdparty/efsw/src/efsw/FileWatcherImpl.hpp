#ifndef EFSW_FILEWATCHERIMPL_HPP
#define EFSW_FILEWATCHERIMPL_HPP

#include <efsw/Atomic.hpp>
#include <efsw/Mutex.hpp>
#include <efsw/Thread.hpp>
#include <efsw/Watcher.hpp>
#include <efsw/base.hpp>
#include <efsw/efsw.hpp>

namespace efsw {

class FileWatcherImpl {
  public:
	FileWatcherImpl( FileWatcher* parent );

	virtual ~FileWatcherImpl();

	/// Add a directory watch
	/// On error returns WatchID with Error type.
	virtual WatchID addWatch( const std::string& directory, FileWatchListener* watcher,
							  bool recursive, const std::vector<WatcherOption>& options = {} ) = 0;

	/// Remove a directory watch. This is a brute force lazy search O(nlogn).
	virtual void removeWatch( const std::string& directory ) = 0;

	/// Remove a directory watch. This is a map lookup O(logn).
	virtual void removeWatch( WatchID watchid ) = 0;

	/// Updates the watcher. Must be called often.
	virtual void watch() = 0;

	/// Handles the action
	virtual void handleAction( Watcher* watch, const std::string& filename, bool isDir,
							   unsigned long action, std::string oldFilename = "" ) = 0;

	/// @return Returns a list of the directories that are being watched
	virtual std::vector<std::string> directories() = 0;

	/// @return true if the backend init successfully
	virtual bool initOK();

	/// @return If the link is allowed according to the current path and the state of out scope
	/// links
	virtual bool linkAllowed( const std::string& curPath, const std::string& link );

	/// Search if a directory already exists in the watches
	virtual bool pathInWatches( const std::string& path ) = 0;

  protected:
	friend class FileWatcher;
	friend class DirWatcherGeneric;

	FileWatcher* mFileWatcher;
	Atomic<bool> mInitOK;
	bool mIsGeneric;

	int getOptionValue( const std::vector<WatcherOption>& options, Option option,
						int defaultValue );
};

} // namespace efsw

#endif
