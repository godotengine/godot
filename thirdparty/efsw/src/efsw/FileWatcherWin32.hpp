#ifndef EFSW_FILEWATCHERWIN32_HPP
#define EFSW_FILEWATCHERWIN32_HPP

#include <efsw/base.hpp>

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN32

#include <efsw/WatcherWin32.hpp>
#include <map>
#include <unordered_set>
#include <vector>

namespace efsw {

/// Implementation for Win32 based on ReadDirectoryChangesW.
/// @class FileWatcherWin32
class FileWatcherWin32 : public FileWatcherImpl {
  public:
	/// type for a map from WatchID to WatcherWin32 pointer
	typedef std::unordered_set<WatcherStructWin32*> Watches;

	FileWatcherWin32( FileWatcher* parent );

	virtual ~FileWatcherWin32();

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

	static void registerPendingDelete( const PendingDelete& pd );
	static bool tryMatchMove( const FileID& fileID, std::string& outDir, std::string& outFileName,
							  bool isExtended = true );
	void purgeExpiredDeletes();

  protected:
	HANDLE mIOCP;
	Watches mWatches;

	/// The last watchid
	WatchID mLastWatchID;
	Thread* mThread;
	Mutex mWatchesLock;

	bool pathInWatches( const std::string& path ) override;

	/// Remove all directory watches.
	void removeAllWatches();

	void removeWatch( WatcherStructWin32* watch );

  private:
	void run();

	static std::mutex sPendingMutex;
	static std::vector<PendingDelete> sPendingDeletes;
};

} // namespace efsw

#endif

#endif
