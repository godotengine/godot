#ifndef EFSW_WATCHERGENERIC_HPP
#define EFSW_WATCHERGENERIC_HPP

#include <efsw/FileWatcherImpl.hpp>

namespace efsw {

class DirWatcherGeneric;

class WatcherGeneric : public Watcher {
  public:
	FileWatcherImpl* WatcherImpl;
	DirWatcherGeneric* DirWatch;

	WatcherGeneric( WatchID id, const std::string& directory, FileWatchListener* fwl,
					FileWatcherImpl* fw, bool recursive );

	~WatcherGeneric();

	void watch() override;

	void watchDir( std::string dir );

	bool pathInWatches( std::string path );
};

} // namespace efsw

#endif
