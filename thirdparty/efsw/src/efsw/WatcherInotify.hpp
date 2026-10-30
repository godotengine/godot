#ifndef EFSW_WATCHERINOTIFY_HPP
#define EFSW_WATCHERINOTIFY_HPP

#include <efsw/FileInfo.hpp>
#include <efsw/FileWatcherImpl.hpp>

namespace efsw {

class WatcherInotify : public Watcher {
  public:
	WatcherInotify();

	bool inParentTree( WatcherInotify* parent );

	WatcherInotify* Parent;
	WatchID InotifyID;

	FileInfo DirInfo;
	bool syntheticEvents{ false };
};

} // namespace efsw

#endif
