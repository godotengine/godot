#ifndef EFSW_WATCHERIMPL_HPP
#define EFSW_WATCHERIMPL_HPP

#include <efsw/base.hpp>
#include <efsw/efsw.hpp>

namespace efsw {

/** @brief Base Watcher class */
class Watcher {
  public:
	Watcher();

	Watcher( WatchID id, std::string directory, FileWatchListener* listener, bool recursive );

	virtual ~Watcher() {}

	virtual void watch() {}

	WatchID ID;
	std::string Directory;
	FileWatchListener* Listener;
	bool Recursive;
	std::string OldFileName;
};

} // namespace efsw

#endif
