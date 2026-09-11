#include <efsw/WatcherInotify.hpp>

namespace efsw {

WatcherInotify::WatcherInotify() : Watcher(), Parent( NULL ) {}

bool WatcherInotify::inParentTree( WatcherInotify* parent ) {
	WatcherInotify* tNext = Parent;

	while ( NULL != tNext ) {
		if ( tNext == parent ) {
			return true;
		}

		tNext = tNext->Parent;
	}

	return false;
}

} // namespace efsw
