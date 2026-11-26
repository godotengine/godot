#include <efsw/DirWatcherGeneric.hpp>
#include <efsw/FileSystem.hpp>
#include <efsw/WatcherGeneric.hpp>

namespace efsw {

WatcherGeneric::WatcherGeneric( WatchID id, const std::string& directory, FileWatchListener* fwl,
								FileWatcherImpl* fw, bool recursive ) :
	Watcher( id, directory, fwl, recursive ), WatcherImpl( fw ), DirWatch( NULL ) {
	FileSystem::dirAddSlashAtEnd( Directory );

	DirWatch = new DirWatcherGeneric( NULL, this, directory, recursive, false );

	DirWatch->addChilds( false );
}

WatcherGeneric::~WatcherGeneric() {
	efSAFE_DELETE( DirWatch );
}

void WatcherGeneric::watch() {
	DirWatch->watch();
}

void WatcherGeneric::watchDir( std::string dir ) {
	DirWatch->watchDir( dir );
}

bool WatcherGeneric::pathInWatches( std::string path ) {
	return DirWatch->pathInWatches( path );
}

} // namespace efsw
