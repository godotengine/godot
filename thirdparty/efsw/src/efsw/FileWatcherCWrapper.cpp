#include <algorithm>
#include <efsw/Lock.hpp>
#include <efsw/Mutex.hpp>
#include <efsw/efsw.h>
#include <efsw/efsw.hpp>
#include <vector>

#define TOBOOL( i ) ( ( i ) == 0 ? false : true )

/*************************************************************************************************/
class Watcher_CAPI : public efsw::FileWatchListener {
  public:
	efsw_watcher mWatcher;
	efsw_pfn_fileaction_callback mFn;
	void* mParam;
	efsw_pfn_handle_missed_fileactions mFnMissedFa;

  public:
	Watcher_CAPI( efsw_watcher watcher, efsw_pfn_fileaction_callback fn, void* param,
				  efsw_pfn_handle_missed_fileactions fnfa ) :
		mWatcher( watcher ), mFn( fn ), mParam( param ), mFnMissedFa( fnfa ) {}

	void handleFileAction( efsw::WatchID watchid, const std::string& dir,
						   const std::string& filename, bool isDir, efsw::Action action,
						   std::string oldFilename = "" ) {
		mFn( mWatcher, watchid, dir.c_str(), filename.c_str(), isDir, (enum efsw_action)action,
			 oldFilename.c_str(), mParam );
	}

	void handleMissedFileActions( efsw::WatchID watchid, const std::string& dir ) {
		if ( mFnMissedFa ) {
			mFnMissedFa( mWatcher, watchid, dir.c_str() );
		}
	}
};

/*************************************************************************************************
 * globals
 */
static std::vector<Watcher_CAPI*> g_callbacks;
static efsw::Mutex g_callbacksMutex;

Watcher_CAPI* find_callback( efsw_watcher watcher, efsw_pfn_fileaction_callback fn, void* param ) {
	efsw::Lock l( g_callbacksMutex );
	for ( Watcher_CAPI* callback : g_callbacks ) {
		if ( callback->mFn == fn && callback->mWatcher == watcher && callback->mParam == param )
			return callback;
	}
	return NULL;
}

void remove_callback( efsw_watcher watcher ) {
	efsw::Lock l( g_callbacksMutex );
	auto found = std::find_if( g_callbacks.begin(), g_callbacks.end(),
							   [watcher]( Watcher_CAPI* cb ) { return cb->mWatcher == watcher; } );
	if ( found != g_callbacks.end() ) {
		Watcher_CAPI* callback = *found;
		delete callback;
		g_callbacks.erase( found );
	}
}

/*************************************************************************************************/
efsw_watcher efsw_create( int generic_mode ) {
	return ( efsw_watcher ) new efsw::FileWatcher( TOBOOL( generic_mode ) );
}

void efsw_release( efsw_watcher watcher ) {
	remove_callback( watcher );
	delete (efsw::FileWatcher*)watcher;
}

const char* efsw_getlasterror() {
	static std::string log_str;
	log_str = efsw::Errors::Log::getLastErrorLog();
	return log_str.c_str();
}

EFSW_API void efsw_clearlasterror() {
	efsw::Errors::Log::clearLastError();
}

efsw_watchid efsw_addwatch( efsw_watcher watcher, const char* directory,
							efsw_pfn_fileaction_callback callback_fn, int recursive, void* param ) {
	return efsw_addwatch_withoptions( watcher, directory, callback_fn, recursive, 0, 0, param,
									  nullptr );
}

efsw_watchid
efsw_addwatch_withoptions( efsw_watcher watcher, const char* directory,
						   efsw_pfn_fileaction_callback callback_fn, int recursive,
						   efsw_watcher_option* options, int options_number, void* param,
						   efsw_pfn_handle_missed_fileactions callback_fn_missed_file_actions ) {
	Watcher_CAPI* callback = find_callback( watcher, callback_fn, param );

	if ( callback == NULL ) {
		callback = new Watcher_CAPI( watcher, callback_fn, param, callback_fn_missed_file_actions );
		efsw::Lock l( g_callbacksMutex );
		g_callbacks.push_back( callback );
	}

	std::vector<efsw::WatcherOption> watcher_options{};
	for ( int i = 0; i < options_number; i++ ) {
		efsw_watcher_option* option = &options[i];
		watcher_options.emplace_back(
			efsw::WatcherOption{ static_cast<efsw::Option>( option->option ), option->value } );
	}

	return ( (efsw::FileWatcher*)watcher )
		->addWatch( std::string( directory ), callback, TOBOOL( recursive ), watcher_options );
}

void efsw_removewatch( efsw_watcher watcher, const char* directory ) {
	( (efsw::FileWatcher*)watcher )->removeWatch( std::string( directory ) );
}

void efsw_removewatch_byid( efsw_watcher watcher, efsw_watchid watchid ) {
	( (efsw::FileWatcher*)watcher )->removeWatch( watchid );
}

void efsw_watch( efsw_watcher watcher ) {
	( (efsw::FileWatcher*)watcher )->watch();
}

void efsw_follow_symlinks( efsw_watcher watcher, int enable ) {
	( (efsw::FileWatcher*)watcher )->followSymlinks( TOBOOL( enable ) );
}

int efsw_follow_symlinks_isenabled( efsw_watcher watcher ) {
	return (int)( (efsw::FileWatcher*)watcher )->followSymlinks();
}

void efsw_allow_outofscopelinks( efsw_watcher watcher, int allow ) {
	( (efsw::FileWatcher*)watcher )->allowOutOfScopeLinks( TOBOOL( allow ) );
}

int efsw_outofscopelinks_isallowed( efsw_watcher watcher ) {
	return (int)( (efsw::FileWatcher*)watcher )->allowOutOfScopeLinks();
}
