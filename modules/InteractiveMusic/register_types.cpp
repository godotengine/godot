#include "register_types.h"

#include "core/class_db.h"
#include "audio_stream_playlist.h"
#include "audio_stream_playback_playlist.h"

void register_InteractiveMusic_types() {

       
		ClassDB::register_class<AudioStreamPlaylist>();
       
		ClassDB::register_class<AudioStreamPlaylistPlayback>();
}

void unregister_InteractiveMusic_types() {
   //nothing to do here
}
