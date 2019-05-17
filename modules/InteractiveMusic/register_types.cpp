#include "register_types.h"

#include "core/class_db.h"
#include "audio_stream_transitioner.h"
#include "audio_stream_transitioner_playback.h"

void register_InteractiveMusic_types() {

        ClassDB::register_class<AudioStreamTransitioner>();
        ClassDB::register_class<AudioStreamTransitionerPlayback>();
}

void unregister_InteractiveMusic_types() {
   //nothing to do here
}