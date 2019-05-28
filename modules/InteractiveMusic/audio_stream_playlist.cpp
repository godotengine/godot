#include "audio_stream_playlist.h"

AudioStreamPlaylist::AudioStreamPlaylist() {

}

Ref<AudioStreamPlayback> AudioStreamPlaylist::instance_playback() {
	Ref<AudioStreamPlaybackPlaylist> playlist;
	playlist.instance();
	playlist->base = Ref<AudioStreamPlaylist>(this);
	return playlist;
}

String AudioStreamPlaylist::get_stream_name() const {
	return "Playlist";
}
void AudioStreamPlaylist::reset() {
	set_position(0);
}
void AudioStreamPlaylist::set_position(uint64_t p) {
	pos = p;
}

void AudioStreamPlaylist::set_stream_count(int count) {
	stream_count = count;
}

void AudioStreamPlaylist::set_bpm(int beats) {
	bpm = beats;
}

void AudioStreamPlaylist::play(Vector<Ref<AudioStream> > audio_streams, int stream_count) {
	for (int i = 0; i < stream_count; i++) {
		//do something with audio_streams, looping through them and writing to a buffer?
		//


	}
}

void AudioStreamPlaylist::_get_property_list(List<PropertyInfo> *p_list) {
	p_list->push_back(PropertyInfo(Variant::INT, "bpm", PROPERTY_HINT_RANGE, "50,250,1"));
	p_list->push_back(PropertyInfo(Variant::INT, "order", PROPERTY_HINT_ENUM, "Sequence,Shuffle"));
	p_list->push_back(PropertyInfo(Variant::INT, "stream_count", PROPERTY_HINT_RANGE, "0,100,1"));

	for (int i = 0; i < stream_count; i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "audio_streams" + (i + 1), PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"));
	}
}

