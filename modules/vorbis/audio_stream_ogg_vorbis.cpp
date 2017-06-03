/*************************************************************************/
/*  audio_stream_ogg_vorbis.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "audio_stream_ogg_vorbis.h"

size_t AudioStreamPlaybackOGGVorbis::_ov_read_func(void *p_dst, size_t p_data, size_t p_count, void *_f) {

	//printf("read to %p, %i bytes, %i nmemb, %p\n",p_dst,p_data,p_count,_f);
	FileAccess *fa = (FileAccess *)_f;
	size_t read_total = p_data * p_count;

	if (fa->eof_reached())
		return 0;

	uint8_t *dst = (uint8_t *)p_dst;

	int read = fa->get_buffer(dst, read_total);

	return read;
}

int AudioStreamPlaybackOGGVorbis::_ov_seek_func(void *_f, ogg_int64_t offs, int whence) {

//printf("seek to %p, offs %i, whence %i\n",_f,(int)offs,whence);

#ifdef SEEK_SET
	//printf("seek set defined\n");
	FileAccess *fa = (FileAccess *)_f;

	if (whence == SEEK_SET) {

		fa->seek(offs);
	} else if (whence == SEEK_CUR) {

		fa->seek(fa->get_pos() + offs);
	} else if (whence == SEEK_END) {

		fa->seek_end(offs);
	} else {

		ERR_PRINT("BUG, wtf was whence set to?\n");
	}
	int ret = fa->eof_reached() ? -1 : 0;
	//printf("returning %i\n",ret);
	return ret;

#else
	return -1; // no seeking
#endif
}
int AudioStreamPlaybackOGGVorbis::_ov_close_func(void *_f) {

	//printf("close %p\n",_f);
	if (!_f)
		return 0;
	FileAccess *fa = (FileAccess *)_f;
	if (fa->is_open())
		fa->close();
	return 0;
}
long AudioStreamPlaybackOGGVorbis::_ov_tell_func(void *_f) {

	//printf("close %p\n",_f);

	FileAccess *fa = (FileAccess *)_f;
	return fa->get_pos();
}

int AudioStreamPlaybackOGGVorbis::mix(int16_t *p_bufer, int p_frames) {

	if (!playing)
		return 0;

	int total = p_frames;
	while (true) {

		int todo = p_frames;

		if (todo == 0 || todo < MIN_MIX) {
			break;
		}

//printf("to mix %i - mix me %i bytes\n",to_mix,to_mix*stream_channels*sizeof(int16_t));

#ifdef BIG_ENDIAN_ENABLED
		long ret = ov_read(&vf, (char *)p_bufer, todo * stream_channels * sizeof(int16_t), 1, 2, 1, &current_section);
#else
		long ret = ov_read(&vf, (char *)p_bufer, todo * stream_channels * sizeof(int16_t), 0, 2, 1, &current_section);
#endif

		if (ret < 0) {

			playing = false;
			ERR_EXPLAIN("Error reading OGG Vorbis File: " + file);
			ERR_BREAK(ret < 0);
		} else if (ret == 0) { // end of song, reload?

			ov_clear(&vf);

			_close_file();

			if (!has_loop()) {

				playing = false;
				repeats = 1;
				break;
			}

			f = FileAccess::open(file, FileAccess::READ);

			int errv = ov_open_callbacks(f, &vf, NULL, 0, _ov_callbacks);
			if (errv != 0) {
				playing = false;
				break; // :(
			}

			if (loop_restart_time) {
				bool ok = ov_time_seek(&vf, loop_restart_time) == 0;
				if (!ok) {
					playing = false;
					//ERR_EXPLAIN("loop restart time rejected");
					ERR_PRINT("loop restart time rejected")
				}

				frames_mixed = stream_srate * loop_restart_time;
			} else {

				frames_mixed = 0;
			}
			repeats++;
			continue;
		}

		ret /= stream_channels;
		ret /= sizeof(int16_t);

		frames_mixed += ret;

		p_bufer += ret * stream_channels;
		p_frames -= ret;
	}

	return total - p_frames;
}

void AudioStreamPlaybackOGGVorbis::play(float p_from) {

	if (playing)
		stop();

	if (_load_stream() != OK)
		return;

	frames_mixed = 0;
	playing = true;
	if (p_from > 0) {
		seek_pos(p_from);
	}
}

void AudioStreamPlaybackOGGVorbis::_close_file() {

	if (f) {

		memdelete(f);
		f = NULL;
	}
}

bool AudioStreamPlaybackOGGVorbis::is_playing() const {
	return playing;
}
void AudioStreamPlaybackOGGVorbis::stop() {

	_clear_stream();
	playing = false;
	//_clear();
}

float AudioStreamPlaybackOGGVorbis::get_pos() const {

	int32_t frames = int32_t(frames_mixed);
	if (frames < 0)
		frames = 0;
	return double(frames) / stream_srate;
}

void AudioStreamPlaybackOGGVorbis::seek_pos(float p_time) {

	if (!playing)
		return;
	bool ok = ov_time_seek(&vf, p_time) == 0;
	ERR_FAIL_COND(!ok);
	frames_mixed = stream_srate * p_time;
}

String AudioStreamPlaybackOGGVorbis::get_stream_name() const {

	return "";
}

void AudioStreamPlaybackOGGVorbis::set_loop(bool p_enable) {

	loops = p_enable;
}

bool AudioStreamPlaybackOGGVorbis::has_loop() const {

	return loops;
}

int AudioStreamPlaybackOGGVorbis::get_loop_count() const {
	return repeats;
}

Error AudioStreamPlaybackOGGVorbis::set_file(const String &p_file) {

	file = p_file;
	stream_valid = false;
	Error err;
	f = FileAccess::open(file, FileAccess::READ, &err);

	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	int errv = ov_open_callbacks(f, &vf, NULL, 0, _ov_callbacks);
	switch (errv) {

		case OV_EREAD: { // - A read from media returned an error.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CANT_READ);
		} break;
		case OV_EVERSION: // - Vorbis version mismatch.
		case OV_ENOTVORBIS: { // - Bitstream is not Vorbis data.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
		} break;
		case OV_EBADHEADER: { // - Invalid Vorbis bitstream header.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} break;
		case OV_EFAULT: { // - Internal logic fault; indicates a bug or heap/stack corruption.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_BUG);
		} break;
	}
	const vorbis_info *vinfo = ov_info(&vf, -1);
	stream_channels = vinfo->channels;
	stream_srate = vinfo->rate;
	length = ov_time_total(&vf, -1);
	ov_clear(&vf);
	memdelete(f);
	f = NULL;
	stream_valid = true;

	return OK;
}

Error AudioStreamPlaybackOGGVorbis::_load_stream() {

	ERR_FAIL_COND_V(!stream_valid, ERR_UNCONFIGURED);

	_clear_stream();
	if (file == "")
		return ERR_INVALID_DATA;

	Error err;
	f = FileAccess::open(file, FileAccess::READ, &err);
	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	int errv = ov_open_callbacks(f, &vf, NULL, 0, _ov_callbacks);
	switch (errv) {

		case OV_EREAD: { // - A read from media returned an error.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CANT_READ);
		} break;
		case OV_EVERSION: // - Vorbis version mismatch.
		case OV_ENOTVORBIS: { // - Bitstream is not Vorbis data.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
		} break;
		case OV_EBADHEADER: { // - Invalid Vorbis bitstream header.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} break;
		case OV_EFAULT: { // - Internal logic fault; indicates a bug or heap/stack corruption.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_BUG);
		} break;
	}
	repeats = 0;
	stream_loaded = true;

	return OK;
}

float AudioStreamPlaybackOGGVorbis::get_length() const {

	if (!stream_loaded) {
		if (const_cast<AudioStreamPlaybackOGGVorbis *>(this)->_load_stream() != OK)
			return 0;
	}
	return length;
}

void AudioStreamPlaybackOGGVorbis::_clear_stream() {

	if (!stream_loaded)
		return;

	ov_clear(&vf);
	_close_file();

	stream_loaded = false;
	//stream_channels=1;
	playing = false;
}

void AudioStreamPlaybackOGGVorbis::set_paused(bool p_paused) {

	paused = p_paused;
}

bool AudioStreamPlaybackOGGVorbis::is_paused(bool p_paused) const {

	return paused;
}

AudioStreamPlaybackOGGVorbis::AudioStreamPlaybackOGGVorbis() {

	loops = false;
	playing = false;
	_ov_callbacks.read_func = _ov_read_func;
	_ov_callbacks.seek_func = _ov_seek_func;
	_ov_callbacks.close_func = _ov_close_func;
	_ov_callbacks.tell_func = _ov_tell_func;
	f = NULL;
	stream_loaded = false;
	stream_valid = false;
	repeats = 0;
	paused = true;
	stream_channels = 0;
	stream_srate = 0;
	current_section = 0;
	length = 0;
	loop_restart_time = 0;
}

AudioStreamPlaybackOGGVorbis::~AudioStreamPlaybackOGGVorbis() {

	_clear_stream();
}

RES ResourceFormatLoaderAudioStreamOGGVorbis::load(const String &p_path, const String &p_original_path, Error *r_error) {
	if (r_error)
		*r_error = OK;

	AudioStreamOGGVorbis *ogg_stream = memnew(AudioStreamOGGVorbis);
	ogg_stream->set_file(p_path);
	return Ref<AudioStreamOGGVorbis>(ogg_stream);
}

void ResourceFormatLoaderAudioStreamOGGVorbis::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ogg");
}
String ResourceFormatLoaderAudioStreamOGGVorbis::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "ogg")
		return "AudioStreamOGGVorbis";
	return "";
}

bool ResourceFormatLoaderAudioStreamOGGVorbis::handles_type(const String &p_type) const {
	return (p_type == "AudioStream" || p_type == "AudioStreamOGG" || p_type == "AudioStreamOGGVorbis");
}
