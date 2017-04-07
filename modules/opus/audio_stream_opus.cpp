/*************************************************************************/
/*  audio_stream_opus.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Author: George Marques <george@gmarqu.es>                             */
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
#include "audio_stream_opus.h"

const float AudioStreamPlaybackOpus::osrate = 48000.0f;

int AudioStreamPlaybackOpus::_op_read_func(void *_stream, unsigned char *_ptr, int _nbytes) {
	FileAccess *fa = (FileAccess *)_stream;

	if (fa->eof_reached())
		return 0;

	uint8_t *dst = (uint8_t *)_ptr;

	int read = fa->get_buffer(dst, _nbytes);

	return read;
}

int AudioStreamPlaybackOpus::_op_seek_func(void *_stream, opus_int64 _offset, int _whence) {

#ifdef SEEK_SET
	FileAccess *fa = (FileAccess *)_stream;

	switch (_whence) {
		case SEEK_SET: {
			fa->seek(_offset);
		} break;
		case SEEK_CUR: {
			fa->seek(fa->get_pos() + _offset);
		} break;
		case SEEK_END: {
			fa->seek_end(_offset);
		} break;
		default: {
			ERR_PRINT("BUG, wtf was whence set to?\n");
		}
	}
	int ret = fa->eof_reached() ? -1 : 0;
	return ret;
#else
	return -1; // no seeking
#endif
}

int AudioStreamPlaybackOpus::_op_close_func(void *_stream) {
	if (!_stream)
		return 0;
	FileAccess *fa = (FileAccess *)_stream;
	if (fa->is_open())
		fa->close();
	return 0;
}

opus_int64 AudioStreamPlaybackOpus::_op_tell_func(void *_stream) {
	FileAccess *_fa = (FileAccess *)_stream;
	return (opus_int64)_fa->get_pos();
}

void AudioStreamPlaybackOpus::_clear_stream() {
	if (!stream_loaded)
		return;

	op_free(opus_file);
	_close_file();

	stream_loaded = false;
	stream_channels = 1;
	playing = false;
}

void AudioStreamPlaybackOpus::_close_file() {
	if (f) {
		memdelete(f);
		f = NULL;
	}
}

Error AudioStreamPlaybackOpus::_load_stream() {

	ERR_FAIL_COND_V(!stream_valid, ERR_UNCONFIGURED);

	_clear_stream();
	if (file == "")
		return ERR_INVALID_DATA;

	Error err;
	f = FileAccess::open(file, FileAccess::READ, &err);

	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	int _err = 0;

	opus_file = op_open_callbacks(f, &_op_callbacks, NULL, 0, &_err);

	switch (_err) {
		case OP_EREAD: { // - Can't read the file.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CANT_READ);
		} break;
		case OP_EVERSION: // - Unrecognized version number.
		case OP_ENOTFORMAT: // - Stream is not Opus data.
		case OP_EIMPL: { // - Stream used non-implemented feature.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
		} break;
		case OP_EBADLINK: // - Failed to find old data after seeking.
		case OP_EBADTIMESTAMP: // - Timestamp failed the validity checks.
		case OP_EBADHEADER: { // - Invalid or mising Opus bitstream header.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} break;
		case OP_EFAULT: { // - Internal logic fault; indicates a bug or heap/stack corruption.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_BUG);
		} break;
	}
	repeats = 0;
	stream_loaded = true;

	return OK;
}

AudioStreamPlaybackOpus::AudioStreamPlaybackOpus() {
	loops = false;
	playing = false;
	f = NULL;
	stream_loaded = false;
	stream_valid = false;
	repeats = 0;
	paused = true;
	stream_channels = 0;
	current_section = 0;
	length = 0;
	loop_restart_time = 0;
	pre_skip = 0;

	_op_callbacks.read = _op_read_func;
	_op_callbacks.seek = _op_seek_func;
	_op_callbacks.tell = _op_tell_func;
	_op_callbacks.close = _op_close_func;
}

Error AudioStreamPlaybackOpus::set_file(const String &p_file) {
	file = p_file;
	stream_valid = false;
	Error err;
	f = FileAccess::open(file, FileAccess::READ, &err);

	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	int _err;

	opus_file = op_open_callbacks(f, &_op_callbacks, NULL, 0, &_err);

	switch (_err) {
		case OP_EREAD: { // - Can't read the file.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CANT_READ);
		} break;
		case OP_EVERSION: // - Unrecognized version number.
		case OP_ENOTFORMAT: // - Stream is not Opus data.
		case OP_EIMPL: { // - Stream used non-implemented feature.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
		} break;
		case OP_EBADLINK: // - Failed to find old data after seeking.
		case OP_EBADTIMESTAMP: // - Timestamp failed the validity checks.
		case OP_EBADHEADER: { // - Invalid or mising Opus bitstream header.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} break;
		case OP_EFAULT: { // - Internal logic fault; indicates a bug or heap/stack corruption.
			memdelete(f);
			f = NULL;
			ERR_FAIL_V(ERR_BUG);
		} break;
	}

	const OpusHead *oinfo = op_head(opus_file, -1);

	stream_channels = oinfo->channel_count;
	pre_skip = oinfo->pre_skip;
	frames_mixed = pre_skip;
	ogg_int64_t len = op_pcm_total(opus_file, -1);
	if (len < 0) {
		length = 0;
	} else {
		length = (len / osrate);
	}

	op_free(opus_file);
	memdelete(f);
	f = NULL;
	stream_valid = true;

	return OK;
}

void AudioStreamPlaybackOpus::play(float p_from) {
	if (playing)
		stop();

	if (_load_stream() != OK)
		return;

	frames_mixed = pre_skip;
	playing = true;
	if (p_from > 0) {
		seek_pos(p_from);
	}
}

void AudioStreamPlaybackOpus::stop() {
	_clear_stream();
	playing = false;
}

void AudioStreamPlaybackOpus::seek_pos(float p_time) {
	if (!playing) return;
	ogg_int64_t pcm_offset = (ogg_int64_t)(p_time * osrate);
	bool ok = op_pcm_seek(opus_file, pcm_offset) == 0;
	if (!ok) {
		ERR_PRINT("Seek time over stream size.");
		return;
	}
	frames_mixed = osrate * p_time;
}

int AudioStreamPlaybackOpus::mix(int16_t *p_bufer, int p_frames) {
	if (!playing)
		return 0;

	int total = p_frames;

	while (true) {

		int todo = p_frames;

		if (todo == 0 || todo < MIN_MIX) {
			break;
		}

		int ret = op_read(opus_file, (opus_int16 *)p_bufer, todo * stream_channels, &current_section);
		if (ret < 0) {
			playing = false;
			ERR_EXPLAIN("Error reading Opus File: " + file);
			ERR_BREAK(ret < 0);
		} else if (ret == 0) { // end of song, reload?
			op_free(opus_file);

			_close_file();

			f = FileAccess::open(file, FileAccess::READ);

			int errv = 0;
			opus_file = op_open_callbacks(f, &_op_callbacks, NULL, 0, &errv);
			if (errv != 0) {
				playing = false;
				break; // :(
			}

			if (!has_loop()) {
				playing = false;
				repeats = 1;
				break;
			}

			if (loop_restart_time) {
				bool ok = op_pcm_seek(opus_file, (loop_restart_time * osrate) + pre_skip) == 0;
				if (!ok) {
					playing = false;
					ERR_PRINT("loop restart time rejected")
				}

				frames_mixed = (loop_restart_time * osrate) + pre_skip;
			} else {
				frames_mixed = pre_skip;
			}
			repeats++;
			continue;
		}

		stream_channels = op_head(opus_file, current_section)->channel_count;

		frames_mixed += ret;

		p_bufer += ret * stream_channels;
		p_frames -= ret;
	}

	return total - p_frames;
}

float AudioStreamPlaybackOpus::get_length() const {
	if (!stream_loaded) {
		if (const_cast<AudioStreamPlaybackOpus *>(this)->_load_stream() != OK)
			return 0;
	}
	return length;
}

float AudioStreamPlaybackOpus::get_pos() const {

	int32_t frames = int32_t(frames_mixed);
	if (frames < 0)
		frames = 0;
	return double(frames) / osrate;
}

int AudioStreamPlaybackOpus::get_minimum_buffer_size() const {
	return MIN_MIX;
}

AudioStreamPlaybackOpus::~AudioStreamPlaybackOpus() {
	_clear_stream();
}

RES ResourceFormatLoaderAudioStreamOpus::load(const String &p_path, const String &p_original_path, Error *r_error) {
	if (r_error)
		*r_error = OK;

	AudioStreamOpus *opus_stream = memnew(AudioStreamOpus);
	opus_stream->set_file(p_path);
	return Ref<AudioStreamOpus>(opus_stream);
}

void ResourceFormatLoaderAudioStreamOpus::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("opus");
}
String ResourceFormatLoaderAudioStreamOpus::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "opus")
		return "AudioStreamOpus";
	return "";
}

bool ResourceFormatLoaderAudioStreamOpus::handles_type(const String &p_type) const {
	return (p_type == "AudioStream" || p_type == "AudioStreamOpus");
}
