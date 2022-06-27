/*************************************************************************/
/*  movie_writer.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "movie_writer.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"

MovieWriter *MovieWriter::writers[MovieWriter::MAX_WRITERS];
uint32_t MovieWriter::writer_count = 0;

void MovieWriter::add_writer(MovieWriter *p_writer) {
	ERR_FAIL_COND(writer_count == MAX_WRITERS);
	writers[writer_count++] = p_writer;
}

MovieWriter *MovieWriter::find_writer_for_file(const String &p_file) {
	for (int32_t i = writer_count - 1; i >= 0; i--) { // More recent last, to have override ability.
		if (writers[i]->handles_file(p_file)) {
			return writers[i];
		}
	}
	return nullptr;
}

uint32_t MovieWriter::get_audio_mix_rate() const {
	uint32_t ret = 0;
	if (GDVIRTUAL_REQUIRED_CALL(_get_audio_mix_rate, ret)) {
		return ret;
	}
	return 48000;
}
AudioServer::SpeakerMode MovieWriter::get_audio_speaker_mode() const {
	AudioServer::SpeakerMode ret = AudioServer::SPEAKER_MODE_STEREO;
	if (GDVIRTUAL_REQUIRED_CALL(_get_audio_speaker_mode, ret)) {
		return ret;
	}
	return AudioServer::SPEAKER_MODE_STEREO;
}

Error MovieWriter::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	Error ret = OK;
	if (GDVIRTUAL_REQUIRED_CALL(_write_begin, p_movie_size, p_fps, p_base_path, ret)) {
		return ret;
	}
	return ERR_UNCONFIGURED;
}

Error MovieWriter::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	Error ret = OK;
	if (GDVIRTUAL_REQUIRED_CALL(_write_frame, p_image, p_audio_data, ret)) {
		return ret;
	}
	return ERR_UNCONFIGURED;
}

void MovieWriter::write_end() {
	GDVIRTUAL_REQUIRED_CALL(_write_end);
}

bool MovieWriter::handles_file(const String &p_path) const {
	bool ret = false;
	if (GDVIRTUAL_REQUIRED_CALL(_handles_file, p_path, ret)) {
		return ret;
	}
	return false;
}

void MovieWriter::get_supported_extensions(List<String> *r_extensions) const {
	Vector<String> exts;
	if (GDVIRTUAL_REQUIRED_CALL(_get_supported_extensions, exts)) {
		for (int i = 0; i < exts.size(); i++) {
			r_extensions->push_back(exts[i]);
		}
	}
}

void MovieWriter::begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	mix_rate = get_audio_mix_rate();
	AudioDriverDummy::get_dummy_singleton()->set_mix_rate(mix_rate);
	AudioDriverDummy::get_dummy_singleton()->set_speaker_mode(AudioDriver::SpeakerMode(get_audio_speaker_mode()));
	fps = p_fps;
	if ((mix_rate % fps) != 0) {
		WARN_PRINT("Audio mix rate (" + itos(mix_rate) + ") can not be divided by fps (" + itos(fps) + "). Audio may go out of sync over time.");
	}

	audio_channels = AudioDriverDummy::get_dummy_singleton()->get_channels();
	audio_mix_buffer.resize(mix_rate * audio_channels / fps);

	write_begin(p_movie_size, p_fps, p_base_path);
}

void MovieWriter::_bind_methods() {
	ClassDB::bind_static_method("MovieWriter", D_METHOD("add_writer", "writer"), &MovieWriter::add_writer);

	GDVIRTUAL_BIND(_get_audio_mix_rate)
	GDVIRTUAL_BIND(_get_audio_speaker_mode)

	GDVIRTUAL_BIND(_handles_file, "path")

	GDVIRTUAL_BIND(_write_begin, "movie_size", "fps", "base_path")
	GDVIRTUAL_BIND(_write_frame, "frame_image", "audio_frame_block")
	GDVIRTUAL_BIND(_write_end)

	GLOBAL_DEF("editor/movie_writer/mix_rate_hz", 48000);
	GLOBAL_DEF("editor/movie_writer/speaker_mode", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("editor/movie_writer/speaker_mode", PropertyInfo(Variant::INT, "editor/movie_writer/speaker_mode", PROPERTY_HINT_ENUM, "Stereo,3.1,5.1,7.1"));
	GLOBAL_DEF("editor/movie_writer/mjpeg_quality", 0.75);
	// used by the editor
	GLOBAL_DEF_BASIC("editor/movie_writer/movie_file", "");
	GLOBAL_DEF_BASIC("editor/movie_writer/disable_vsync", false);
	GLOBAL_DEF_BASIC("editor/movie_writer/fps", 60);
	ProjectSettings::get_singleton()->set_custom_property_info("editor/movie_writer/fps", PropertyInfo(Variant::INT, "editor/movie_writer/fps", PROPERTY_HINT_RANGE, "1,300,1"));
}

void MovieWriter::set_extensions_hint() {
	RBSet<String> found;
	for (uint32_t i = 0; i < writer_count; i++) {
		List<String> extensions;
		writers[i]->get_supported_extensions(&extensions);
		for (const String &ext : extensions) {
			found.insert(ext);
		}
	}

	String ext_hint;

	for (const String &S : found) {
		if (ext_hint != "") {
			ext_hint += ",";
		}
		ext_hint += "*." + S;
	}
	ProjectSettings::get_singleton()->set_custom_property_info("editor/movie_writer/movie_file", PropertyInfo(Variant::STRING, "editor/movie_writer/movie_file", PROPERTY_HINT_GLOBAL_SAVE_FILE, ext_hint));
}

void MovieWriter::add_frame(const Ref<Image> &p_image) {
	AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
	write_frame(p_image, audio_mix_buffer.ptr());
}

void MovieWriter::end() {
	write_end();
}
/////////////////////////////////////////

uint32_t MovieWriterPNGWAV::get_audio_mix_rate() const {
	return mix_rate;
}
AudioServer::SpeakerMode MovieWriterPNGWAV::get_audio_speaker_mode() const {
	return speaker_mode;
}

void MovieWriterPNGWAV::get_supported_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("png");
}

bool MovieWriterPNGWAV::handles_file(const String &p_path) const {
	return p_path.get_extension().to_lower() == "png";
}

String MovieWriterPNGWAV::zeros_str(uint32_t p_index) {
	char zeros[MAX_TRAILING_ZEROS + 1];
	for (uint32_t i = 0; i < MAX_TRAILING_ZEROS; i++) {
		uint32_t idx = MAX_TRAILING_ZEROS - i - 1;
		uint32_t digit = (p_index / uint32_t(Math::pow(double(10), double(idx)))) % 10;
		zeros[i] = '0' + digit;
	}
	zeros[MAX_TRAILING_ZEROS] = 0;
	return zeros;
}

Error MovieWriterPNGWAV::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	// Quick & Dirty PNGWAV Code based on - https://docs.microsoft.com/en-us/windows/win32/directshow/avi-riff-file-reference

	base_path = p_base_path.get_basename();
	if (base_path.is_relative_path()) {
		base_path = "res://" + base_path;
	}

	{
		//Remove existing files before writing anew
		uint32_t idx = 0;
		Ref<DirAccess> d = DirAccess::open(base_path.get_base_dir());
		String file = base_path.get_file();
		while (true) {
			String path = file + zeros_str(idx) + ".png";
			if (d->remove(path) != OK) {
				break;
			}
		}
	}

	f_wav = FileAccess::open(base_path + ".wav", FileAccess::WRITE_READ);
	ERR_FAIL_COND_V(f_wav.is_null(), ERR_CANT_OPEN);

	fps = p_fps;

	f_wav->store_buffer((const uint8_t *)"RIFF", 4);
	int total_size = 4 /* WAVE */ + 8 /* fmt+size */ + 16 /* format */ + 8 /* data+size */;
	f_wav->store_32(total_size); //will store final later
	f_wav->store_buffer((const uint8_t *)"WAVE", 4);

	/* FORMAT CHUNK */

	f_wav->store_buffer((const uint8_t *)"fmt ", 4);

	uint32_t channels = 2;
	switch (speaker_mode) {
		case AudioServer::SPEAKER_MODE_STEREO:
			channels = 2;
			break;
		case AudioServer::SPEAKER_SURROUND_31:
			channels = 4;
			break;
		case AudioServer::SPEAKER_SURROUND_51:
			channels = 6;
			break;
		case AudioServer::SPEAKER_SURROUND_71:
			channels = 8;
			break;
	}

	f_wav->store_32(16); //standard format, no extra fields
	f_wav->store_16(1); // compression code, standard PCM
	f_wav->store_16(channels); //CHANNELS: 2

	f_wav->store_32(mix_rate);

	/* useless stuff the format asks for */

	int bits_per_sample = 32;
	int blockalign = bits_per_sample / 8 * channels;
	int bytes_per_sec = mix_rate * blockalign;

	audio_block_size = (mix_rate / fps) * blockalign;

	f_wav->store_32(bytes_per_sec);
	f_wav->store_16(blockalign); // block align (unused)
	f_wav->store_16(bits_per_sample);

	/* DATA CHUNK */

	f_wav->store_buffer((const uint8_t *)"data", 4);

	f_wav->store_32(0); //data size... wooh
	wav_data_size_pos = f_wav->get_position();

	return OK;
}

Error MovieWriterPNGWAV::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	ERR_FAIL_COND_V(!f_wav.is_valid(), ERR_UNCONFIGURED);

	Vector<uint8_t> png_buffer = p_image->save_png_to_buffer();

	Ref<FileAccess> fi = FileAccess::open(base_path + zeros_str(frame_count) + ".png", FileAccess::WRITE);
	fi->store_buffer(png_buffer.ptr(), png_buffer.size());
	f_wav->store_buffer((const uint8_t *)p_audio_data, audio_block_size);

	frame_count++;

	return OK;
}

void MovieWriterPNGWAV::write_end() {
	if (f_wav.is_valid()) {
		uint32_t total_size = 4 /* WAVE */ + 8 /* fmt+size */ + 16 /* format */ + 8 /* data+size */;
		uint32_t datasize = f_wav->get_position() - wav_data_size_pos;
		f_wav->seek(4);
		f_wav->store_32(total_size + datasize);
		f_wav->seek(0x28);
		f_wav->store_32(datasize);
	}
}

MovieWriterPNGWAV::MovieWriterPNGWAV() {
	mix_rate = GLOBAL_GET("editor/movie_writer/mix_rate_hz");
	speaker_mode = AudioServer::SpeakerMode(int(GLOBAL_GET("editor/movie_writer/speaker_mode")));
}
