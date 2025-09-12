/**************************************************************************/
/*  audio_stream_mp3.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_NO_STDIO

#include "audio_stream_mp3.h"

int AudioStreamPlaybackMP3::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	if (!active) {
		return 0;
	}

	int todo = p_frames;

	int frames_mixed_this_step = p_frames;

	int beat_length_frames = -1;
	bool use_loop = looping_override ? looping : mp3_stream->loop;

	bool beat_loop = use_loop && mp3_stream->get_bpm() > 0 && mp3_stream->get_beat_count() > 0;
	if (beat_loop) {
		beat_length_frames = mp3_stream->get_beat_count() * mp3_stream->sample_rate * 60 / mp3_stream->get_bpm();
	}

	while (todo && active) {
		mp3dec_frame_info_t frame_info;
		mp3d_sample_t *buf_frame = nullptr;

		int samples_mixed = mp3dec_ex_read_frame(&mp3d, &buf_frame, &frame_info, mp3_stream->channels);

		if (samples_mixed) {
			p_buffer[p_frames - todo] = AudioFrame(buf_frame[0], buf_frame[samples_mixed - 1]);
			if (loop_fade_remaining < FADE_SIZE) {
				p_buffer[p_frames - todo] += loop_fade[loop_fade_remaining] * (float(FADE_SIZE - loop_fade_remaining) / float(FADE_SIZE));
				loop_fade_remaining++;
			}
			--todo;
			++frames_mixed;

			if (beat_loop && (int)frames_mixed >= beat_length_frames) {
				for (int i = 0; i < FADE_SIZE; i++) {
					samples_mixed = mp3dec_ex_read_frame(&mp3d, &buf_frame, &frame_info, mp3_stream->channels);
					loop_fade[i] = AudioFrame(buf_frame[0], buf_frame[samples_mixed - 1]);
					if (!samples_mixed) {
						break;
					}
				}
				loop_fade_remaining = 0;
				seek(mp3_stream->loop_offset);
				loops++;
			}
		}

		else {
			//EOF
			if (use_loop) {
				seek(mp3_stream->loop_offset);
				loops++;
			} else {
				frames_mixed_this_step = p_frames - todo;
				//fill remainder with silence
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
		}
	}
	return frames_mixed_this_step;
}

float AudioStreamPlaybackMP3::get_stream_sampling_rate() {
	return mp3_stream->sample_rate;
}

void AudioStreamPlaybackMP3::start(double p_from_pos) {
	active = true;
	seek(p_from_pos);
	loops = 0;
	begin_resample();
}

void AudioStreamPlaybackMP3::stop() {
	active = false;
}

bool AudioStreamPlaybackMP3::is_playing() const {
	return active;
}

int AudioStreamPlaybackMP3::get_loop_count() const {
	return loops;
}

double AudioStreamPlaybackMP3::get_playback_position() const {
	return double(frames_mixed) / mp3_stream->sample_rate;
}

void AudioStreamPlaybackMP3::seek(double p_time) {
	if (!active) {
		return;
	}

	if (p_time >= mp3_stream->get_length()) {
		p_time = 0;
	}

	frames_mixed = uint32_t(mp3_stream->sample_rate * p_time);
	mp3dec_ex_seek(&mp3d, (uint64_t)frames_mixed * mp3_stream->channels);
}

void AudioStreamPlaybackMP3::tag_used_streams() {
	mp3_stream->tag_used(get_playback_position());
}

void AudioStreamPlaybackMP3::set_is_sample(bool p_is_sample) {
	_is_sample = p_is_sample;
}

bool AudioStreamPlaybackMP3::get_is_sample() const {
	return _is_sample;
}

Ref<AudioSamplePlayback> AudioStreamPlaybackMP3::get_sample_playback() const {
	return sample_playback;
}

void AudioStreamPlaybackMP3::set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {
	sample_playback = p_playback;
	if (sample_playback.is_valid()) {
		sample_playback->stream_playback = Ref<AudioStreamPlayback>(this);
	}
}

void AudioStreamPlaybackMP3::set_parameter(const StringName &p_name, const Variant &p_value) {
	if (p_name == SNAME("looping")) {
		if (p_value == Variant()) {
			looping_override = false;
			looping = false;
		} else {
			looping_override = true;
			looping = p_value;
		}
	}
}

Variant AudioStreamPlaybackMP3::get_parameter(const StringName &p_name) const {
	if (looping_override && p_name == SNAME("looping")) {
		return looping;
	}
	return Variant();
}

AudioStreamPlaybackMP3::~AudioStreamPlaybackMP3() {
	mp3dec_ex_close(&mp3d);
}

Ref<AudioStreamPlayback> AudioStreamMP3::instantiate_playback() {
	Ref<AudioStreamPlaybackMP3> mp3s;

	ERR_FAIL_COND_V_MSG(data.is_empty(), mp3s,
			"This AudioStreamMP3 does not have an audio file assigned "
			"to it. AudioStreamMP3 should not be created from the "
			"inspector or with `.new()`. Instead, load an audio file.");

	mp3s.instantiate();
	mp3s->mp3_stream = Ref<AudioStreamMP3>(this);

	int errorcode = mp3dec_ex_open_buf(&mp3s->mp3d, data.ptr(), data_len, MP3D_SEEK_TO_SAMPLE);

	mp3s->frames_mixed = 0;
	mp3s->active = false;
	mp3s->loops = 0;

	if (errorcode) {
		ERR_FAIL_COND_V(errorcode, Ref<AudioStreamPlaybackMP3>());
	}

	return mp3s;
}

String AudioStreamMP3::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamMP3::clear_data() {
	data.clear();
}

void AudioStreamMP3::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();

	mp3dec_ex_t *mp3d = memnew(mp3dec_ex_t);
	int err = mp3dec_ex_open_buf(mp3d, p_data.ptr(), src_data_len, MP3D_SEEK_TO_SAMPLE);
	if (err || mp3d->info.hz == 0) {
		memdelete(mp3d);
		ERR_FAIL_MSG("Failed to decode mp3 file. Make sure it is a valid mp3 audio file.");
	}

	channels = mp3d->info.channels;
	sample_rate = mp3d->info.hz;
	length = float(mp3d->samples) / (sample_rate * float(channels));

	mp3dec_ex_close(mp3d);
	memdelete(mp3d);

	data = p_data;
	data_len = src_data_len;
}

Vector<uint8_t> AudioStreamMP3::get_data() const {
	return data;
}

void AudioStreamMP3::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamMP3::has_loop() const {
	return loop;
}

void AudioStreamMP3::set_loop_offset(double p_seconds) {
	loop_offset = p_seconds;
}

double AudioStreamMP3::get_loop_offset() const {
	return loop_offset;
}

double AudioStreamMP3::get_length() const {
	return length;
}

bool AudioStreamMP3::is_monophonic() const {
	return false;
}

void AudioStreamMP3::get_parameter_list(List<Parameter> *r_parameters) {
	r_parameters->push_back(Parameter(PropertyInfo(Variant::BOOL, "looping", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CHECKABLE), Variant()));
}

void AudioStreamMP3::set_bpm(double p_bpm) {
	ERR_FAIL_COND(p_bpm < 0);
	bpm = p_bpm;
	emit_changed();
}

double AudioStreamMP3::get_bpm() const {
	return bpm;
}

void AudioStreamMP3::set_beat_count(int p_beat_count) {
	ERR_FAIL_COND(p_beat_count < 0);
	beat_count = p_beat_count;
	emit_changed();
}

int AudioStreamMP3::get_beat_count() const {
	return beat_count;
}

void AudioStreamMP3::set_bar_beats(int p_bar_beats) {
	ERR_FAIL_COND(p_bar_beats < 0);
	bar_beats = p_bar_beats;
	emit_changed();
}

int AudioStreamMP3::get_bar_beats() const {
	return bar_beats;
}

void AudioStreamMP3::set_tags(const Dictionary &p_tags) {
	tags = p_tags;
}

Dictionary AudioStreamMP3::get_tags() const {
	return tags;
}

Ref<AudioSample> AudioStreamMP3::generate_sample() const {
	Ref<AudioSample> sample;
	sample.instantiate();
	sample->stream = this;
	sample->loop_mode = loop
			? AudioSample::LoopMode::LOOP_FORWARD
			: AudioSample::LoopMode::LOOP_DISABLED;
	sample->loop_begin = loop_offset;
	sample->loop_end = 0;
	return sample;
}

// In ID3v2.3 and higher tag size is stored as 32 bit syncsafe integer.
// See https://id3.org/id3v2.4.0-structure (section 6.2)
#define TAG_SIZE_INT_32_SYNCSAFE(m_buf, m_pos)  \
	(((m_buf[m_pos] & 0x7f) << 21) |            \
			((m_buf[m_pos + 1] & 0x7f) << 14) | \
			((m_buf[m_pos + 2] & 0x7f) << 7) |  \
			(m_buf[m_pos + 3] & 0x7f))

// Some software can write tag sizes as non-syncsafe integers, so we check for that
#define TAG_SIZE_INT_32_AUTO(m_buf, m_pos)           \
	((m_buf[m_pos] & 0b10000000) == 0 &&             \
			(m_buf[m_pos + 1] & 0b10000000) == 0 &&  \
			(m_buf[m_pos + 2] & 0b10000000) == 0 &&  \
			(m_buf[m_pos + 3] & 0b10000000) == 0)    \
			? TAG_SIZE_INT_32_SYNCSAFE(m_buf, m_pos) \
			: ((m_buf[m_pos] << 24) |                \
					  (m_buf[m_pos + 1] << 16) |     \
					  (m_buf[m_pos + 2] << 8) |      \
					  m_buf[m_pos + 3])

// In ID3v2.2 tag size is stored as 28 bit integer.
// See https://id3.org/id3v2.4.0-structure (section 6.2)
#define TAG_SIZE_INT_28(m_buf, m_pos) \
	((m_buf[m_pos] << 16) |           \
			(m_buf[m_pos + 1] << 8) | \
			m_buf[m_pos + 2])

#define TAG_STRING_FROM_ENCODING(m_string, m_ptr, m_size, m_encoding, m_frame_id)                        \
	switch (m_encoding) {                                                                                \
		case TAG_ENCODING_LATIN1:                                                                        \
		case TAG_ENCODING_UTF8:                                                                          \
			m_string.append_utf8((char *)(m_ptr), m_size);                                               \
			break;                                                                                       \
		case TAG_ENCODING_UTF16:                                                                         \
		case TAG_ENCODING_UTF16A:                                                                        \
			m_string.append_utf16((char16_t *)(m_ptr), m_size);                                          \
			break;                                                                                       \
		default:                                                                                         \
			ERR_PRINT(vformat("Invalid encoding value '%d' in tag frame '%s'", m_encoding, m_frame_id)); \
			break;                                                                                       \
	}

Ref<AudioStreamMP3> AudioStreamMP3::load_from_buffer(const Vector<uint8_t> &p_stream_data) {
	Ref<AudioStreamMP3> mp3_stream;
	mp3_stream.instantiate();
	mp3_stream->set_data(p_stream_data);
	ERR_FAIL_COND_V_MSG(mp3_stream->get_data().is_empty(), Ref<AudioStreamMP3>(), "MP3 decoding failed. Check that your data is a valid MP3 audio stream.");

	HashMap<String, String> tag_map;

	if (p_stream_data.size() >= 10 && p_stream_data[0] == 'I' && p_stream_data[1] == 'D' && p_stream_data[2] == '3') {
		const size_t TAG_HEADER_SIZE = 10;
		char tag_header[TAG_HEADER_SIZE];
		memcpy((uint8_t *)&tag_header, &p_stream_data[0], TAG_HEADER_SIZE);

		uint8_t tag_ver = tag_header[3];
		uint8_t tag_flags = tag_header[5];
		size_t tag_data_size = TAG_SIZE_INT_32_SYNCSAFE(tag_header, 6);

		bool unsync = (tag_flags & 0b10000000) != 0;
		bool extended_header = tag_ver >= 3 && (tag_flags & 0b01000000) != 0;
		bool tag_compression = tag_ver == 2 && (tag_flags & 0b01000000) != 0;

		Vector<uint8_t> tag_data;
		tag_data.resize(tag_data_size);
		memcpy((uint8_t *)&tag_data[0], &p_stream_data[TAG_HEADER_SIZE], tag_data_size);

		if (unsync) {
			// To read unsynchronized data, we need to replace 0xff,0x00 with 0xff
			for (size_t i = 0; i < tag_data_size; i++) {
				if ((i + 1) < tag_data_size && tag_data[i] == 0xff && tag_data[i + 1] == 0x00) {
					tag_data.remove_at(i + 1);
					tag_data_size -= 1;
				}
			}
		}

		size_t pos = 0;
		if (extended_header) {
			size_t extended_header_size = TAG_SIZE_INT_32_SYNCSAFE(tag_data, 0);
			pos += extended_header_size;
		}

		const size_t FRAME_HEADER_SIZE_V3 = 10;
		const size_t FRAME_HEADER_SIZE_V2 = 6;
		size_t frame_header_size = tag_ver >= 3 ? FRAME_HEADER_SIZE_V3 : FRAME_HEADER_SIZE_V2;

		const size_t FRAME_ID_SIZE_V3 = 4;
		const size_t FRAME_ID_SIZE_V2 = 3;
		size_t frame_id_size = tag_ver >= 3 ? FRAME_ID_SIZE_V3 : FRAME_ID_SIZE_V2;

		bool invalid_version = tag_ver < 2 || tag_ver > 4;
		bool tag_is_unsupported = tag_compression;

		while (pos < tag_data_size) {
			if (invalid_version || tag_is_unsupported) {
				break;
			}

			if (!tag_data[pos]) {
				// We have hit the padding zeroes
				break;
			}

			uint8_t frame_header[FRAME_HEADER_SIZE_V3] = { 0 };
			memcpy(frame_header, &tag_data[pos], frame_header_size);

			char frame_id[FRAME_ID_SIZE_V3] = { 0 };
			memcpy(frame_id, frame_header, frame_id_size);

			// If frame ID is not alphanumeric, then it is malformed and we should stop
			bool frame_id_is_invalid = false;
			for (size_t i = 0; i < frame_id_size; ++i) {
				if (!((frame_id[i] >= 'A' && frame_id[i] <= 'Z') ||
							(frame_id[i] >= '0' && frame_id[i] <= '9'))) {
					frame_id_is_invalid = true;
				}
			}
			if (frame_id_is_invalid) {
				break;
			}

			size_t frame_size = tag_ver >= 3
					? TAG_SIZE_INT_32_AUTO(frame_header, frame_id_size)
					: TAG_SIZE_INT_28(frame_header, frame_id_size);

			uint8_t frame_format_flags = frame_header[9];
			bool frame_compression = tag_ver >= 3 && (frame_format_flags & 0b00001000) != 0;
			bool frame_encryption = tag_ver >= 3 && (frame_format_flags & 0b00000100) != 0;

			// All text frame IDs begin with "T".
			// See https://id3.org/id3v2.4.0-frames (section 4.2)
			bool frame_is_text = frame_id[0] == 'T';
			bool frame_is_user_defined_text =
					frame_id[0] == 'T' && frame_id[1] == 'X' && frame_id[2] == 'X' &&
					(tag_ver == 2 || frame_id[3] == 'X');
			bool frame_is_comment =
					frame_id[0] == 'C' && frame_id[1] == 'O' && frame_id[2] == 'M' &&
					(tag_ver == 2 || frame_id[3] == 'M');

			// Skip frames with these features, since they are extremely uncommon
			// so implementing them is not worth it.
			bool frame_is_unsupported = frame_compression || frame_encryption;

			if ((frame_is_text || frame_is_comment) && !frame_is_unsupported) {
				const uint8_t TAG_ENCODING_LATIN1 = 0;
				const uint8_t TAG_ENCODING_UTF16 = 1; // with BOM
				const uint8_t TAG_ENCODING_UTF16A = 2; // without BOM
				const uint8_t TAG_ENCODING_UTF8 = 3;
				const uint8_t TAG_ENCODING_SIZE = 1;
				const uint8_t TAG_COMMENT_LANG_SIZE = 3;

				size_t frame_data_pos = pos + frame_header_size;
				// First byte describes text encoding
				uint8_t encoding = tag_data[frame_data_pos];
				// Remaining bytes are the text in that encoding
				size_t text_size = (frame_size >= TAG_ENCODING_SIZE) ? (frame_size - TAG_ENCODING_SIZE) : 0;
				frame_data_pos += TAG_ENCODING_SIZE;

				String tag_name;

				if (frame_is_user_defined_text || frame_is_comment) {
					if (frame_is_comment) {
						// Three bytes describing comment language
						frame_data_pos += TAG_COMMENT_LANG_SIZE;
						text_size = (text_size >= TAG_COMMENT_LANG_SIZE)
								? (text_size - TAG_COMMENT_LANG_SIZE)
								: 0;
					}
					// TXXX and COMM frames have description and text separated by a terminator (0x00)
					// so we use the description as the tag name
					size_t i_step = (encoding == TAG_ENCODING_UTF16 || encoding == TAG_ENCODING_UTF16A)
							? sizeof(char16_t)
							: sizeof(uint8_t);
					for (size_t i = 0; i < text_size; i += i_step) {
						if (!tag_data[frame_data_pos + i]) {
							TAG_STRING_FROM_ENCODING(
									tag_name, &tag_data[frame_data_pos], i, encoding, frame_id)

							size_t text_pos_offset = i + i_step;
							frame_data_pos += text_pos_offset;
							text_size = (text_size >= text_pos_offset) ? (text_size - text_pos_offset) : 0;
						}
					}
					// Description can be empty, so just use the frame ID.
					if (tag_name.is_empty()) {
						tag_name.append_utf8(frame_id, frame_id_size);
					}
				} else {
					tag_name.append_utf8(frame_id, frame_id_size);
				}

				if (text_size > 0) {
					Vector<uint8_t> text;
					text.resize(text_size);
					memcpy((uint8_t *)&text[0], &tag_data[frame_data_pos], text_size);
					// Make sure that the string terminator is present
					text.push_back(0);

					String tag_value;

					TAG_STRING_FROM_ENCODING(
							tag_value, &text[0], text.size(), encoding, frame_id)

					tag_map[tag_name] = tag_value;
				}
			}

			pos += frame_header_size + frame_size;
		}
	}

	if (!tag_map.is_empty()) {
		HashMap<String, String> tag_id_remaps;
		tag_id_remaps.reserve(15);
		// Use Vorbis comment names for parity with other AudioStreams.
		// See https://wiki.hydrogenaudio.org/index.php?title=Tag_Mapping#Mapping_Tables
		// For ID3-exclusive tags, use Musicbrainz Picard internal names
		// https://picard-docs.musicbrainz.org/en/appendices/tag_mapping.html
		// Titles
		tag_id_remaps["TALB"] = "album";
		tag_id_remaps["TAL"] = "album"; // v2.2
		tag_id_remaps["TSOA"] = "albumsort";
		tag_id_remaps["TOAL"] = "originalalbum";
		tag_id_remaps["TOT"] = "originalalbum"; // v2.2
		tag_id_remaps["TSST"] = "discsubtitle";
		tag_id_remaps["TIT1"] = "grouping";
		tag_id_remaps["TT1"] = "grouping"; // v2.2
		tag_id_remaps["TIT2"] = "title";
		tag_id_remaps["TT2"] = "title"; // v2.2
		tag_id_remaps["TSOT"] = "titlesort";
		tag_id_remaps["TIT3"] = "subtitle";
		tag_id_remaps["TT3"] = "subtitle"; // v2.2
		tag_id_remaps["MVNM"] = "movementname";
		tag_id_remaps["MVN"] = "movementname"; // v2.2
		// People & Organizations
		tag_id_remaps["TPE2"] = "albumartist";
		tag_id_remaps["TP2"] = "albumartist"; // v2.2
		tag_id_remaps["TSO2"] = "albumartistsort";
		tag_id_remaps["TPE1"] = "artist";
		tag_id_remaps["TP1"] = "artist"; // v2.2
		tag_id_remaps["TSOP"] = "artistsort";
		tag_id_remaps["TCOM"] = "composer";
		tag_id_remaps["TCM"] = "composer"; // v2.2
		tag_id_remaps["TSOC"] = "composersort";
		tag_id_remaps["TPE3"] = "conductor";
		tag_id_remaps["TP3"] = "conductor"; // v2.2
		tag_id_remaps["TIPL"] = "involvedpeople"; // v2.4
		tag_id_remaps["IPLS"] = "involvedpeople"; // v2.3
		tag_id_remaps["IPL"] = "involvedpeople"; // v2.2
		tag_id_remaps["IPL"] = "involvedpeople";
		tag_id_remaps["TEXT"] = "lyricist";
		tag_id_remaps["TXT"] = "lyricist"; // v2.2
		tag_id_remaps["TMCL"] = "musiciancredits"; // v2.4
		tag_id_remaps["TOPE"] = "originalartist";
		tag_id_remaps["TOA"] = "originalartist"; // v2.2
		tag_id_remaps["TOLY"] = "originallyricist";
		tag_id_remaps["TOL"] = "originallyricist"; // v2.2
		tag_id_remaps["TPUB"] = "organization";
		tag_id_remaps["TPB"] = "organization"; // v2.2
		tag_id_remaps["TRSN"] = "netradiostation";
		tag_id_remaps["TRSO"] = "netradioowner";
		tag_id_remaps["TPE4"] = "remixer";
		tag_id_remaps["TP4"] = "remixer"; // v2.2
		// Counts & Indexes
		tag_id_remaps["TPOS"] = "discnumber";
		tag_id_remaps["TPA"] = "discnumber"; // v2.2
		tag_id_remaps["TRCK"] = "tracknumber";
		tag_id_remaps["TRK"] = "tracknumber"; // v2.2
		tag_id_remaps["TLEN"] = "length";
		tag_id_remaps["TLE"] = "length"; // v2.2
		// Dates
		tag_id_remaps["TDRC"] = "date"; // v2.4
		tag_id_remaps["TYER"] = "date"; // v2.3
		tag_id_remaps["TDAT"] = "date"; // v2.3
		tag_id_remaps["TIME"] = "date"; // v2.3
		tag_id_remaps["TYE"] = "date"; // v2.2
		tag_id_remaps["TDA"] = "date"; // v2.2
		tag_id_remaps["TIM"] = "date"; // v2.2
		tag_id_remaps["TDOR"] = "originaldate"; // 2.4
		tag_id_remaps["TORY"] = "originaldate"; // 2.3
		tag_id_remaps["TOR"] = "originaldate"; // v2.2
		tag_id_remaps["TRDA"] = "recordingdate"; // v2.3
		tag_id_remaps["TRD"] = "recordingdate"; // v2.2
		tag_id_remaps["TDRL"] = "releasedate"; // v2.4
		// Identifiers
		tag_id_remaps["TSRC"] = "isrc";
		tag_id_remaps["TRC"] = "isrc"; // v2.2
		tag_id_remaps["TGID"] = "podcastid";
		// Flags
		tag_id_remaps["TCMP"] = "compilation";
		tag_id_remaps["PCST"] = "podcast";
		// Ripping & Encoding
		tag_id_remaps["TENC"] = "encodedby";
		tag_id_remaps["TEN"] = "encodedby"; // v2.2
		tag_id_remaps["TSSE"] = "encodersettings";
		tag_id_remaps["TSS"] = "encodersettings"; // v2.2
		tag_id_remaps["TDEN"] = "encodingtime"; // v2.4
		tag_id_remaps["TFLT"] = "filetype";
		tag_id_remaps["TFT"] = "filetype"; // v2.2
		tag_id_remaps["TMED"] = "media";
		tag_id_remaps["TMT"] = "media"; // v2.2
		tag_id_remaps["TOFN"] = "originalfilename";
		tag_id_remaps["TOF"] = "originalfilename"; // v2.2
		// Style
		tag_id_remaps["TCON"] = "genre";
		tag_id_remaps["TCO"] = "genre"; // v2.2
		tag_id_remaps["TBPM"] = "bpm";
		tag_id_remaps["TBP"] = "bpm"; // v2.2
		tag_id_remaps["TKEY"] = "key";
		tag_id_remaps["TKE"] = "key"; // v2.2
		tag_id_remaps["TMOO"] = "mood";
		// Miscellaneous
		tag_id_remaps["COMM"] = "comment";
		tag_id_remaps["COM"] = "comment"; // v2.2
		tag_id_remaps["TDES"] = "podcastdesc";
		tag_id_remaps["TCOP"] = "copyright";
		tag_id_remaps["TCR"] = "copyright"; // v2.2
		tag_id_remaps["TLAN"] = "language";
		tag_id_remaps["TLA"] = "language"; // v2.2
		tag_id_remaps["TOWN"] = "fileowner";
		tag_id_remaps["TCAT"] = "podcastcategory";
		tag_id_remaps["TKWD"] = "podcastkeywords";
		tag_id_remaps["TGID"] = "podcastid";
		tag_id_remaps["TPRO"] = "productioncopyright";
		tag_id_remaps["TDLY"] = "playlistdelay";
		tag_id_remaps["TDY"] = "playlistdelay"; // v2.2
		tag_id_remaps["TDTG"] = "taggingtime";
		tag_id_remaps["TSIZ"] = "size";
		tag_id_remaps["TSI"] = "size"; // v2.2

		Dictionary tag_dictionary;
		for (const KeyValue<String, String> &E : tag_map) {
			HashMap<String, String>::ConstIterator remap = tag_id_remaps.find(E.key);
			String tag_key = E.key.to_lower();
			if (remap) {
				tag_key = remap->value;
			}

			tag_dictionary[tag_key] = E.value;
		}
		mp3_stream->set_tags(tag_dictionary);
	}

	return mp3_stream;
}

Ref<AudioStreamMP3> AudioStreamMP3::load_from_file(const String &p_path) {
	const Vector<uint8_t> stream_data = FileAccess::get_file_as_bytes(p_path);
	ERR_FAIL_COND_V_MSG(stream_data.is_empty(), Ref<AudioStreamMP3>(), vformat("Cannot open file '%s'.", p_path));
	return load_from_buffer(stream_data);
}

void AudioStreamMP3::_bind_methods() {
	ClassDB::bind_static_method("AudioStreamMP3", D_METHOD("load_from_buffer", "stream_data"), &AudioStreamMP3::load_from_buffer);
	ClassDB::bind_static_method("AudioStreamMP3", D_METHOD("load_from_file", "path"), &AudioStreamMP3::load_from_file);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamMP3::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamMP3::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamMP3::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamMP3::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamMP3::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamMP3::get_loop_offset);

	ClassDB::bind_method(D_METHOD("set_bpm", "bpm"), &AudioStreamMP3::set_bpm);
	ClassDB::bind_method(D_METHOD("get_bpm"), &AudioStreamMP3::get_bpm);

	ClassDB::bind_method(D_METHOD("set_beat_count", "count"), &AudioStreamMP3::set_beat_count);
	ClassDB::bind_method(D_METHOD("get_beat_count"), &AudioStreamMP3::get_beat_count);

	ClassDB::bind_method(D_METHOD("set_bar_beats", "count"), &AudioStreamMP3::set_bar_beats);
	ClassDB::bind_method(D_METHOD("get_bar_beats"), &AudioStreamMP3::get_bar_beats);

	ClassDB::bind_method(D_METHOD("set_tags", "tags"), &AudioStreamMP3::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &AudioStreamMP3::get_tags);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), "set_bpm", "get_bpm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,512,1,or_greater"), "set_beat_count", "get_beat_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bar_beats", PROPERTY_HINT_RANGE, "2,32,1,or_greater"), "set_bar_beats", "get_bar_beats");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tags", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_tags", "get_tags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamMP3::AudioStreamMP3() {
}

AudioStreamMP3::~AudioStreamMP3() {
	clear_data();
}
