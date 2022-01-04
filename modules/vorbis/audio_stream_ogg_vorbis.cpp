/*************************************************************************/
/*  audio_stream_ogg_vorbis.cpp                                          */
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

#include "audio_stream_ogg_vorbis.h"

#include "core/io/file_access.h"
#include "core/variant/typed_array.h"
#include "thirdparty/libogg/ogg/ogg.h"

int AudioStreamPlaybackOGGVorbis::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND_V(!ready, 0);
	ERR_FAIL_COND_V(!active, 0);

	int todo = p_frames;

	int start_buffer = 0;

	int frames_mixed_this_step = p_frames;

	while (todo && active) {
		AudioFrame *buffer = p_buffer;
		if (start_buffer > 0) {
			buffer = buffer + start_buffer;
		}
		int mixed = _mix_frames_vorbis(buffer, todo);
		if (mixed < 0) {
			return 0;
		}
		todo -= mixed;
		frames_mixed += mixed;
		start_buffer += mixed;
		if (!have_packets_left && !have_samples_left) {
			//end of file!
			bool is_not_empty = mixed > 0 || vorbis_stream->get_length() > 0;
			if (vorbis_stream->loop && is_not_empty) {
				//loop

				seek(vorbis_stream->loop_offset);
				loops++;
				// we still have buffer to fill, start from this element in the next iteration.
				start_buffer = p_frames - todo;
			} else {
				frames_mixed_this_step = p_frames - todo;
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

int AudioStreamPlaybackOGGVorbis::_mix_frames_vorbis(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND_V(!ready, 0);
	if (!have_samples_left) {
		ogg_packet *packet = nullptr;
		int err;

		if (!vorbis_data_playback->next_ogg_packet(&packet)) {
			have_packets_left = false;
			WARN_PRINT("ran out of packets in stream");
			return -1;
		}

		ERR_FAIL_COND_V_MSG((err = vorbis_synthesis(&block, packet)), 0, "Error during vorbis synthesis " + itos(err));
		ERR_FAIL_COND_V_MSG((err = vorbis_synthesis_blockin(&dsp_state, &block)), 0, "Error during vorbis block processing " + itos(err));

		have_packets_left = !packet->e_o_s;
	}

	float **pcm; // Accessed with pcm[channel_idx][sample_idx].

	int frames = vorbis_synthesis_pcmout(&dsp_state, &pcm);
	if (frames > p_frames) {
		frames = p_frames;
		have_samples_left = true;
	} else {
		have_samples_left = false;
	}

	if (info.channels > 1) {
		for (int frame = 0; frame < frames; frame++) {
			p_buffer[frame].l = pcm[0][frame];
			p_buffer[frame].r = pcm[1][frame];
		}
	} else {
		for (int frame = 0; frame < frames; frame++) {
			p_buffer[frame].l = pcm[0][frame];
			p_buffer[frame].r = pcm[0][frame];
		}
	}
	vorbis_synthesis_read(&dsp_state, frames);
	return frames;
}

float AudioStreamPlaybackOGGVorbis::get_stream_sampling_rate() {
	return vorbis_data->get_sampling_rate();
}

bool AudioStreamPlaybackOGGVorbis::_alloc_vorbis() {
	vorbis_info_init(&info);
	info_is_allocated = true;
	vorbis_comment_init(&comment);
	comment_is_allocated = true;

	ERR_FAIL_COND_V(vorbis_data.is_null(), false);
	vorbis_data_playback = vorbis_data->instance_playback();

	ogg_packet *packet;
	int err;

	for (int i = 0; i < 3; i++) {
		if (!vorbis_data_playback->next_ogg_packet(&packet)) {
			WARN_PRINT("Not enough packets to parse header");
			return false;
		}

		err = vorbis_synthesis_headerin(&info, &comment, packet);
		ERR_FAIL_COND_V_MSG(err != 0, false, "Error parsing header");
	}

	err = vorbis_synthesis_init(&dsp_state, &info);
	ERR_FAIL_COND_V_MSG(err != 0, false, "Error initializing dsp state");
	dsp_state_is_allocated = true;

	err = vorbis_block_init(&dsp_state, &block);
	ERR_FAIL_COND_V_MSG(err != 0, false, "Error initializing block");
	block_is_allocated = true;

	ready = true;

	return true;
}

void AudioStreamPlaybackOGGVorbis::start(float p_from_pos) {
	ERR_FAIL_COND(!ready);
	active = true;
	seek(p_from_pos);
	loops = 0;
	_begin_resample();
}

void AudioStreamPlaybackOGGVorbis::stop() {
	active = false;
}

bool AudioStreamPlaybackOGGVorbis::is_playing() const {
	return active;
}

int AudioStreamPlaybackOGGVorbis::get_loop_count() const {
	return loops;
}

float AudioStreamPlaybackOGGVorbis::get_playback_position() const {
	return float(frames_mixed) / vorbis_data->get_sampling_rate();
}

void AudioStreamPlaybackOGGVorbis::seek(float p_time) {
	ERR_FAIL_COND(!ready);
	ERR_FAIL_COND(vorbis_stream.is_null());
	if (!active) {
		return;
	}

	vorbis_synthesis_restart(&dsp_state);

	if (p_time >= vorbis_stream->get_length()) {
		p_time = 0;
	}
	frames_mixed = uint32_t(vorbis_data->get_sampling_rate() * p_time);

	const int64_t desired_sample = p_time * get_stream_sampling_rate();

	if (!vorbis_data_playback->seek_page(desired_sample)) {
		WARN_PRINT("seek failed");
		return;
	}

	ogg_packet *packet;
	if (!vorbis_data_playback->next_ogg_packet(&packet)) {
		WARN_PRINT_ONCE("seeking beyond limits");
		return;
	}

	// The granule position of the page we're seeking through.
	int64_t granule_pos = 0;

	int headers_remaining = 0;
	int samples_in_page = 0;
	int err;
	while (true) {
		if (vorbis_synthesis_idheader(packet)) {
			headers_remaining = 3;
		}
		if (!headers_remaining) {
			ERR_FAIL_COND_MSG((err = vorbis_synthesis(&block, packet)), "Error during vorbis synthesis " + itos(err));
			ERR_FAIL_COND_MSG((err = vorbis_synthesis_blockin(&dsp_state, &block)), "Error during vorbis block processing " + itos(err));

			int samples_out = vorbis_synthesis_pcmout(&dsp_state, nullptr);
			ERR_FAIL_COND_MSG((err = vorbis_synthesis_read(&dsp_state, samples_out)), "Error during vorbis read updating " + itos(err));

			samples_in_page += samples_out;

		} else {
			headers_remaining--;
		}
		if (packet->granulepos != -1 && headers_remaining == 0) {
			// This indicates the end of the page.
			granule_pos = packet->granulepos;
			break;
		}
		if (packet->e_o_s) {
			break;
		}
		if (!vorbis_data_playback->next_ogg_packet(&packet)) {
			// We should get an e_o_s flag before this happens.
			WARN_PRINT("Vorbis file ended without warning.");
			break;
		}
	}

	int64_t samples_to_burn = samples_in_page - (granule_pos - desired_sample);

	if (samples_to_burn > samples_in_page) {
		WARN_PRINT("Burning more samples than we have in this page. Check seek algorithm.");
	} else if (samples_to_burn < 0) {
		WARN_PRINT("Burning negative samples doesn't make sense. Check seek algorithm.");
	}

	// Seek again, this time we'll burn a specific number of samples instead of all of them.
	if (!vorbis_data_playback->seek_page(desired_sample)) {
		WARN_PRINT("seek failed");
		return;
	}

	if (!vorbis_data_playback->next_ogg_packet(&packet)) {
		WARN_PRINT_ONCE("seeking beyond limits");
		return;
	}
	vorbis_synthesis_restart(&dsp_state);

	while (true) {
		if (vorbis_synthesis_idheader(packet)) {
			headers_remaining = 3;
		}
		if (!headers_remaining) {
			ERR_FAIL_COND_MSG((err = vorbis_synthesis(&block, packet)), "Error during vorbis synthesis " + itos(err));
			ERR_FAIL_COND_MSG((err = vorbis_synthesis_blockin(&dsp_state, &block)), "Error during vorbis block processing " + itos(err));

			int samples_out = vorbis_synthesis_pcmout(&dsp_state, nullptr);
			int read_samples = samples_to_burn > samples_out ? samples_out : samples_to_burn;
			ERR_FAIL_COND_MSG((err = vorbis_synthesis_read(&dsp_state, samples_out)), "Error during vorbis read updating " + itos(err));
			samples_to_burn -= read_samples;

			if (samples_to_burn <= 0) {
				break;
			}
		} else {
			headers_remaining--;
		}
		if (packet->granulepos != -1 && headers_remaining == 0) {
			// This indicates the end of the page.
			break;
		}
		if (packet->e_o_s) {
			break;
		}
		if (!vorbis_data_playback->next_ogg_packet(&packet)) {
			// We should get an e_o_s flag before this happens.
			WARN_PRINT("Vorbis file ended without warning.");
			break;
		}
	}
}

AudioStreamPlaybackOGGVorbis::~AudioStreamPlaybackOGGVorbis() {
	if (block_is_allocated) {
		vorbis_block_clear(&block);
	}
	if (dsp_state_is_allocated) {
		vorbis_dsp_clear(&dsp_state);
	}
	if (comment_is_allocated) {
		vorbis_comment_clear(&comment);
	}
	if (info_is_allocated) {
		vorbis_info_clear(&info);
	}
}

Ref<AudioStreamPlayback> AudioStreamOGGVorbis::instance_playback() {
	Ref<AudioStreamPlaybackOGGVorbis> ovs;

	ERR_FAIL_COND_V(packet_sequence.is_null(), nullptr);

	ovs.instantiate();
	ovs->vorbis_stream = Ref<AudioStreamOGGVorbis>(this);
	ovs->vorbis_data = packet_sequence;
	ovs->frames_mixed = 0;
	ovs->active = false;
	ovs->loops = 0;
	if (ovs->_alloc_vorbis()) {
		return ovs;
	}
	// Failed to allocate data structures.
	return nullptr;
}

String AudioStreamOGGVorbis::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamOGGVorbis::maybe_update_info() {
	ERR_FAIL_COND(packet_sequence.is_null());

	vorbis_info info;
	vorbis_comment comment;
	int err;

	vorbis_info_init(&info);
	vorbis_comment_init(&comment);

	int packet_count = 0;
	Ref<OGGPacketSequencePlayback> packet_sequence_playback = packet_sequence->instance_playback();

	for (int i = 0; i < 3; i++) {
		ogg_packet *packet;
		if (!packet_sequence_playback->next_ogg_packet(&packet)) {
			WARN_PRINT("Failed to get header packet");
			break;
		}
		if (i == 0) {
			packet->b_o_s = 1;
		}

		if (i == 0) {
			ERR_FAIL_COND(!vorbis_synthesis_idheader(packet));
		}

		err = vorbis_synthesis_headerin(&info, &comment, packet);
		ERR_FAIL_COND_MSG(err != 0, "Error parsing header packet " + itos(i) + ": " + itos(err));

		packet_count++;
	}

	packet_sequence->set_sampling_rate(info.rate);

	vorbis_comment_clear(&comment);
	vorbis_info_clear(&info);
}

void AudioStreamOGGVorbis::set_packet_sequence(Ref<OGGPacketSequence> p_packet_sequence) {
	packet_sequence = p_packet_sequence;
	if (packet_sequence.is_valid()) {
		maybe_update_info();
	}
}

Ref<OGGPacketSequence> AudioStreamOGGVorbis::get_packet_sequence() const {
	return packet_sequence;
}

void AudioStreamOGGVorbis::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamOGGVorbis::has_loop() const {
	return loop;
}

void AudioStreamOGGVorbis::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamOGGVorbis::get_loop_offset() const {
	return loop_offset;
}

float AudioStreamOGGVorbis::get_length() const {
	ERR_FAIL_COND_V(packet_sequence.is_null(), 0);
	return packet_sequence->get_length();
}

bool AudioStreamOGGVorbis::is_monophonic() const {
	return false;
}

void AudioStreamOGGVorbis::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_packet_sequence", "packet_sequence"), &AudioStreamOGGVorbis::set_packet_sequence);
	ClassDB::bind_method(D_METHOD("get_packet_sequence"), &AudioStreamOGGVorbis::get_packet_sequence);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOGGVorbis::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOGGVorbis::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamOGGVorbis::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamOGGVorbis::get_loop_offset);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "packet_sequence", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_packet_sequence", "get_packet_sequence");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamOGGVorbis::AudioStreamOGGVorbis() {}

AudioStreamOGGVorbis::~AudioStreamOGGVorbis() {}
