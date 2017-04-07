/*************************************************************************/
/*  audio_driver_bb10.cpp                                                */
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
#include "audio_driver_bb10.h"

#include <errno.h>

Error AudioDriverBB10::init() {
	return init(NULL);
};

Error AudioDriverBB10::init(const char *p_name) {

	active = false;
	thread_exited = false;
	exit_thread = false;
	pcm_open = false;
	samples_in = NULL;
	samples_out = NULL;

	mix_rate = 44100;
	speaker_mode = SPEAKER_MODE_STEREO;

	char *dev_name;
	if (p_name == NULL) {
		dev_name = "pcmPreferred";
	} else {
		dev_name = (char *)p_name;
	}
	printf("******** reconnecting to device %s\n", dev_name);
	int ret = snd_pcm_open_name(&pcm_handle, dev_name, SND_PCM_OPEN_PLAYBACK);
	ERR_FAIL_COND_V(ret < 0, FAILED);
	pcm_open = true;

	snd_pcm_channel_info_t cinfo;
	zeromem(&cinfo, sizeof(cinfo));
	cinfo.channel = SND_PCM_CHANNEL_PLAYBACK;
	snd_pcm_plugin_info(pcm_handle, &cinfo);

	printf("rates %i, %i, %i, %i, %i\n", cinfo.rates, cinfo.rates & SND_PCM_RATE_44100, cinfo.rates & SND_PCM_RATE_32000, cinfo.rates & SND_PCM_RATE_22050, cinfo.max_rate);

	mix_rate = cinfo.max_rate;

	printf("formats %i, %i, %i\n", cinfo.formats, cinfo.formats & SND_PCM_FMT_S16_BE, cinfo.formats & SND_PCM_FMT_S16_LE);
	ERR_FAIL_COND_V(!(cinfo.formats & SND_PCM_FMT_S16_LE), FAILED);

	printf("voices %i\n", cinfo.max_voices);
	speaker_mode = SPEAKER_MODE_STEREO;

	snd_pcm_channel_params_t cp;
	zeromem(&cp, sizeof(cp));
	cp.mode = SND_PCM_MODE_BLOCK;
	cp.channel = SND_PCM_CHANNEL_PLAYBACK;
	cp.start_mode = SND_PCM_START_DATA;
	cp.stop_mode = SND_PCM_STOP_STOP;
	//cp.buf.block.frag_size = cinfo.max_fragment_size;
	cp.buf.block.frag_size = 512;
	cp.buf.block.frags_max = 1;
	cp.buf.block.frags_min = 1;
	cp.format.interleave = 1;
	cp.format.rate = mix_rate;
	cp.format.voices = speaker_mode;
	cp.format.format = SND_PCM_SFMT_S16_LE;

	ret = snd_pcm_plugin_params(pcm_handle, &cp);
	printf("ret is %i, %i\n", ret, cp.why_failed);
	ERR_FAIL_COND_V(ret < 0, FAILED);

	ret = snd_pcm_plugin_prepare(pcm_handle, SND_PCM_CHANNEL_PLAYBACK);
	ERR_FAIL_COND_V(ret < 0, FAILED);

	snd_mixer_group_t group;
	zeromem(&group, sizeof(group));
	snd_pcm_channel_setup_t setup;
	zeromem(&setup, sizeof(setup));
	setup.channel = SND_PCM_CHANNEL_PLAYBACK;
	setup.mode = SND_PCM_MODE_BLOCK;
	setup.mixer_gid = &group.gid;
	ret = snd_pcm_plugin_setup(pcm_handle, &setup);
	ERR_FAIL_COND_V(ret < 0, FAILED);

	pcm_frag_size = setup.buf.block.frag_size;
	pcm_max_frags = 1;

	sample_buf_count = pcm_frag_size * pcm_max_frags / 2;
	printf("sample count %i, %i, %i\n", sample_buf_count, pcm_frag_size, pcm_max_frags);
	samples_in = memnew_arr(int32_t, sample_buf_count);
	samples_out = memnew_arr(int16_t, sample_buf_count);

	thread = Thread::create(AudioDriverBB10::thread_func, this);

	return OK;
};

void AudioDriverBB10::thread_func(void *p_udata) {

	AudioDriverBB10 *ad = (AudioDriverBB10 *)p_udata;

	int channels = speaker_mode;
	int frame_count = ad->sample_buf_count / channels;
	int bytes_out = frame_count * channels * 2;

	while (!ad->exit_thread) {

		if (!ad->active) {

			for (int i = 0; i < ad->sample_buf_count; i++) {

				ad->samples_out[i] = 0;
			};
		} else {

			ad->lock();

			ad->audio_server_process(frame_count, ad->samples_in);

			ad->unlock();

			for (int i = 0; i < frame_count * channels; i++) {

				ad->samples_out[i] = ad->samples_in[i] >> 16;
			}
		};

		int todo = bytes_out;
		int total = 0;

		while (todo) {

			uint8_t *src = (uint8_t *)ad->samples_out;
			int wrote = snd_pcm_plugin_write(ad->pcm_handle, (void *)(src + total), todo);
			if (wrote < 0) {
				// error?
				break;
			};
			total += wrote;
			todo -= wrote;
			if (wrote < todo) {
				if (ad->thread_exited) {
					break;
				};
				printf("pcm_write underrun %i, errno %i\n", (int)ad->thread_exited, errno);
				snd_pcm_channel_status_t status;
				zeromem(&status, sizeof(status));
				// put in non-blocking mode
				snd_pcm_nonblock_mode(ad->pcm_handle, 1);
				status.channel = SND_PCM_CHANNEL_PLAYBACK;
				int ret = snd_pcm_plugin_status(ad->pcm_handle, &status);
				//printf("status return %i, %i, %i, %i, %i\n", ret, errno, status.status, SND_PCM_STATUS_READY, SND_PCM_STATUS_UNDERRUN);
				snd_pcm_nonblock_mode(ad->pcm_handle, 0);
				if (ret < 0) {
					break;
				};
				if (status.status == SND_PCM_STATUS_READY ||
						status.status == SND_PCM_STATUS_UNDERRUN) {
					snd_pcm_plugin_prepare(ad->pcm_handle, SND_PCM_CHANNEL_PLAYBACK);
				} else {
					break;
				};
			};
		};
	};

	snd_pcm_plugin_flush(ad->pcm_handle, SND_PCM_CHANNEL_PLAYBACK);

	ad->thread_exited = true;
	printf("**************** audio thread exit\n");
};

void AudioDriverBB10::start() {

	active = true;
};

int AudioDriverBB10::get_mix_rate() const {

	return mix_rate;
};

AudioDriver::SpeakerMode AudioDriverBB10::get_speaker_mode() const {

	return speaker_mode;
};

void AudioDriverBB10::lock() {

	if (!thread)
		return;
	mutex->lock();
};

void AudioDriverBB10::unlock() {

	if (!thread)
		return;
	mutex->unlock();
};

void AudioDriverBB10::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (pcm_open)
		snd_pcm_close(pcm_handle);

	if (samples_in) {
		memdelete_arr(samples_in);
		memdelete_arr(samples_out);
	};

	memdelete(thread);
	thread = NULL;
};

AudioDriverBB10::AudioDriverBB10() {

	mutex = Mutex::create();
};

AudioDriverBB10::~AudioDriverBB10() {

	memdelete(mutex);
	mutex = NULL;
};
