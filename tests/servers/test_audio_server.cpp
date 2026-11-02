/**************************************************************************/
/*  test_audio_server.cpp                                                 */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_audio_server)

#include "servers/audio/audio_server.h"

#ifdef TOOLS_ENABLED
#include "editor/audio/audio_stream_preview.h"
#include "scene/resources/audio_stream_wav.h"
#endif

namespace TestAudioServer {

// Reproduce the Web driver's name lookup without requiring a browser or audio device.
class SampleBusDriver : public AudioDriver {
	AudioDriver *previous = AudioDriver::get_singleton();
	Ref<AudioBusLayout> previous_layout = AudioServer::get_singleton()->generate_bus_layout();

public:
	Vector<int> sends;

	SampleBusDriver() {
		previous->lock();
		set_singleton();
	}
	~SampleBusDriver() override {
		AudioServer::get_singleton()->set_bus_layout(previous_layout);
		previous->set_singleton();
		previous->unlock();
	}
	const char *get_name() const override { return "SampleBusTest"; }
	Error init() override { return OK; }
	void start() override {}
	void finish() override {}
	int get_mix_rate() const override { return previous->get_mix_rate(); }
	SpeakerMode get_speaker_mode() const override { return previous->get_speaker_mode(); }
	void lock() override {}
	void unlock() override {}
	void set_sample_bus_count(int p_count) override {
		sends.resize(p_count);
		sends.fill(-2);
	}
	void set_sample_bus_send(int p_bus, const StringName &p_send) override {
		sends.write[p_bus] = AudioServer::get_singleton()->get_bus_index(p_send);
	}
};

TEST_CASE("[Audio] Bus layout is complete before drivers resolve sends") {
	SampleBusDriver driver;
	Ref<AudioBusLayout> layout;
	layout.instantiate();
	layout->set("bus/1/name", "Effects");
	layout->set("bus/1/send", "Later");
	layout->set("bus/2/name", "Music");
	layout->set("bus/2/send", "Missing");
	layout->set("bus/3/name", "Later");
	layout->set("bus/3/send", "Master");
	AudioServer::get_singleton()->set_bus_layout(layout);
	CHECK(driver.sends.size() == 4);
	CHECK(driver.sends[1] == 3);
	CHECK(driver.sends[2] == -1);
	CHECK(driver.sends[3] == 0);

	Ref<AudioBusLayout> replacement;
	replacement.instantiate();
	replacement->set("bus/1/name", "Replacement");
	replacement->set("bus/1/send", "Master");
	AudioServer::get_singleton()->set_bus_layout(replacement);
	CHECK(driver.sends.size() == 2);
	CHECK(driver.sends[1] == 0);
}

} // namespace TestAudioServer

#ifdef TOOLS_ENABLED
TEST_CASE("[Audio] Audio preview destruction waits for decoding") {
	Ref<AudioStreamWAV> stream;
	stream.instantiate();
	stream->set_format(AudioStreamWAV::FORMAT_8_BITS);
	stream->set_mix_rate(44100);
	Vector<uint8_t> silence;
	silence.resize(60 * 44100);
	silence.fill(0);
	stream->set_data(silence);

	AudioStreamPreviewGenerator *generator = memnew(AudioStreamPreviewGenerator);
	Ref<AudioStreamPreview> preview = generator->generate_preview(stream);
	memdelete(generator);

	CHECK(preview->get_version() == 2);
	CHECK(AudioStreamPreviewGenerator::get_singleton() == nullptr);
}
#endif
