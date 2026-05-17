/**************************************************************************/
/*  test_audio_driver_wasapi.cpp                                          */
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

TEST_FORCE_LINK(test_audio_driver_wasapi)

#ifdef WASAPI_ENABLED

#include "drivers/wasapi/audio_driver_wasapi.h"

class TestAudioDriverWASAPIAccessor {
public:
	static DWORD get_stream_flags(bool p_input, int p_configured_mix_rate, int p_device_mix_rate) {
		return AudioDriverWASAPI::get_audio_client_setup(p_input, p_configured_mix_rate, p_device_mix_rate).stream_flags;
	}

	static int get_stream_mix_rate(bool p_input, int p_configured_mix_rate, int p_device_mix_rate) {
		return AudioDriverWASAPI::get_audio_client_setup(p_input, p_configured_mix_rate, p_device_mix_rate).stream_mix_rate;
	}
};

namespace TestAudioDriverWASAPI {

TEST_CASE("[Audio][WASAPI] input streams use the capture endpoint mix rate") {
	CHECK(TestAudioDriverWASAPIAccessor::get_stream_flags(true, 48000, 44100) == 0);
	CHECK(TestAudioDriverWASAPIAccessor::get_stream_mix_rate(true, 48000, 44100) == 44100);
}

TEST_CASE("[Audio][WASAPI] render streams use rate adjustment for non-default mix rates") {
	const DWORD stream_flags = TestAudioDriverWASAPIAccessor::get_stream_flags(false, 48000, 44100);

	CHECK((stream_flags & AUDCLNT_STREAMFLAGS_RATEADJUST) != 0);
	CHECK(TestAudioDriverWASAPIAccessor::get_stream_mix_rate(false, 48000, 44100) == 48000);
}

} // namespace TestAudioDriverWASAPI

#endif // WASAPI_ENABLED
