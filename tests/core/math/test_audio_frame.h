/**************************************************************************/
/*  test_audio_frame.h                                                    */
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

#pragma once

#include "tests/test_macros.h"

#include "core/math/audio_frame.h"

namespace TestAudioFrame {

TEST_CASE("[AudioFrame][lerp] lerp timeline check") {
	AudioFrame original(0.0f, 0.0f);
	AudioFrame target(1.0f, -1.0f);

	Vector<AudioFrame> expected = {
		{ 0.0f, 0.0f }, // t=0
		{ 0.25f, -0.25f }, // t=0.25
		{ 0.5f, -0.5f }, // t=0.5
		{ 0.75f, -0.75f }, // t=0.75
		{ 1.0f, -1.0f } // t=1
	};

	Vector<float> t_values = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

	for (int i = 0; i < t_values.size(); i++) {
		AudioFrame result = original.lerp(target, t_values[i]);
		CHECK(result.left == doctest::Approx(expected[i].left));
		CHECK(result.right == doctest::Approx(expected[i].right));
	}
}
} // namespace TestAudioFrame
