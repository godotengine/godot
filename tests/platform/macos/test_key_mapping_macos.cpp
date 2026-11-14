/**************************************************************************/
/*  test_key_mapping_macos.cpp                                            */
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

TEST_FORCE_LINK(test_key_mapping_macos)

#ifdef MACOS_ENABLED

#include "platform/macos/key_mapping_macos.h"

#include <Carbon/Carbon.h>

namespace TestKeyMappingMacOS {

// Equivalent to NSEventModifierFlagCommand; AppKit headers are Objective-C only.
constexpr unsigned int NS_MODIFIER_FLAG_COMMAND = 1 << 20;
// Equivalent to NSEventModifierFlagShift.
constexpr unsigned int NS_MODIFIER_FLAG_SHIFT = 1 << 17;

// Holds an input source alive for as long as its layout data is used.
struct InputSourceLayout {
	TISInputSourceRef source = nullptr;
	const UCKeyboardLayout *layout = nullptr;

	explicit InputSourceLayout(const char *p_input_source_id) {
		CFStringRef source_id = CFStringCreateWithCString(kCFAllocatorDefault, p_input_source_id, kCFStringEncodingUTF8);
		const void *keys[] = { kTISPropertyInputSourceID };
		const void *values[] = { source_id };
		CFDictionaryRef filter = CFDictionaryCreate(kCFAllocatorDefault, keys, values, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
		// `true` includes layouts that are installed but not enabled in System Settings.
		CFArrayRef sources = TISCreateInputSourceList(filter, true);
		if (sources && CFArrayGetCount(sources) > 0) {
			source = (TISInputSourceRef)CFRetain(CFArrayGetValueAtIndex(sources, 0));
			CFDataRef layout_data = (CFDataRef)TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData);
			if (layout_data) {
				layout = (const UCKeyboardLayout *)CFDataGetBytePtr(layout_data);
			}
		}
		if (sources) {
			CFRelease(sources);
		}
		CFRelease(filter);
		CFRelease(source_id);
	}

	~InputSourceLayout() {
		if (source) {
			CFRelease(source);
		}
	}
};

TEST_CASE("[KeyMappingMacOS] Dvorak - QWERTY Command layout") {
	InputSourceLayout dvorak_qwerty("com.apple.keylayout.DVORAK-QWERTYCMD");
	REQUIRE_MESSAGE(dvorak_qwerty.layout != nullptr, "The 'Dvorak - QWERTY ⌘' layout should ship with macOS.");

	SUBCASE("Without Command, keys map to Dvorak") {
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_X, 0, false) == Key::Q);
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_Q, 0, false) == Key::APOSTROPHE);
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_S, 0, false) == Key::O);
	}

	SUBCASE("With Command, keys map to QWERTY so shortcuts match native apps") {
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_X, NS_MODIFIER_FLAG_COMMAND, false) == Key::X);
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_Q, NS_MODIFIER_FLAG_COMMAND, false) == Key::Q);
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_S, NS_MODIFIER_FLAG_COMMAND, false) == Key::S);
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_Z, NS_MODIFIER_FLAG_COMMAND, false) == Key::Z);
	}

	SUBCASE("Key label ignores Command and shows the Dvorak glyph") {
		CHECK(KeyMappingMacOS::remap_key(dvorak_qwerty.layout, kVK_ANSI_X, NS_MODIFIER_FLAG_COMMAND, true) == Key::Q);
	}
}

TEST_CASE("[KeyMappingMacOS] US layout is unaffected by modifiers") {
	InputSourceLayout us("com.apple.keylayout.US");
	REQUIRE_MESSAGE(us.layout != nullptr, "The 'U.S.' layout should ship with macOS.");

	CHECK(KeyMappingMacOS::remap_key(us.layout, kVK_ANSI_X, 0, false) == Key::X);
	CHECK(KeyMappingMacOS::remap_key(us.layout, kVK_ANSI_X, NS_MODIFIER_FLAG_COMMAND, false) == Key::X);
	// Shift must not be forwarded to the layout, otherwise Shift+1 would become Key::EXCLAM.
	CHECK(KeyMappingMacOS::remap_key(us.layout, kVK_ANSI_1, NS_MODIFIER_FLAG_SHIFT, false) == Key::KEY_1);
	CHECK(KeyMappingMacOS::remap_key(us.layout, kVK_ANSI_1, NS_MODIFIER_FLAG_SHIFT | NS_MODIFIER_FLAG_COMMAND, false) == Key::KEY_1);
}

} // namespace TestKeyMappingMacOS

#endif // MACOS_ENABLED
