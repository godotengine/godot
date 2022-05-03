/*************************************************************************/
/*  editor_fonts.h                                                       */
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

#ifndef EDITOR_FONTS_H
#define EDITOR_FONTS_H

#include "scene/resources/theme.h"

#include <atomic>

class EditorFonts {
	static EditorFonts *singleton;

	const float embolden_strength = 0.6;

	HashMap<String, Ref<FontData>> cache;

	struct PreRenderRequest {
		char32_t start = 0;
		char32_t end = 0;
		int size = 0;
		Ref<FontData> data;

		PreRenderRequest() {}
		PreRenderRequest(Ref<FontData> &p_data, char32_t p_start = 0, char32_t p_end = 0, int p_size = 16) {
			start = p_start;
			end = p_end;
			size = p_size;
			data = p_data;
		}
	};

	Vector<PreRenderRequest> pre_render_rq;
	std::atomic<bool> pre_render_exit;
	Thread pre_render_thread;

	void stop_pre_render();
	void start_pre_render();
	static void _pre_render_func(void *);

	struct InternalFontData {
		const uint8_t *ptr;
		size_t size;
		String hash;

		InternalFontData() {}
		InternalFontData(const uint8_t *p_ptr, size_t p_size);
	};
	HashMap<String, InternalFontData> internal_fonts;

	struct ExternalFontData {
		PackedByteArray data;
		String hash;
		ExternalFontData() {}
		ExternalFontData(const String &p_path);
	};
	HashMap<String, ExternalFontData> external_fonts;
	Vector<String> fallback_list;

	Ref<FontData> load_cached_font(const HashMap<String, Ref<FontData>> &p_old_cache, const String &p_id, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf, bool p_embolden, bool p_slanted);
	Ref<Font> make_font(const Ref<FontData> &p_default, const Ref<FontData> &p_custom, const Vector<Ref<FontData>> &p_fallback, const String &p_variations = String());

public:
	static EditorFonts *get_singleton();

	bool has_external_editor_font_data(const String &p_id) const;
	PackedByteArray get_external_editor_font_data(const String &p_id) const;

	bool has_internal_editor_font_data(const String &p_id) const;
	const uint8_t *get_internal_editor_font_data_ptr(const String &p_id) const;
	size_t get_internal_editor_font_data_size(const String &p_id) const;

	void load_fonts(Ref<Theme> &p_theme);

	EditorFonts();
	~EditorFonts();
};

#endif
