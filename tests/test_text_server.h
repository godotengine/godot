/*************************************************************************/
/*  test_text_server.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef TOOLS_ENABLED

#ifndef TEST_TEXT_SERVER_H
#define TEST_TEXT_SERVER_H

#include "editor/builtin_fonts.gen.h"
#include "servers/text_server.h"
#include "tests/test_macros.h"

namespace TestTextServer {

TEST_SUITE("[[TextServer]") {
	TEST_CASE("[TextServer] Init, font loading and shaping") {
		TextServerManager *tsman = memnew(TextServerManager);
		Error err = OK;

		SUBCASE("[TextServer] Init") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);
				TEST_FAIL_COND((err != OK || ts == nullptr), "Text server ", TextServerManager::get_interface_name(i), " init failed.");
			}
		}

		SUBCASE("[TextServer] Loading fonts") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);

				RID font = ts->create_font_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf");
				TEST_FAIL_COND(font == RID(), "Loading font failed.");
				ts->free(font);
			}
		}

		SUBCASE("[TextServer] Text layout: Font fallback") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);

				Vector<RID> font;
				font.push_back(ts->create_font_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf"));
				font.push_back(ts->create_font_memory(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size, "ttf"));

				String test = U"คนอ้วน khon uan ראה";
				//                 6^       17^

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				Vector<TextServer::Glyph> glyphs = ts->shaped_text_get_glyphs(ctx);
				TEST_FAIL_COND(glyphs.size() == 0, "Shaping failed");
				for (int j = 0; j < glyphs.size(); j++) {
					if (glyphs[j].start < 6) {
						TEST_FAIL_COND(glyphs[j].font_rid != font[1], "Incorrect font selected.");
					}
					if ((glyphs[j].start > 6) && (glyphs[j].start < 16)) {
						TEST_FAIL_COND(glyphs[j].font_rid != font[0], "Incorrect font selected.");
					}
					if (glyphs[j].start > 16) {
						TEST_FAIL_COND(glyphs[j].font_rid != RID(), "Incorrect font selected.");
						TEST_FAIL_COND(glyphs[j].index != test[glyphs[j].start], "Incorrect glyph index.");
					}
					TEST_FAIL_COND((glyphs[j].start < 0 || glyphs[j].end > test.length()), "Incorrect glyph range.");
					TEST_FAIL_COND(glyphs[j].font_size != 16, "Incorrect glyph font size.");
				}

				ts->free(ctx);

				for (int j = 0; j < font.size(); j++) {
					ts->free(font[j]);
				}
				font.clear();
			}
		}

		SUBCASE("[TextServer] Text layout: BiDi") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);

				if (!ts->has_feature(TextServer::FEATURE_BIDI_LAYOUT)) {
					continue;
				}

				Vector<RID> font;
				font.push_back(ts->create_font_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf"));
				font.push_back(ts->create_font_memory(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size, "ttf"));

				String test = U"Arabic (اَلْعَرَبِيَّةُ, al-ʿarabiyyah)";
				//                    7^      26^

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				Vector<TextServer::Glyph> glyphs = ts->shaped_text_get_glyphs(ctx);
				TEST_FAIL_COND(glyphs.size() == 0, "Shaping failed");
				for (int j = 0; j < glyphs.size(); j++) {
					if (glyphs[j].count > 0) {
						if (glyphs[j].start < 7) {
							TEST_FAIL_COND(((glyphs[j].flags & TextServer::GRAPHEME_IS_RTL) == TextServer::GRAPHEME_IS_RTL), "Incorrect direction.");
						}
						if ((glyphs[j].start > 8) && (glyphs[j].start < 23)) {
							TEST_FAIL_COND(((glyphs[j].flags & TextServer::GRAPHEME_IS_RTL) != TextServer::GRAPHEME_IS_RTL), "Incorrect direction.");
						}
						if (glyphs[j].start > 26) {
							TEST_FAIL_COND(((glyphs[j].flags & TextServer::GRAPHEME_IS_RTL) == TextServer::GRAPHEME_IS_RTL), "Incorrect direction.");
						}
					}
				}

				ts->free(ctx);

				for (int j = 0; j < font.size(); j++) {
					ts->free(font[j]);
				}
				font.clear();
			}
		}

		SUBCASE("[TextServer] Text layout: Line breaking") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);

				String test_1 = U"test test test";
				//                   5^  10^

				Vector<RID> font;
				font.push_back(ts->create_font_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf"));
				font.push_back(ts->create_font_memory(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size, "ttf"));

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test_1, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				Vector<Vector2i> brks = ts->shaped_text_get_line_breaks(ctx, 1);
				TEST_FAIL_COND(brks.size() != 3, "Invalid line breaks number.");
				if (brks.size() == 3) {
					TEST_FAIL_COND(brks[0] != Vector2i(0, 5), "Invalid line break position.");
					TEST_FAIL_COND(brks[1] != Vector2i(5, 10), "Invalid line break position.");
					TEST_FAIL_COND(brks[2] != Vector2i(10, 14), "Invalid line break position.");
				}

				ts->free(ctx);

				for (int j = 0; j < font.size(); j++) {
					ts->free(font[j]);
				}
				font.clear();
			}
		}

		SUBCASE("[TextServer] Text layout: Justification") {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				TextServer *ts = TextServerManager::initialize(i, err);

				Vector<RID> font;
				font.push_back(ts->create_font_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf"));
				font.push_back(ts->create_font_memory(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size, "ttf"));

				String test_1 = U"الحمد";
				String test_2 = U"الحمد test";
				String test_3 = U"test test";
				//                    7^      26^

				RID ctx;
				bool ok;
				float width_old, width;
				if (ts->has_feature(TextServer::FEATURE_KASHIDA_JUSTIFICATION)) {
					ctx = ts->create_shaped_text();
					TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
					ok = ts->shaped_text_add_string(ctx, test_1, font, 16);
					TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

					width_old = ts->shaped_text_get_width(ctx);
					width = ts->shaped_text_fit_to_width(ctx, 100, TextServer::JUSTIFICATION_WORD_BOUND);
					TEST_FAIL_COND((width != width_old), "Invalid fill width.");
					width = ts->shaped_text_fit_to_width(ctx, 100, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA);
					TEST_FAIL_COND((width <= width_old || width > 100), "Invalid fill width.");

					ts->free(ctx);

					ctx = ts->create_shaped_text();
					TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
					ok = ts->shaped_text_add_string(ctx, test_2, font, 16);
					TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

					width_old = ts->shaped_text_get_width(ctx);
					width = ts->shaped_text_fit_to_width(ctx, 100, TextServer::JUSTIFICATION_WORD_BOUND);
					TEST_FAIL_COND((width <= width_old || width > 100), "Invalid fill width.");
					width = ts->shaped_text_fit_to_width(ctx, 100, TextServer::JUSTIFICATION_WORD_BOUND | TextServer::JUSTIFICATION_KASHIDA);
					TEST_FAIL_COND((width <= width_old || width > 100), "Invalid fill width.");

					ts->free(ctx);
				}

				ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				ok = ts->shaped_text_add_string(ctx, test_3, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				width_old = ts->shaped_text_get_width(ctx);
				width = ts->shaped_text_fit_to_width(ctx, 100, TextServer::JUSTIFICATION_WORD_BOUND);
				TEST_FAIL_COND((width <= width_old || width > 100), "Invalid fill width.");

				ts->free(ctx);

				for (int j = 0; j < font.size(); j++) {
					ts->free(font[j]);
				}
				font.clear();
			}
		}

		memdelete(tsman);
	}
}
}; // namespace TestTextServer

#endif // TEST_TEXT_SERVER_H
#endif // TOOLS_ENABLED
