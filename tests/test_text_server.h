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
		SUBCASE("[TextServer] Loading fonts") {
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				RID font = ts->create_font();
				ts->font_set_data_ptr(font, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
				TEST_FAIL_COND(font == RID(), "Loading font failed.");
				ts->free(font);
			}
		}

		SUBCASE("[TextServer] Text layout: Font fallback") {
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				RID font1 = ts->create_font();
				ts->font_set_data_ptr(font1, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
				RID font2 = ts->create_font();
				ts->font_set_data_ptr(font2, _font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size);

				Vector<RID> font;
				font.push_back(font1);
				font.push_back(font2);

				String test = U"คนอ้วน khon uan ראה";
				//                 6^       17^

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				const Glyph *glyphs = ts->shaped_text_get_glyphs(ctx);
				int gl_size = ts->shaped_text_get_glyph_count(ctx);
				TEST_FAIL_COND(gl_size == 0, "Shaping failed");
				for (int j = 0; j < gl_size; j++) {
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
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				if (!ts->has_feature(TextServer::FEATURE_BIDI_LAYOUT)) {
					continue;
				}

				RID font1 = ts->create_font();
				ts->font_set_data_ptr(font1, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
				RID font2 = ts->create_font();
				ts->font_set_data_ptr(font2, _font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size);

				Vector<RID> font;
				font.push_back(font1);
				font.push_back(font2);

				String test = U"Arabic (اَلْعَرَبِيَّةُ, al-ʿarabiyyah)";
				//                    7^      26^

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				const Glyph *glyphs = ts->shaped_text_get_glyphs(ctx);
				int gl_size = ts->shaped_text_get_glyph_count(ctx);
				TEST_FAIL_COND(gl_size == 0, "Shaping failed");
				for (int j = 0; j < gl_size; j++) {
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
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				String test_1 = U"test test test";
				//                   5^  10^

				RID font1 = ts->create_font();
				ts->font_set_data_ptr(font1, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
				RID font2 = ts->create_font();
				ts->font_set_data_ptr(font2, _font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size);

				Vector<RID> font;
				font.push_back(font1);
				font.push_back(font2);

				RID ctx = ts->create_shaped_text();
				TEST_FAIL_COND(ctx == RID(), "Creating text buffer failed.");
				bool ok = ts->shaped_text_add_string(ctx, test_1, font, 16);
				TEST_FAIL_COND(!ok, "Adding text to the buffer failed.");

				PackedInt32Array brks = ts->shaped_text_get_line_breaks(ctx, 1);
				TEST_FAIL_COND(brks.size() != 6, "Invalid line breaks number.");
				if (brks.size() == 6) {
					TEST_FAIL_COND(brks[0] != 0, "Invalid line break position.");
					TEST_FAIL_COND(brks[1] != 5, "Invalid line break position.");

					TEST_FAIL_COND(brks[2] != 5, "Invalid line break position.");
					TEST_FAIL_COND(brks[3] != 10, "Invalid line break position.");

					TEST_FAIL_COND(brks[4] != 10, "Invalid line break position.");
					TEST_FAIL_COND(brks[5] != 14, "Invalid line break position.");
				}

				ts->free(ctx);

				for (int j = 0; j < font.size(); j++) {
					ts->free(font[j]);
				}
				font.clear();
			}
		}

		SUBCASE("[TextServer] Text layout: Justification") {
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				RID font1 = ts->create_font();
				ts->font_set_data_ptr(font1, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
				RID font2 = ts->create_font();
				ts->font_set_data_ptr(font2, _font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size);

				Vector<RID> font;
				font.push_back(font1);
				font.push_back(font2);

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

		SUBCASE("[TextServer] Strip Diacritics") {
			for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
				Ref<TextServer> ts = TextServerManager::get_singleton()->get_interface(i);
				TEST_FAIL_COND(ts.is_null(), "Invalid TS interface.");

				if (ts->has_feature(TextServer::FEATURE_SHAPING)) {
					CHECK(ts->strip_diacritics(U"ٱلسَّلَامُ عَلَيْكُمْ") == U"ٱلسلام عليكم");
				}

				CHECK(ts->strip_diacritics(U"pêches épinards tomates fraises") == U"peches epinards tomates fraises");
				CHECK(ts->strip_diacritics(U"ΆΈΉΊΌΎΏΪΫϓϔ") == U"ΑΕΗΙΟΥΩΙΥΥΥ");
				CHECK(ts->strip_diacritics(U"άέήίΐϊΰϋόύώ") == U"αεηιιιυυουω");
				CHECK(ts->strip_diacritics(U"ЀЁЃ ЇЌЍӢӤЙ ЎӮӰӲ ӐӒӖӚӜӞ ӦӪ Ӭ Ӵ Ӹ") == U"ЕЕГ ІКИИИИ УУУУ ААЕӘЖЗ ОӨ Э Ч Ы");
				CHECK(ts->strip_diacritics(U"ѐёѓ їќѝӣӥй ўӯӱӳ ӑӓӗӛӝӟ ӧӫ ӭ ӵ ӹ") == U"еег ікииии уууу ааеәжз оө э ч ы");
				CHECK(ts->strip_diacritics(U"ÀÁÂÃÄÅĀĂĄÇĆĈĊČĎÈÉÊËĒĔĖĘĚĜĞĠĢĤÌÍÎÏĨĪĬĮİĴĶĹĻĽÑŃŅŇŊÒÓÔÕÖØŌŎŐƠŔŖŘŚŜŞŠŢŤÙÚÛÜŨŪŬŮŰŲƯŴÝŶŹŻŽ") == U"AAAAAAAAACCCCCDEEEEEEEEEGGGGHIIIIIIIIIJKLLLNNNNŊOOOOOØOOOORRRSSSSTTUUUUUUUUUUUWYYZZZ");
				CHECK(ts->strip_diacritics(U"àáâãäåāăąçćĉċčďèéêëēĕėęěĝğġģĥìíîïĩīĭįĵķĺļľñńņňŋòóôõöøōŏőơŕŗřśŝşšţťùúûüũūŭůűųưŵýÿŷźżž") == U"aaaaaaaaacccccdeeeeeeeeegggghiiiiiiiijklllnnnnŋoooooøoooorrrssssttuuuuuuuuuuuwyyyzzz");
				CHECK(ts->strip_diacritics(U"ǍǏȈǑǪǬȌȎȪȬȮȰǓǕǗǙǛȔȖǞǠǺȀȂȦǢǼǦǴǨǸȆȐȒȘȚȞȨ Ḁ ḂḄḆ Ḉ ḊḌḎḐḒ ḔḖḘḚḜ Ḟ Ḡ ḢḤḦḨḪ ḬḮ ḰḲḴ ḶḸḺḼ ḾṀṂ ṄṆṈṊ ṌṎṐṒ ṔṖ ṘṚṜṞ ṠṢṤṦṨ ṪṬṮṰ ṲṴṶṸṺ") == U"AIIOOOOOOOOOUUUUUUUAAAAAAÆÆGGKNERRSTHE A BBB C DDDDD EEEEE F G HHHHH II KKK LLLL MMM NNNN OOOO PP RRRR SSSSS TTTT UUUUU");
				CHECK(ts->strip_diacritics(U"ǎǐȉȋǒǫǭȍȏȫȭȯȱǔǖǘǚǜȕȗǟǡǻȁȃȧǣǽǧǵǩǹȇȑȓșțȟȩ ḁ ḃḅḇ ḉ ḋḍḏḑḓ ḟ ḡ ḭḯ ḱḳḵ ḷḹḻḽ ḿṁṃ ṅṇṉṋ ṍṏṑṓ ṗṕ ṙṛṝṟ ṡṣṥṧṩ ṫṭṯṱ ṳṵṷṹṻ") == U"aiiiooooooooouuuuuuuaaaaaaææggknerrsthe a bbb c ddddd f g ii kkk llll mmm nnnn oooo pp rrrr sssss tttt uuuuu");
				CHECK(ts->strip_diacritics(U"ṼṾ ẀẂẄẆẈ ẊẌ Ẏ ẐẒẔ") == U"VV WWWWW XX Y ZZZ");
				CHECK(ts->strip_diacritics(U"ṽṿ ẁẃẅẇẉ ẋẍ ẏ ẑẓẕ ẖ ẗẘẙẛ") == U"vv wwwww xx y zzz h twys");
			}
		}
	}
}
}; // namespace TestTextServer

#endif // TEST_TEXT_SERVER_H
#endif // TOOLS_ENABLED
