/*************************************************************************/
/*  dynamic_font_import_settings.cpp                                     */
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

#include "dynamic_font_import_settings.h"

#include "editor/editor_file_dialog.h"
#include "editor/editor_file_system.h"
#include "editor/editor_inspector.h"
#include "editor/editor_locale_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"

/*************************************************************************/
/* Settings data                                                         */
/*************************************************************************/

class DynamicFontImportSettingsData : public RefCounted {
	GDCLASS(DynamicFontImportSettingsData, RefCounted)
	friend class DynamicFontImportSettings;

	Map<StringName, Variant> settings;
	Map<StringName, Variant> defaults;
	List<ResourceImporter::ImportOption> options;
	DynamicFontImportSettings *owner = nullptr;

	bool _set(const StringName &p_name, const Variant &p_value) {
		if (defaults.has(p_name) && defaults[p_name] == p_value) {
			settings.erase(p_name);
		} else {
			settings[p_name] = p_value;
		}
		return true;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (settings.has(p_name)) {
			r_ret = settings[p_name];
			return true;
		}
		if (defaults.has(p_name)) {
			r_ret = defaults[p_name];
			return true;
		}
		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		for (const List<ResourceImporter::ImportOption>::Element *E = options.front(); E; E = E->next()) {
			if (owner && owner->import_settings_data.is_valid()) {
				if (owner->import_settings_data->get("multichannel_signed_distance_field") && (E->get().option.name == "size" || E->get().option.name == "outline_size" || E->get().option.name == "oversampling")) {
					continue;
				}
				if (!owner->import_settings_data->get("multichannel_signed_distance_field") && (E->get().option.name == "msdf_pixel_range" || E->get().option.name == "msdf_size")) {
					continue;
				}
			}
			p_list->push_back(E->get().option);
		}
	}
};

/*************************************************************************/
/* Glyph ranges                                                          */
/*************************************************************************/

struct UniRange {
	int32_t start;
	int32_t end;
	String name;
};

// Unicode Character Blocks
// Source: https://www.unicode.org/Public/14.0.0/ucd/Blocks.txt
static UniRange unicode_ranges[] = {
	{ 0x0000, 0x007f, U"Basic Latin" },
	{ 0x0080, 0x00ff, U"Latin-1 Supplement" },
	{ 0x0100, 0x017f, U"Latin Extended-A" },
	{ 0x0180, 0x024f, U"Latin Extended-B" },
	{ 0x0250, 0x02af, U"IPA Extensions" },
	{ 0x02b0, 0x02ff, U"Spacing Modifier Letters" },
	{ 0x0300, 0x036f, U"Combining Diacritical Marks" },
	{ 0x0370, 0x03ff, U"Greek and Coptic" },
	{ 0x0400, 0x04ff, U"Cyrillic" },
	{ 0x0500, 0x052f, U"Cyrillic Supplement" },
	{ 0x0530, 0x058f, U"Armenian" },
	{ 0x0590, 0x05ff, U"Hebrew" },
	{ 0x0600, 0x06ff, U"Arabic" },
	{ 0x0700, 0x074f, U"Syriac" },
	{ 0x0750, 0x077f, U"Arabic Supplement" },
	{ 0x0780, 0x07bf, U"Thaana" },
	{ 0x07c0, 0x07ff, U"NKo" },
	{ 0x0800, 0x083f, U"Samaritan" },
	{ 0x0840, 0x085f, U"Mandaic" },
	{ 0x0860, 0x086f, U"Syriac Supplement" },
	{ 0x0870, 0x089f, U"Arabic Extended-B" },
	{ 0x08a0, 0x08ff, U"Arabic Extended-A" },
	{ 0x0900, 0x097f, U"Devanagari" },
	{ 0x0980, 0x09ff, U"Bengali" },
	{ 0x0a00, 0x0a7f, U"Gurmukhi" },
	{ 0x0a80, 0x0aff, U"Gujarati" },
	{ 0x0b00, 0x0b7f, U"Oriya" },
	{ 0x0b80, 0x0bff, U"Tamil" },
	{ 0x0c00, 0x0c7f, U"Telugu" },
	{ 0x0c80, 0x0cff, U"Kannada" },
	{ 0x0d00, 0x0d7f, U"Malayalam" },
	{ 0x0d80, 0x0dff, U"Sinhala" },
	{ 0x0e00, 0x0e7f, U"Thai" },
	{ 0x0e80, 0x0eff, U"Lao" },
	{ 0x0f00, 0x0fff, U"Tibetan" },
	{ 0x1000, 0x109f, U"Myanmar" },
	{ 0x10a0, 0x10ff, U"Georgian" },
	{ 0x1100, 0x11ff, U"Hangul Jamo" },
	{ 0x1200, 0x137f, U"Ethiopic" },
	{ 0x1380, 0x139f, U"Ethiopic Supplement" },
	{ 0x13a0, 0x13ff, U"Cherokee" },
	{ 0x1400, 0x167f, U"Unified Canadian Aboriginal Syllabics" },
	{ 0x1680, 0x169f, U"Ogham" },
	{ 0x16a0, 0x16ff, U"Runic" },
	{ 0x1700, 0x171f, U"Tagalog" },
	{ 0x1720, 0x173f, U"Hanunoo" },
	{ 0x1740, 0x175f, U"Buhid" },
	{ 0x1760, 0x177f, U"Tagbanwa" },
	{ 0x1780, 0x17ff, U"Khmer" },
	{ 0x1800, 0x18af, U"Mongolian" },
	{ 0x18b0, 0x18ff, U"Unified Canadian Aboriginal Syllabics Extended" },
	{ 0x1900, 0x194f, U"Limbu" },
	{ 0x1950, 0x197f, U"Tai Le" },
	{ 0x1980, 0x19df, U"New Tai Lue" },
	{ 0x19e0, 0x19ff, U"Khmer Symbols" },
	{ 0x1a00, 0x1a1f, U"Buginese" },
	{ 0x1a20, 0x1aaf, U"Tai Tham" },
	{ 0x1ab0, 0x1aff, U"Combining Diacritical Marks Extended" },
	{ 0x1b00, 0x1b7f, U"Balinese" },
	{ 0x1b80, 0x1bbf, U"Sundanese" },
	{ 0x1bc0, 0x1bff, U"Batak" },
	{ 0x1c00, 0x1c4f, U"Lepcha" },
	{ 0x1c50, 0x1c7f, U"Ol Chiki" },
	{ 0x1c80, 0x1c8f, U"Cyrillic Extended-C" },
	{ 0x1c90, 0x1cbf, U"Georgian Extended" },
	{ 0x1cc0, 0x1ccf, U"Sundanese Supplement" },
	{ 0x1cd0, 0x1cff, U"Vedic Extensions" },
	{ 0x1d00, 0x1d7f, U"Phonetic Extensions" },
	{ 0x1d80, 0x1dbf, U"Phonetic Extensions Supplement" },
	{ 0x1dc0, 0x1dff, U"Combining Diacritical Marks Supplement" },
	{ 0x1e00, 0x1eff, U"Latin Extended Additional" },
	{ 0x1f00, 0x1fff, U"Greek Extended" },
	{ 0x2000, 0x206f, U"General Punctuation" },
	{ 0x2070, 0x209f, U"Superscripts and Subscripts" },
	{ 0x20a0, 0x20cf, U"Currency Symbols" },
	{ 0x20d0, 0x20ff, U"Combining Diacritical Marks for Symbols" },
	{ 0x2100, 0x214f, U"Letterlike Symbols" },
	{ 0x2150, 0x218f, U"Number Forms" },
	{ 0x2190, 0x21ff, U"Arrows" },
	{ 0x2200, 0x22ff, U"Mathematical Operators" },
	{ 0x2300, 0x23ff, U"Miscellaneous Technical" },
	{ 0x2400, 0x243f, U"Control Pictures" },
	{ 0x2440, 0x245f, U"Optical Character Recognition" },
	{ 0x2460, 0x24ff, U"Enclosed Alphanumerics" },
	{ 0x2500, 0x257f, U"Box Drawing" },
	{ 0x2580, 0x259f, U"Block Elements" },
	{ 0x25a0, 0x25ff, U"Geometric Shapes" },
	{ 0x2600, 0x26ff, U"Miscellaneous Symbols" },
	{ 0x2700, 0x27bf, U"Dingbats" },
	{ 0x27c0, 0x27ef, U"Miscellaneous Mathematical Symbols-A" },
	{ 0x27f0, 0x27ff, U"Supplemental Arrows-A" },
	{ 0x2800, 0x28ff, U"Braille Patterns" },
	{ 0x2900, 0x297f, U"Supplemental Arrows-B" },
	{ 0x2980, 0x29ff, U"Miscellaneous Mathematical Symbols-B" },
	{ 0x2a00, 0x2aff, U"Supplemental Mathematical Operators" },
	{ 0x2b00, 0x2bff, U"Miscellaneous Symbols and Arrows" },
	{ 0x2c00, 0x2c5f, U"Glagolitic" },
	{ 0x2c60, 0x2c7f, U"Latin Extended-C" },
	{ 0x2c80, 0x2cff, U"Coptic" },
	{ 0x2d00, 0x2d2f, U"Georgian Supplement" },
	{ 0x2d30, 0x2d7f, U"Tifinagh" },
	{ 0x2d80, 0x2ddf, U"Ethiopic Extended" },
	{ 0x2de0, 0x2dff, U"Cyrillic Extended-A" },
	{ 0x2e00, 0x2e7f, U"Supplemental Punctuation" },
	{ 0x2e80, 0x2eff, U"CJK Radicals Supplement" },
	{ 0x2f00, 0x2fdf, U"Kangxi Radicals" },
	{ 0x2ff0, 0x2fff, U"Ideographic Description Characters" },
	{ 0x3000, 0x303f, U"CJK Symbols and Punctuation" },
	{ 0x3040, 0x309f, U"Hiragana" },
	{ 0x30a0, 0x30ff, U"Katakana" },
	{ 0x3100, 0x312f, U"Bopomofo" },
	{ 0x3130, 0x318f, U"Hangul Compatibility Jamo" },
	{ 0x3190, 0x319f, U"Kanbun" },
	{ 0x31a0, 0x31bf, U"Bopomofo Extended" },
	{ 0x31c0, 0x31ef, U"CJK Strokes" },
	{ 0x31f0, 0x31ff, U"Katakana Phonetic Extensions" },
	{ 0x3200, 0x32ff, U"Enclosed CJK Letters and Months" },
	{ 0x3300, 0x33ff, U"CJK Compatibility" },
	{ 0x3400, 0x4dbf, U"CJK Unified Ideographs Extension A" },
	{ 0x4dc0, 0x4dff, U"Yijing Hexagram Symbols" },
	{ 0x4e00, 0x9fff, U"CJK Unified Ideographs" },
	{ 0xa000, 0xa48f, U"Yi Syllables" },
	{ 0xa490, 0xa4cf, U"Yi Radicals" },
	{ 0xa4d0, 0xa4ff, U"Lisu" },
	{ 0xa500, 0xa63f, U"Vai" },
	{ 0xa640, 0xa69f, U"Cyrillic Extended-B" },
	{ 0xa6a0, 0xa6ff, U"Bamum" },
	{ 0xa700, 0xa71f, U"Modifier Tone Letters" },
	{ 0xa720, 0xa7ff, U"Latin Extended-D" },
	{ 0xa800, 0xa82f, U"Syloti Nagri" },
	{ 0xa830, 0xa83f, U"Common Indic Number Forms" },
	{ 0xa840, 0xa87f, U"Phags-pa" },
	{ 0xa880, 0xa8df, U"Saurashtra" },
	{ 0xa8e0, 0xa8ff, U"Devanagari Extended" },
	{ 0xa900, 0xa92f, U"Kayah Li" },
	{ 0xa930, 0xa95f, U"Rejang" },
	{ 0xa960, 0xa97f, U"Hangul Jamo Extended-A" },
	{ 0xa980, 0xa9df, U"Javanese" },
	{ 0xa9e0, 0xa9ff, U"Myanmar Extended-B" },
	{ 0xaa00, 0xaa5f, U"Cham" },
	{ 0xaa60, 0xaa7f, U"Myanmar Extended-A" },
	{ 0xaa80, 0xaadf, U"Tai Viet" },
	{ 0xaae0, 0xaaff, U"Meetei Mayek Extensions" },
	{ 0xab00, 0xab2f, U"Ethiopic Extended-A" },
	{ 0xab30, 0xab6f, U"Latin Extended-E" },
	{ 0xab70, 0xabbf, U"Cherokee Supplement" },
	{ 0xabc0, 0xabff, U"Meetei Mayek" },
	{ 0xac00, 0xd7af, U"Hangul Syllables" },
	{ 0xd7b0, 0xd7ff, U"Hangul Jamo Extended-B" },
	//{ 0xd800, 0xdb7f, U"High Surrogates" },
	//{ 0xdb80, 0xdbff, U"High Private Use Surrogates" },
	//{ 0xdc00, 0xdfff, U"Low Surrogates" },
	{ 0xe000, 0xf8ff, U"Private Use Area" },
	{ 0xf900, 0xfaff, U"CJK Compatibility Ideographs" },
	{ 0xfb00, 0xfb4f, U"Alphabetic Presentation Forms" },
	{ 0xfb50, 0xfdff, U"Arabic Presentation Forms-A" },
	//{ 0xfe00, 0xfe0f, U"Variation Selectors" },
	{ 0xfe10, 0xfe1f, U"Vertical Forms" },
	{ 0xfe20, 0xfe2f, U"Combining Half Marks" },
	{ 0xfe30, 0xfe4f, U"CJK Compatibility Forms" },
	{ 0xfe50, 0xfe6f, U"Small Form Variants" },
	{ 0xfe70, 0xfeff, U"Arabic Presentation Forms-B" },
	{ 0xff00, 0xffef, U"Halfwidth and Fullwidth Forms" },
	//{ 0xfff0, 0xffff, U"Specials" },
	{ 0x10000, 0x1007f, U"Linear B Syllabary" },
	{ 0x10080, 0x100ff, U"Linear B Ideograms" },
	{ 0x10100, 0x1013f, U"Aegean Numbers" },
	{ 0x10140, 0x1018f, U"Ancient Greek Numbers" },
	{ 0x10190, 0x101cf, U"Ancient Symbols" },
	{ 0x101d0, 0x101ff, U"Phaistos Disc" },
	{ 0x10280, 0x1029f, U"Lycian" },
	{ 0x102a0, 0x102df, U"Carian" },
	{ 0x102e0, 0x102ff, U"Coptic Epact Numbers" },
	{ 0x10300, 0x1032f, U"Old Italic" },
	{ 0x10330, 0x1034f, U"Gothic" },
	{ 0x10350, 0x1037f, U"Old Permic" },
	{ 0x10380, 0x1039f, U"Ugaritic" },
	{ 0x103a0, 0x103df, U"Old Persian" },
	{ 0x10400, 0x1044f, U"Deseret" },
	{ 0x10450, 0x1047f, U"Shavian" },
	{ 0x10480, 0x104af, U"Osmanya" },
	{ 0x104b0, 0x104ff, U"Osage" },
	{ 0x10500, 0x1052f, U"Elbasan" },
	{ 0x10530, 0x1056f, U"Caucasian Albanian" },
	{ 0x10570, 0x105bf, U"Vithkuqi" },
	{ 0x10600, 0x1077f, U"Linear A" },
	{ 0x10780, 0x107bf, U"Latin Extended-F" },
	{ 0x10800, 0x1083f, U"Cypriot Syllabary" },
	{ 0x10840, 0x1085f, U"Imperial Aramaic" },
	{ 0x10860, 0x1087f, U"Palmyrene" },
	{ 0x10880, 0x108af, U"Nabataean" },
	{ 0x108e0, 0x108ff, U"Hatran" },
	{ 0x10900, 0x1091f, U"Phoenician" },
	{ 0x10920, 0x1093f, U"Lydian" },
	{ 0x10980, 0x1099f, U"Meroitic Hieroglyphs" },
	{ 0x109a0, 0x109ff, U"Meroitic Cursive" },
	{ 0x10a00, 0x10a5f, U"Kharoshthi" },
	{ 0x10a60, 0x10a7f, U"Old South Arabian" },
	{ 0x10a80, 0x10a9f, U"Old North Arabian" },
	{ 0x10ac0, 0x10aff, U"Manichaean" },
	{ 0x10b00, 0x10b3f, U"Avestan" },
	{ 0x10b40, 0x10b5f, U"Inscriptional Parthian" },
	{ 0x10b60, 0x10b7f, U"Inscriptional Pahlavi" },
	{ 0x10b80, 0x10baf, U"Psalter Pahlavi" },
	{ 0x10c00, 0x10c4f, U"Old Turkic" },
	{ 0x10c80, 0x10cff, U"Old Hungarian" },
	{ 0x10d00, 0x10d3f, U"Hanifi Rohingya" },
	{ 0x10e60, 0x10e7f, U"Rumi Numeral Symbols" },
	{ 0x10e80, 0x10ebf, U"Yezidi" },
	{ 0x10f00, 0x10f2f, U"Old Sogdian" },
	{ 0x10f30, 0x10f6f, U"Sogdian" },
	{ 0x10f70, 0x10faf, U"Old Uyghur" },
	{ 0x10fb0, 0x10fdf, U"Chorasmian" },
	{ 0x10fe0, 0x10fff, U"Elymaic" },
	{ 0x11000, 0x1107f, U"Brahmi" },
	{ 0x11080, 0x110cf, U"Kaithi" },
	{ 0x110d0, 0x110ff, U"Sora Sompeng" },
	{ 0x11100, 0x1114f, U"Chakma" },
	{ 0x11150, 0x1117f, U"Mahajani" },
	{ 0x11180, 0x111df, U"Sharada" },
	{ 0x111e0, 0x111ff, U"Sinhala Archaic Numbers" },
	{ 0x11200, 0x1124f, U"Khojki" },
	{ 0x11280, 0x112af, U"Multani" },
	{ 0x112b0, 0x112ff, U"Khudawadi" },
	{ 0x11300, 0x1137f, U"Grantha" },
	{ 0x11400, 0x1147f, U"Newa" },
	{ 0x11480, 0x114df, U"Tirhuta" },
	{ 0x11580, 0x115ff, U"Siddham" },
	{ 0x11600, 0x1165f, U"Modi" },
	{ 0x11660, 0x1167f, U"Mongolian Supplement" },
	{ 0x11680, 0x116cf, U"Takri" },
	{ 0x11700, 0x1174f, U"Ahom" },
	{ 0x11800, 0x1184f, U"Dogra" },
	{ 0x118a0, 0x118ff, U"Warang Citi" },
	{ 0x11900, 0x1195f, U"Dives Akuru" },
	{ 0x119a0, 0x119ff, U"Nandinagari" },
	{ 0x11a00, 0x11a4f, U"Zanabazar Square" },
	{ 0x11a50, 0x11aaf, U"Soyombo" },
	{ 0x11ab0, 0x11abf, U"Unified Canadian Aboriginal Syllabics Extended-A" },
	{ 0x11ac0, 0x11aff, U"Pau Cin Hau" },
	{ 0x11c00, 0x11c6f, U"Bhaiksuki" },
	{ 0x11c70, 0x11cbf, U"Marchen" },
	{ 0x11d00, 0x11d5f, U"Masaram Gondi" },
	{ 0x11d60, 0x11daf, U"Gunjala Gondi" },
	{ 0x11ee0, 0x11eff, U"Makasar" },
	{ 0x11fb0, 0x11fbf, U"Lisu Supplement" },
	{ 0x11fc0, 0x11fff, U"Tamil Supplement" },
	{ 0x12000, 0x123ff, U"Cuneiform" },
	{ 0x12400, 0x1247f, U"Cuneiform Numbers and Punctuation" },
	{ 0x12480, 0x1254f, U"Early Dynastic Cuneiform" },
	{ 0x12f90, 0x12fff, U"Cypro-Minoan" },
	{ 0x13000, 0x1342f, U"Egyptian Hieroglyphs" },
	{ 0x13430, 0x1343f, U"Egyptian Hieroglyph Format Controls" },
	{ 0x14400, 0x1467f, U"Anatolian Hieroglyphs" },
	{ 0x16800, 0x16a3f, U"Bamum Supplement" },
	{ 0x16a40, 0x16a6f, U"Mro" },
	{ 0x16a70, 0x16acf, U"Tangsa" },
	{ 0x16ad0, 0x16aff, U"Bassa Vah" },
	{ 0x16b00, 0x16b8f, U"Pahawh Hmong" },
	{ 0x16e40, 0x16e9f, U"Medefaidrin" },
	{ 0x16f00, 0x16f9f, U"Miao" },
	{ 0x16fe0, 0x16fff, U"Ideographic Symbols and Punctuation" },
	{ 0x17000, 0x187ff, U"Tangut" },
	{ 0x18800, 0x18aff, U"Tangut Components" },
	{ 0x18b00, 0x18cff, U"Khitan Small Script" },
	{ 0x18d00, 0x18d7f, U"Tangut Supplement" },
	{ 0x1aff0, 0x1afff, U"Kana Extended-B" },
	{ 0x1b000, 0x1b0ff, U"Kana Supplement" },
	{ 0x1b100, 0x1b12f, U"Kana Extended-A" },
	{ 0x1b130, 0x1b16f, U"Small Kana Extension" },
	{ 0x1b170, 0x1b2ff, U"Nushu" },
	{ 0x1bc00, 0x1bc9f, U"Duployan" },
	{ 0x1bca0, 0x1bcaf, U"Shorthand Format Controls" },
	{ 0x1cf00, 0x1cfcf, U"Znamenny Musical Notation" },
	{ 0x1d000, 0x1d0ff, U"Byzantine Musical Symbols" },
	{ 0x1d100, 0x1d1ff, U"Musical Symbols" },
	{ 0x1d200, 0x1d24f, U"Ancient Greek Musical Notation" },
	{ 0x1d2e0, 0x1d2ff, U"Mayan Numerals" },
	{ 0x1d300, 0x1d35f, U"Tai Xuan Jing Symbols" },
	{ 0x1d360, 0x1d37f, U"Counting Rod Numerals" },
	{ 0x1d400, 0x1d7ff, U"Mathematical Alphanumeric Symbols" },
	{ 0x1d800, 0x1daaf, U"Sutton SignWriting" },
	{ 0x1df00, 0x1dfff, U"Latin Extended-G" },
	{ 0x1e000, 0x1e02f, U"Glagolitic Supplement" },
	{ 0x1e100, 0x1e14f, U"Nyiakeng Puachue Hmong" },
	{ 0x1e290, 0x1e2bf, U"Toto" },
	{ 0x1e2c0, 0x1e2ff, U"Wancho" },
	{ 0x1e7e0, 0x1e7ff, U"Ethiopic Extended-B" },
	{ 0x1e800, 0x1e8df, U"Mende Kikakui" },
	{ 0x1e900, 0x1e95f, U"Adlam" },
	{ 0x1ec70, 0x1ecbf, U"Indic Siyaq Numbers" },
	{ 0x1ed00, 0x1ed4f, U"Ottoman Siyaq Numbers" },
	{ 0x1ee00, 0x1eeff, U"Arabic Mathematical Alphabetic Symbols" },
	{ 0x1f000, 0x1f02f, U"Mahjong Tiles" },
	{ 0x1f030, 0x1f09f, U"Domino Tiles" },
	{ 0x1f0a0, 0x1f0ff, U"Playing Cards" },
	{ 0x1f100, 0x1f1ff, U"Enclosed Alphanumeric Supplement" },
	{ 0x1f200, 0x1f2ff, U"Enclosed Ideographic Supplement" },
	{ 0x1f300, 0x1f5ff, U"Miscellaneous Symbols and Pictographs" },
	{ 0x1f600, 0x1f64f, U"Emoticons" },
	{ 0x1f650, 0x1f67f, U"Ornamental Dingbats" },
	{ 0x1f680, 0x1f6ff, U"Transport and Map Symbols" },
	{ 0x1f700, 0x1f77f, U"Alchemical Symbols" },
	{ 0x1f780, 0x1f7ff, U"Geometric Shapes Extended" },
	{ 0x1f800, 0x1f8ff, U"Supplemental Arrows-C" },
	{ 0x1f900, 0x1f9ff, U"Supplemental Symbols and Pictographs" },
	{ 0x1fa00, 0x1fa6f, U"Chess Symbols" },
	{ 0x1fa70, 0x1faff, U"Symbols and Pictographs Extended-A" },
	{ 0x1fb00, 0x1fbff, U"Symbols for Legacy Computing" },
	{ 0x20000, 0x2a6df, U"CJK Unified Ideographs Extension B" },
	{ 0x2a700, 0x2b73f, U"CJK Unified Ideographs Extension C" },
	{ 0x2b740, 0x2b81f, U"CJK Unified Ideographs Extension D" },
	{ 0x2b820, 0x2ceaf, U"CJK Unified Ideographs Extension E" },
	{ 0x2ceb0, 0x2ebef, U"CJK Unified Ideographs Extension F" },
	{ 0x2f800, 0x2fa1f, U"CJK Compatibility Ideographs Supplement" },
	{ 0x30000, 0x3134f, U"CJK Unified Ideographs Extension G" },
	//{ 0xe0000, 0xe007f, U"Tags" },
	//{ 0xe0100, 0xe01ef, U"Variation Selectors Supplement" },
	{ 0xf0000, 0xfffff, U"Supplementary Private Use Area-A" },
	{ 0x100000, 0x10ffff, U"Supplementary Private Use Area-B" },
	{ 0x10ffff, 0x10ffff, String() }
};

void DynamicFontImportSettings::_add_glyph_range_item(int32_t p_start, int32_t p_end, const String &p_name) {
	const int page_size = 512;
	int pages = (p_end - p_start) / page_size;
	int remain = (p_end - p_start) % page_size;

	int32_t start = p_start;
	for (int i = 0; i < pages; i++) {
		TreeItem *item = glyph_tree->create_item(glyph_root);
		ERR_FAIL_NULL(item);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_text(0, _pad_zeros(String::num_int64(start, 16)) + " - " + _pad_zeros(String::num_int64(start + page_size, 16)));
		item->set_text(1, p_name);
		item->set_metadata(0, Vector2i(start, start + page_size));
		start += page_size;
	}
	if (remain > 0) {
		TreeItem *item = glyph_tree->create_item(glyph_root);
		ERR_FAIL_NULL(item);
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_text(0, _pad_zeros(String::num_int64(start, 16)) + " - " + _pad_zeros(String::num_int64(p_end, 16)));
		item->set_text(1, p_name);
		item->set_metadata(0, Vector2i(start, p_end));
	}
}

/*************************************************************************/
/* Page 1 callbacks: Rendering Options                                   */
/*************************************************************************/

void DynamicFontImportSettings::_main_prop_changed(const String &p_edited_property) {
	// Update font preview.

	if (p_edited_property == "antialiased") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_antialiased(import_settings_data->get("antialiased"));
		}
	} else if (p_edited_property == "generate_mipmaps") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_generate_mipmaps(import_settings_data->get("generate_mipmaps"));
		}
	} else if (p_edited_property == "multichannel_signed_distance_field") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_multichannel_signed_distance_field(import_settings_data->get("multichannel_signed_distance_field"));
		}
		_variation_selected();
		_variations_validate();
	} else if (p_edited_property == "msdf_pixel_range") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_msdf_pixel_range(import_settings_data->get("msdf_pixel_range"));
		}
	} else if (p_edited_property == "msdf_size") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_msdf_size(import_settings_data->get("msdf_size"));
		}
	} else if (p_edited_property == "force_autohinter") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_force_autohinter(import_settings_data->get("force_autohinter"));
		}
	} else if (p_edited_property == "hinting") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_hinting((TextServer::Hinting)import_settings_data->get("hinting").operator int());
		}
	} else if (p_edited_property == "subpixel_positioning") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_subpixel_positioning((TextServer::SubpixelPositioning)import_settings_data->get("subpixel_positioning").operator int());
		}
	} else if (p_edited_property == "embolden") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_embolden(import_settings_data->get("embolden"));
		}
	} else if (p_edited_property == "transform") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_transform(import_settings_data->get("transform"));
		}
	} else if (p_edited_property == "oversampling") {
		if (font_preview->get_data_count() > 0) {
			font_preview->get_data(0)->set_oversampling(import_settings_data->get("oversampling"));
		}
	}
	font_preview_label->add_theme_font_override("font", font_preview);
	font_preview_label->update();
}

/*************************************************************************/
/* Page 2 callbacks: Configurations                                      */
/*************************************************************************/

void DynamicFontImportSettings::_variation_add() {
	TreeItem *vars_item = vars_list->create_item(vars_list_root);
	ERR_FAIL_NULL(vars_item);

	vars_item->set_text(0, TTR("New configuration"));
	vars_item->set_editable(0, true);
	vars_item->add_button(1, vars_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove Variation"));
	vars_item->set_button_color(1, 0, Color(1, 1, 1, 0.75));

	Ref<DynamicFontImportSettingsData> import_variation_data;
	import_variation_data.instantiate();
	import_variation_data->owner = this;
	ERR_FAIL_NULL(import_variation_data);

	for (List<ResourceImporter::ImportOption>::Element *E = options_variations.front(); E; E = E->next()) {
		import_variation_data->defaults[E->get().option.name] = E->get().default_value;
	}

	import_variation_data->options = options_variations;
	inspector_vars->edit(import_variation_data.ptr());
	import_variation_data->notify_property_list_changed();

	vars_item->set_metadata(0, import_variation_data);

	_variations_validate();
}

void DynamicFontImportSettings::_variation_selected() {
	TreeItem *vars_item = vars_list->get_selected();
	if (vars_item) {
		Ref<DynamicFontImportSettingsData> import_variation_data = vars_item->get_metadata(0);
		ERR_FAIL_NULL(import_variation_data);

		inspector_vars->edit(import_variation_data.ptr());
		import_variation_data->notify_property_list_changed();
	}
}

void DynamicFontImportSettings::_variation_remove(Object *p_item, int p_column, int p_id) {
	TreeItem *vars_item = (TreeItem *)p_item;
	ERR_FAIL_NULL(vars_item);

	inspector_vars->edit(nullptr);

	vars_list_root->remove_child(vars_item);
	memdelete(vars_item);

	if (vars_list_root->get_first_child()) {
		Ref<DynamicFontImportSettingsData> import_variation_data = vars_list_root->get_first_child()->get_metadata(0);
		inspector_vars->edit(import_variation_data.ptr());
		import_variation_data->notify_property_list_changed();
	}

	_variations_validate();
}

void DynamicFontImportSettings::_variation_changed(const String &p_edited_property) {
	_variations_validate();
}

void DynamicFontImportSettings::_variations_validate() {
	String warn;
	if (!vars_list_root->get_first_child()) {
		warn = TTR("Warning: There are no configurations specified, no glyphs will be pre-rendered.");
	}
	for (TreeItem *vars_item_a = vars_list_root->get_first_child(); vars_item_a; vars_item_a = vars_item_a->get_next()) {
		Ref<DynamicFontImportSettingsData> import_variation_data_a = vars_item_a->get_metadata(0);
		ERR_FAIL_NULL(import_variation_data_a);

		for (TreeItem *vars_item_b = vars_list_root->get_first_child(); vars_item_b; vars_item_b = vars_item_b->get_next()) {
			if (vars_item_b != vars_item_a) {
				bool match = true;
				for (Map<StringName, Variant>::Element *E = import_variation_data_a->settings.front(); E; E = E->next()) {
					Ref<DynamicFontImportSettingsData> import_variation_data_b = vars_item_b->get_metadata(0);
					ERR_FAIL_NULL(import_variation_data_b);
					match = match && (import_variation_data_b->settings[E->key()] == E->get());
				}
				if (match) {
					warn = TTR("Warning: Multiple configurations have identical settings. Duplicates will be ignored.");
					break;
				}
			}
		}
	}
	if (warn.is_empty()) {
		label_warn->set_text("");
		label_warn->hide();
	} else {
		label_warn->set_text(warn);
		label_warn->show();
	}
}

/*************************************************************************/
/* Page 3 callbacks: Text to select glyphs                               */
/*************************************************************************/

void DynamicFontImportSettings::_change_text_opts() {
	Vector<String> ftr = ftr_edit->get_text().split(",");
	for (int i = 0; i < ftr.size(); i++) {
		Vector<String> tokens = ftr[i].split("=");
		if (tokens.size() == 2) {
			text_edit->set_opentype_feature(tokens[0], tokens[1].to_int());
		} else if (tokens.size() == 1) {
			text_edit->set_opentype_feature(tokens[0], 1);
		}
	}
	text_edit->set_language(lang_edit->get_text());
}

void DynamicFontImportSettings::_glyph_clear() {
	selected_glyphs.clear();
	label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(selected_glyphs.size()));
	_range_selected();
}

void DynamicFontImportSettings::_glyph_text_selected() {
	Dictionary ftrs;
	Vector<String> ftr = ftr_edit->get_text().split(",");
	for (int i = 0; i < ftr.size(); i++) {
		Vector<String> tokens = ftr[i].split("=");
		if (tokens.size() == 2) {
			ftrs[tokens[0]] = tokens[1].to_int();
		} else if (tokens.size() == 1) {
			ftrs[tokens[0]] = 1;
		}
	}

	RID text_rid = TS->create_shaped_text();
	if (text_rid.is_valid()) {
		TS->shaped_text_add_string(text_rid, text_edit->get_text(), font_main->get_rids(), 16, ftrs, text_edit->get_language());
		TS->shaped_text_shape(text_rid);
		const Glyph *gl = TS->shaped_text_get_glyphs(text_rid);
		const int gl_size = TS->shaped_text_get_glyph_count(text_rid);

		for (int i = 0; i < gl_size; i++) {
			if (gl[i].font_rid.is_valid() && gl[i].index != 0) {
				selected_glyphs.insert(gl[i].index);
			}
		}
		TS->free_rid(text_rid);
		label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(selected_glyphs.size()));
	}
	_range_selected();
}

/*************************************************************************/
/* Page 4 callbacks: Character map                                       */
/*************************************************************************/

void DynamicFontImportSettings::_glyph_selected() {
	TreeItem *item = glyph_table->get_selected();
	ERR_FAIL_NULL(item);

	Color scol = glyph_table->get_theme_color(SNAME("box_selection_fill_color"), SNAME("Editor"));
	Color fcol = glyph_table->get_theme_color(SNAME("font_selected_color"), SNAME("Editor"));
	scol.a = 1.f;

	int32_t c = item->get_metadata(glyph_table->get_selected_column());
	if (font_main->has_char(c)) {
		if (_char_update(c)) {
			item->set_custom_color(glyph_table->get_selected_column(), fcol);
			item->set_custom_bg_color(glyph_table->get_selected_column(), scol);
		} else {
			item->clear_custom_color(glyph_table->get_selected_column());
			item->clear_custom_bg_color(glyph_table->get_selected_column());
		}
	}
	label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(selected_glyphs.size()));

	item = glyph_tree->get_selected();
	ERR_FAIL_NULL(item);
	Vector2i range = item->get_metadata(0);

	int total_chars = range.y - range.x;
	int selected_count = 0;
	for (int i = range.x; i < range.y; i++) {
		if (!font_main->has_char(i)) {
			total_chars--;
		}

		if (selected_chars.has(i)) {
			selected_count++;
		}
	}

	if (selected_count == total_chars) {
		item->set_checked(0, true);
	} else if (selected_count > 0) {
		item->set_indeterminate(0, true);
	} else {
		item->set_checked(0, false);
	}
}

void DynamicFontImportSettings::_range_edited() {
	TreeItem *item = glyph_tree->get_selected();
	ERR_FAIL_NULL(item);
	Vector2i range = item->get_metadata(0);
	_range_update(range.x, range.y);
}

void DynamicFontImportSettings::_range_selected() {
	TreeItem *item = glyph_tree->get_selected();
	if (item) {
		Vector2i range = item->get_metadata(0);
		_edit_range(range.x, range.y);
	}
}

void DynamicFontImportSettings::_edit_range(int32_t p_start, int32_t p_end) {
	glyph_table->clear();

	TreeItem *root = glyph_table->create_item();
	ERR_FAIL_NULL(root);

	Color scol = glyph_table->get_theme_color(SNAME("box_selection_fill_color"), SNAME("Editor"));
	Color fcol = glyph_table->get_theme_color(SNAME("font_selected_color"), SNAME("Editor"));
	scol.a = 1.f;

	TreeItem *item = nullptr;
	int col = 0;

	for (int32_t c = p_start; c <= p_end; c++) {
		if (col == 0) {
			item = glyph_table->create_item(root);
			ERR_FAIL_NULL(item);
			item->set_text(0, _pad_zeros(String::num_int64(c, 16)));
			item->set_text_alignment(0, HORIZONTAL_ALIGNMENT_LEFT);
			item->set_selectable(0, false);
			item->set_custom_bg_color(0, glyph_table->get_theme_color(SNAME("dark_color_3"), SNAME("Editor")));
		}
		if (font_main->has_char(c)) {
			item->set_text(col + 1, String::chr(c));
			item->set_custom_color(col + 1, Color(1, 1, 1));
			if (selected_chars.has(c) || (font_main->get_data(0).is_valid() && selected_glyphs.has(font_main->get_data(0)->get_glyph_index(get_theme_font_size(SNAME("font_size")) * 2, c)))) {
				item->set_custom_color(col + 1, fcol);
				item->set_custom_bg_color(col + 1, scol);
			} else {
				item->clear_custom_color(col + 1);
				item->clear_custom_bg_color(col + 1);
			}
		} else {
			item->set_custom_bg_color(col + 1, glyph_table->get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
		}
		item->set_metadata(col + 1, c);
		item->set_text_alignment(col + 1, HORIZONTAL_ALIGNMENT_CENTER);
		item->set_selectable(col + 1, true);
		item->set_custom_font(col + 1, font_main);
		item->set_custom_font_size(col + 1, get_theme_font_size(SNAME("font_size")) * 2);

		col++;
		if (col == 16) {
			col = 0;
		}
	}
	label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(selected_glyphs.size()));
}

bool DynamicFontImportSettings::_char_update(int32_t p_char) {
	if (selected_chars.has(p_char)) {
		selected_chars.erase(p_char);
		return false;
	} else if (font_main->get_data(0).is_valid() && selected_glyphs.has(font_main->get_data(0)->get_glyph_index(get_theme_font_size(SNAME("font_size")) * 2, p_char))) {
		selected_glyphs.erase(font_main->get_data(0)->get_glyph_index(get_theme_font_size(SNAME("font_size")) * 2, p_char));
		return false;
	} else {
		selected_chars.insert(p_char);
		return true;
	}
}

void DynamicFontImportSettings::_range_update(int32_t p_start, int32_t p_end) {
	bool all_selected = true;
	for (int32_t i = p_start; i <= p_end; i++) {
		if (font_main->has_char(i)) {
			if (font_main->get_data(0).is_valid()) {
				all_selected = all_selected && (selected_chars.has(i) || (font_main->get_data(0).is_valid() && selected_glyphs.has(font_main->get_data(0)->get_glyph_index(get_theme_font_size(SNAME("font_size")) * 2, i))));
			} else {
				all_selected = all_selected && selected_chars.has(i);
			}
		}
	}
	for (int32_t i = p_start; i <= p_end; i++) {
		if (font_main->has_char(i)) {
			if (!all_selected) {
				selected_chars.insert(i);
			} else {
				selected_chars.erase(i);
				if (font_main->get_data(0).is_valid()) {
					selected_glyphs.erase(font_main->get_data(0)->get_glyph_index(get_theme_font_size(SNAME("font_size")) * 2, i));
				}
			}
		}
	}
	_edit_range(p_start, p_end);

	TreeItem *item = glyph_tree->get_selected();
	ERR_FAIL_NULL(item);
	item->set_checked(0, !all_selected);
}

/*************************************************************************/
/* Page 5 callbacks: CMetadata override                                  */
/*************************************************************************/

void DynamicFontImportSettings::_lang_add() {
	locale_select->popup_locale_dialog();
}

void DynamicFontImportSettings::_lang_add_item(const String &p_locale) {
	TreeItem *lang_item = lang_list->create_item(lang_list_root);
	ERR_FAIL_NULL(lang_item);

	lang_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	lang_item->set_editable(0, true);
	lang_item->set_checked(0, false);
	lang_item->set_text(1, p_locale);
	lang_item->set_editable(1, true);
	lang_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
	lang_item->set_button_color(2, 0, Color(1, 1, 1, 0.75));
}

void DynamicFontImportSettings::_lang_remove(Object *p_item, int p_column, int p_id) {
	TreeItem *lang_item = (TreeItem *)p_item;
	ERR_FAIL_NULL(lang_item);

	lang_list_root->remove_child(lang_item);
	memdelete(lang_item);
}

void DynamicFontImportSettings::_ot_add() {
	menu_ot->set_position(ot_list->get_screen_transform().xform(ot_list->get_local_mouse_position()));
	menu_ot->set_size(Vector2(1, 1));
	menu_ot->popup();
}

void DynamicFontImportSettings::_ot_add_item(int p_option) {
	String name = TS->tag_to_name(p_option);
	for (TreeItem *ot_item = ot_list_root->get_first_child(); ot_item; ot_item = ot_item->get_next()) {
		if (ot_item->get_text(0) == name) {
			return;
		}
	}
	TreeItem *ot_item = ot_list->create_item(ot_list_root);
	ERR_FAIL_NULL(ot_item);

	ot_item->set_text(0, name);
	ot_item->set_editable(0, false);
	ot_item->set_text(1, "1");
	ot_item->set_editable(1, true);
	ot_item->add_button(2, ot_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
	ot_item->set_button_color(2, 0, Color(1, 1, 1, 0.75));
}

void DynamicFontImportSettings::_ot_remove(Object *p_item, int p_column, int p_id) {
	TreeItem *ot_item = (TreeItem *)p_item;
	ERR_FAIL_NULL(ot_item);

	ot_list_root->remove_child(ot_item);
	memdelete(ot_item);
}

void DynamicFontImportSettings::_script_add() {
	menu_scripts->set_position(script_list->get_screen_position() + script_list->get_local_mouse_position());
	menu_scripts->reset_size();
	menu_scripts->popup();
}

void DynamicFontImportSettings::_script_add_item(int p_option) {
	TreeItem *script_item = script_list->create_item(script_list_root);
	ERR_FAIL_NULL(script_item);

	script_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	script_item->set_editable(0, true);
	script_item->set_checked(0, false);
	script_item->set_text(1, script_codes[p_option]);
	script_item->set_editable(1, true);
	script_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
	script_item->set_button_color(2, 0, Color(1, 1, 1, 0.75));
}

void DynamicFontImportSettings::_script_remove(Object *p_item, int p_column, int p_id) {
	TreeItem *script_item = (TreeItem *)p_item;
	ERR_FAIL_NULL(script_item);

	script_list_root->remove_child(script_item);
	memdelete(script_item);
}

/*************************************************************************/
/* Common                                                                */
/*************************************************************************/

DynamicFontImportSettings *DynamicFontImportSettings::singleton = nullptr;

String DynamicFontImportSettings::_pad_zeros(const String &p_hex) const {
	int len = CLAMP(5 - p_hex.length(), 0, 5);
	return String("0").repeat(len) + p_hex;
}

void DynamicFontImportSettings::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("confirmed", callable_mp(this, &DynamicFontImportSettings::_re_import));
		} break;

		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			add_lang->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			add_script->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			add_var->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			add_ot->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
		} break;
	}
}

void DynamicFontImportSettings::_re_import() {
	Map<StringName, Variant> main_settings;

	main_settings["antialiased"] = import_settings_data->get("antialiased");
	main_settings["generate_mipmaps"] = import_settings_data->get("generate_mipmaps");
	main_settings["multichannel_signed_distance_field"] = import_settings_data->get("multichannel_signed_distance_field");
	main_settings["msdf_pixel_range"] = import_settings_data->get("msdf_pixel_range");
	main_settings["msdf_size"] = import_settings_data->get("msdf_size");
	main_settings["force_autohinter"] = import_settings_data->get("force_autohinter");
	main_settings["hinting"] = import_settings_data->get("hinting");
	main_settings["subpixel_positioning"] = import_settings_data->get("subpixel_positioning");
	main_settings["embolden"] = import_settings_data->get("embolden");
	main_settings["transform"] = import_settings_data->get("transform");
	main_settings["oversampling"] = import_settings_data->get("oversampling");
	main_settings["compress"] = import_settings_data->get("compress");

	Vector<String> variations;
	for (TreeItem *vars_item = vars_list_root->get_first_child(); vars_item; vars_item = vars_item->get_next()) {
		String variation;
		Ref<DynamicFontImportSettingsData> import_variation_data = vars_item->get_metadata(0);
		ERR_FAIL_NULL(import_variation_data);

		String name = vars_item->get_text(0);
		variation += ("name=" + name);
		for (Map<StringName, Variant>::Element *E = import_variation_data->settings.front(); E; E = E->next()) {
			if (!variation.is_empty()) {
				variation += ",";
			}
			variation += (String(E->key()) + "=" + String(E->get()));
		}
		variations.push_back(variation);
	}
	main_settings["preload/configurations"] = variations;

	Vector<String> langs_enabled;
	Vector<String> langs_disabled;
	for (TreeItem *lang_item = lang_list_root->get_first_child(); lang_item; lang_item = lang_item->get_next()) {
		bool selected = lang_item->is_checked(0);
		String name = lang_item->get_text(1);
		if (selected) {
			langs_enabled.push_back(name);
		} else {
			langs_disabled.push_back(name);
		}
	}
	main_settings["support_overrides/language_enabled"] = langs_enabled;
	main_settings["support_overrides/language_disabled"] = langs_disabled;

	Vector<String> scripts_enabled;
	Vector<String> scripts_disabled;
	for (TreeItem *script_item = script_list_root->get_first_child(); script_item; script_item = script_item->get_next()) {
		bool selected = script_item->is_checked(0);
		String name = script_item->get_text(1);
		if (selected) {
			scripts_enabled.push_back(name);
		} else {
			scripts_disabled.push_back(name);
		}
	}
	main_settings["support_overrides/script_enabled"] = scripts_enabled;
	main_settings["support_overrides/script_disabled"] = scripts_disabled;

	if (!selected_chars.is_empty()) {
		Vector<String> ranges;
		char32_t start = selected_chars.front()->get();
		for (Set<char32_t>::Element *E = selected_chars.front()->next(); E; E = E->next()) {
			if (E->prev() && ((E->prev()->get() + 1) != E->get())) {
				ranges.push_back(String("0x") + String::num_int64(start, 16) + String("-0x") + String::num_int64(E->prev()->get(), 16));
				start = E->get();
			}
		}
		ranges.push_back(String("0x") + String::num_int64(start, 16) + String("-0x") + String::num_int64(selected_chars.back()->get(), 16));
		main_settings["preload/char_ranges"] = ranges;
	}

	if (!selected_glyphs.is_empty()) {
		Vector<String> ranges;
		int32_t start = selected_glyphs.front()->get();
		for (Set<int32_t>::Element *E = selected_glyphs.front()->next(); E; E = E->next()) {
			if (E->prev() && ((E->prev()->get() + 1) != E->get())) {
				ranges.push_back(String("0x") + String::num_int64(start, 16) + String("-0x") + String::num_int64(E->prev()->get(), 16));
				start = E->get();
			}
		}
		ranges.push_back(String("0x") + String::num_int64(start, 16) + String("-0x") + String::num_int64(selected_glyphs.back()->get(), 16));
		main_settings["preload/glyph_ranges"] = ranges;
	}

	Dictionary ot_ov;
	for (TreeItem *ot_item = ot_list_root->get_first_child(); ot_item; ot_item = ot_item->get_next()) {
		String tag = ot_item->get_text(0);
		int32_t value = ot_item->get_text(1).to_int();
		ot_ov[tag] = value;
	}
	main_settings["opentype_feature_overrides"] = ot_ov;

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("Import settings:");
		for (Map<StringName, Variant>::Element *E = main_settings.front(); E; E = E->next()) {
			print_line(String("    ") + String(E->key()).utf8().get_data() + " == " + String(E->get()).utf8().get_data());
		}
	}

	EditorFileSystem::get_singleton()->reimport_file_with_custom_parameters(base_path, "font_data_dynamic", main_settings);
}

void DynamicFontImportSettings::open_settings(const String &p_path) {
	// Load base font data.
	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);

	// Load font for preview.
	Ref<FontData> dfont_prev;
	dfont_prev.instantiate();
	dfont_prev->set_data(data);

	font_preview.instantiate();
	font_preview->add_data(dfont_prev);

	String sample;
	static const String sample_base = U"12漢字ԱբΑαАбΑαאבابܐܒހށआআਆઆଆஆఆಆആආกิກິༀကႠა한글ሀᎣᐁᚁᚠᜀᜠᝀᝠកᠠᤁᥐAb😀";
	for (int i = 0; i < sample_base.length(); i++) {
		if (dfont_prev->has_char(sample_base[i])) {
			sample += sample_base[i];
		}
	}
	if (sample.is_empty()) {
		sample = dfont_prev->get_supported_chars().substr(0, 6);
	}
	font_preview_label->set_text(sample);

	// Load second copy of font with MSDF disabled for the glyph table and metadata extraction.
	Ref<FontData> dfont_main;
	dfont_main.instantiate();
	dfont_main->set_data(data);
	dfont_main->set_multichannel_signed_distance_field(false);

	font_main.instantiate();
	font_main->add_data(dfont_main);
	text_edit->add_theme_font_override("font", font_main);

	base_path = p_path;

	inspector_vars->edit(nullptr);
	inspector_general->edit(nullptr);

	int gww = get_theme_font(SNAME("font"))->get_string_size("00000", get_theme_font_size(SNAME("font_size"))).x + 50;
	glyph_table->set_column_custom_minimum_width(0, gww);

	glyph_table->clear();
	vars_list->clear();
	lang_list->clear();
	script_list->clear();
	ot_list->clear();

	selected_chars.clear();
	selected_glyphs.clear();
	text_edit->set_text(String());

	vars_list_root = vars_list->create_item();
	lang_list_root = lang_list->create_item();
	script_list_root = script_list->create_item();
	ot_list_root = ot_list->create_item();

	options_variations.clear();
	Dictionary var_list = dfont_main->get_supported_variation_list();
	for (int i = 0; i < var_list.size(); i++) {
		int32_t tag = var_list.get_key_at_index(i);
		Vector3i value = var_list.get_value_at_index(i);
		options_variations.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, TS->tag_to_name(tag), PROPERTY_HINT_RANGE, itos(value.x) + "," + itos(value.y) + ",1"), value.z));
	}
	options_variations.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "size", PROPERTY_HINT_RANGE, "0,127,1"), 16));
	options_variations.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "outline_size", PROPERTY_HINT_RANGE, "0,127,1"), 0));
	options_variations.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "extra_spacing_glyph"), 0));
	options_variations.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "extra_spacing_space"), 0));

	import_settings_data->defaults.clear();
	for (List<ResourceImporter::ImportOption>::Element *E = options_general.front(); E; E = E->next()) {
		import_settings_data->defaults[E->get().option.name] = E->get().default_value;
	}

	Ref<ConfigFile> config;
	config.instantiate();
	ERR_FAIL_NULL(config);

	Error err = config->load(p_path + ".import");
	print_verbose("Loading import settings:");
	if (err == OK) {
		List<String> keys;
		config->get_section_keys("params", &keys);
		for (List<String>::Element *E = keys.front(); E; E = E->next()) {
			String key = E->get();
			print_verbose(String("    ") + key + " == " + String(config->get_value("params", key)));
			if (key == "preload/char_ranges") {
				Vector<String> ranges = config->get_value("params", key);
				for (int i = 0; i < ranges.size(); i++) {
					int32_t start, end;
					Vector<String> tokens = ranges[i].split("-");
					if (tokens.size() == 2) {
						if (!ResourceImporterDynamicFont::_decode_range(tokens[0], start) || !ResourceImporterDynamicFont::_decode_range(tokens[1], end)) {
							WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
							continue;
						}
					} else if (tokens.size() == 1) {
						if (!ResourceImporterDynamicFont::_decode_range(tokens[0], start)) {
							WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
							continue;
						}
						end = start;
					} else {
						WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
						continue;
					}
					for (int32_t j = start; j <= end; j++) {
						selected_chars.insert(j);
					}
				}
			} else if (key == "preload/glyph_ranges") {
				Vector<String> ranges = config->get_value("params", key);
				for (int i = 0; i < ranges.size(); i++) {
					int32_t start, end;
					Vector<String> tokens = ranges[i].split("-");
					if (tokens.size() == 2) {
						if (!ResourceImporterDynamicFont::_decode_range(tokens[0], start) || !ResourceImporterDynamicFont::_decode_range(tokens[1], end)) {
							WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
							continue;
						}
					} else if (tokens.size() == 1) {
						if (!ResourceImporterDynamicFont::_decode_range(tokens[0], start)) {
							WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
							continue;
						}
						end = start;
					} else {
						WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
						continue;
					}
					for (int32_t j = start; j <= end; j++) {
						selected_glyphs.insert(j);
					}
				}
			} else if (key == "preload/configurations") {
				Vector<String> variations = config->get_value("params", key);
				for (int i = 0; i < variations.size(); i++) {
					TreeItem *vars_item = vars_list->create_item(vars_list_root);
					ERR_FAIL_NULL(vars_item);

					vars_item->set_text(0, TTR("Configuration") + " " + itos(i));
					vars_item->set_editable(0, true);
					vars_item->add_button(1, vars_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove Variation"));
					vars_item->set_button_color(1, 0, Color(1, 1, 1, 0.75));

					Ref<DynamicFontImportSettingsData> import_variation_data_custom;
					import_variation_data_custom.instantiate();
					import_variation_data_custom->owner = this;
					ERR_FAIL_NULL(import_variation_data_custom);

					for (List<ResourceImporter::ImportOption>::Element *F = options_variations.front(); F; F = F->next()) {
						import_variation_data_custom->defaults[F->get().option.name] = F->get().default_value;
					}

					import_variation_data_custom->options = options_variations;

					vars_item->set_metadata(0, import_variation_data_custom);
					Vector<String> variation_tags = variations[i].split(",");
					for (int j = 0; j < variation_tags.size(); j++) {
						Vector<String> tokens = variation_tags[j].split("=");
						if (tokens[0] == "name") {
							vars_item->set_text(0, tokens[1]);
						} else if (tokens[0] == "size" || tokens[0] == "outline_size" || tokens[0] == "extra_spacing_space" || tokens[0] == "extra_spacing_glyph") {
							import_variation_data_custom->set(tokens[0], tokens[1].to_int());
						} else {
							import_variation_data_custom->set(tokens[0], tokens[1].to_float());
						}
					}
				}
			} else if (key == "support_overrides/language_enabled") {
				PackedStringArray _langs = config->get_value("params", key);
				for (int i = 0; i < _langs.size(); i++) {
					TreeItem *lang_item = lang_list->create_item(lang_list_root);
					ERR_FAIL_NULL(lang_item);

					lang_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
					lang_item->set_editable(0, true);
					lang_item->set_checked(0, true);
					lang_item->set_text(1, _langs[i]);
					lang_item->set_editable(1, true);
					lang_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
				}
			} else if (key == "support_overrides/language_disabled") {
				PackedStringArray _langs = config->get_value("params", key);
				for (int i = 0; i < _langs.size(); i++) {
					TreeItem *lang_item = lang_list->create_item(lang_list_root);
					ERR_FAIL_NULL(lang_item);

					lang_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
					lang_item->set_editable(0, true);
					lang_item->set_checked(0, false);
					lang_item->set_text(1, _langs[i]);
					lang_item->set_editable(1, true);
					lang_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
				}
			} else if (key == "support_overrides/script_enabled") {
				PackedStringArray _scripts = config->get_value("params", key);
				for (int i = 0; i < _scripts.size(); i++) {
					TreeItem *script_item = script_list->create_item(script_list_root);
					ERR_FAIL_NULL(script_item);

					script_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
					script_item->set_editable(0, true);
					script_item->set_checked(0, true);
					script_item->set_text(1, _scripts[i]);
					script_item->set_editable(1, true);
					script_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
				}
			} else if (key == "support_overrides/script_disabled") {
				PackedStringArray _scripts = config->get_value("params", key);
				for (int i = 0; i < _scripts.size(); i++) {
					TreeItem *script_item = script_list->create_item(script_list_root);
					ERR_FAIL_NULL(script_item);

					script_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
					script_item->set_editable(0, true);
					script_item->set_checked(0, false);
					script_item->set_text(1, _scripts[i]);
					script_item->set_editable(1, true);
					script_item->add_button(2, lang_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
				}
			} else if (key == "opentype_feature_overrides") {
				Dictionary features = config->get_value("params", key);
				for (const Variant *ftr = features.next(nullptr); ftr != nullptr; ftr = features.next(ftr)) {
					TreeItem *ot_item = ot_list->create_item(ot_list_root);
					ERR_FAIL_NULL(ot_item);
					int32_t value = features[*ftr];
					if (ftr->get_type() == Variant::STRING) {
						ot_item->set_text(0, *ftr);
					} else {
						ot_item->set_text(0, TS->tag_to_name(*ftr));
					}
					ot_item->set_editable(0, false);
					ot_item->set_text(1, itos(value));
					ot_item->set_editable(1, true);
					ot_item->add_button(2, ot_list->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_REMOVE_VAR, false, TTR("Remove"));
					ot_item->set_button_color(2, 0, Color(1, 1, 1, 0.75));
				}
			} else {
				Variant value = config->get_value("params", key);
				import_settings_data->defaults[key] = value;
			}
		}
	}
	label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(selected_glyphs.size()));

	import_settings_data->options = options_general;
	inspector_general->edit(import_settings_data.ptr());
	import_settings_data->notify_property_list_changed();

	if (font_preview->get_data_count() > 0) {
		font_preview->get_data(0)->set_antialiased(import_settings_data->get("antialiased"));
		font_preview->get_data(0)->set_multichannel_signed_distance_field(import_settings_data->get("multichannel_signed_distance_field"));
		font_preview->get_data(0)->set_msdf_pixel_range(import_settings_data->get("msdf_pixel_range"));
		font_preview->get_data(0)->set_msdf_size(import_settings_data->get("msdf_size"));
		font_preview->get_data(0)->set_force_autohinter(import_settings_data->get("force_autohinter"));
		font_preview->get_data(0)->set_hinting((TextServer::Hinting)import_settings_data->get("hinting").operator int());
		font_preview->get_data(0)->set_subpixel_positioning((TextServer::SubpixelPositioning)import_settings_data->get("subpixel_positioning").operator int());
		font_preview->get_data(0)->set_embolden(import_settings_data->get("embolden"));
		font_preview->get_data(0)->set_transform(import_settings_data->get("transform"));
		font_preview->get_data(0)->set_oversampling(import_settings_data->get("oversampling"));
	}
	font_preview_label->add_theme_font_override("font", font_preview);
	font_preview_label->update();

	menu_ot->clear();
	menu_ot_ss->clear();
	menu_ot_cv->clear();
	menu_ot_cu->clear();
	bool have_ss = false;
	bool have_cv = false;
	bool have_cu = false;
	Dictionary features = font_preview->get_feature_list();
	for (const Variant *ftr = features.next(nullptr); ftr != nullptr; ftr = features.next(ftr)) {
		String ftr_name = TS->tag_to_name(*ftr);
		if (ftr_name.begins_with("stylistic_set_")) {
			menu_ot_ss->add_item(ftr_name.capitalize(), (int32_t)*ftr);
			have_ss = true;
		} else if (ftr_name.begins_with("character_variant_")) {
			menu_ot_cv->add_item(ftr_name.capitalize(), (int32_t)*ftr);
			have_cv = true;
		} else if (ftr_name.begins_with("custom_")) {
			menu_ot_cu->add_item(ftr_name.replace("custom_", ""), (int32_t)*ftr);
			have_cu = true;
		} else {
			menu_ot->add_item(ftr_name.capitalize(), (int32_t)*ftr);
		}
	}
	if (have_ss) {
		menu_ot->add_submenu_item(RTR("Stylistic Sets"), "SSMenu");
	}
	if (have_cv) {
		menu_ot->add_submenu_item(RTR("Character Variants"), "CVMenu");
	}
	if (have_cu) {
		menu_ot->add_submenu_item(RTR("Custom"), "CUMenu");
	}

	_variations_validate();

	popup_centered_ratio();

	set_title(vformat(TTR("Advanced Import Settings for '%s'"), base_path.get_file()));
}

DynamicFontImportSettings *DynamicFontImportSettings::get_singleton() {
	return singleton;
}

DynamicFontImportSettings::DynamicFontImportSettings() {
	singleton = this;

	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "antialiased"), true));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "generate_mipmaps"), false));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "multichannel_signed_distance_field", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), true));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "msdf_pixel_range", PROPERTY_HINT_RANGE, "1,100,1"), 8));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "msdf_size", PROPERTY_HINT_RANGE, "1,250,1"), 48));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "force_autohinter"), false));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal"), 1));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "subpixel_positioning", PROPERTY_HINT_ENUM, "Disabled,Auto,One half of a pixel,One quarter of a pixel"), 1));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "embolden", PROPERTY_HINT_RANGE, "-2,2,0.01"), 0.f));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::TRANSFORM2D, "transform"), Transform2D()));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "oversampling", PROPERTY_HINT_RANGE, "0,10,0.1"), 0.0));
	options_general.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "compress", PROPERTY_HINT_NONE, ""), false));

	// Popup menus

	locale_select = memnew(EditorLocaleDialog);
	locale_select->connect("locale_selected", callable_mp(this, &DynamicFontImportSettings::_lang_add_item));
	add_child(locale_select);

	menu_scripts = memnew(PopupMenu);
	menu_scripts->set_name("Script");
	script_codes = TranslationServer::get_singleton()->get_all_scripts();
	for (int i = 0; i < script_codes.size(); i++) {
		menu_scripts->add_item(TranslationServer::get_singleton()->get_script_name(script_codes[i]) + " (" + script_codes[i] + ")", i);
	}
	add_child(menu_scripts);
	menu_scripts->connect("id_pressed", callable_mp(this, &DynamicFontImportSettings::_script_add_item));

	menu_ot = memnew(PopupMenu);
	add_child(menu_ot);
	menu_ot->connect("id_pressed", callable_mp(this, &DynamicFontImportSettings::_ot_add_item));

	menu_ot_cv = memnew(PopupMenu);
	menu_ot_cv->set_name("CVMenu");
	menu_ot->add_child(menu_ot_cv);
	menu_ot_cv->connect("id_pressed", callable_mp(this, &DynamicFontImportSettings::_ot_add_item));

	menu_ot_ss = memnew(PopupMenu);
	menu_ot_ss->set_name("SSMenu");
	menu_ot->add_child(menu_ot_ss);
	menu_ot_ss->connect("id_pressed", callable_mp(this, &DynamicFontImportSettings::_ot_add_item));

	menu_ot_cu = memnew(PopupMenu);
	menu_ot_cu->set_name("CUMenu");
	menu_ot->add_child(menu_ot_cu);
	menu_ot_cu->connect("id_pressed", callable_mp(this, &DynamicFontImportSettings::_ot_add_item));

	Color warn_color = (EditorNode::get_singleton()) ? EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("warning_color"), SNAME("Editor")) : Color(1, 1, 0);

	// Root layout

	VBoxContainer *root_vb = memnew(VBoxContainer);
	add_child(root_vb);

	main_pages = memnew(TabContainer);
	main_pages->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
	main_pages->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_pages->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	root_vb->add_child(main_pages);

	label_warn = memnew(Label);
	label_warn->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label_warn->set_text("");
	root_vb->add_child(label_warn);
	label_warn->add_theme_color_override("font_color", warn_color);
	label_warn->hide();

	// Page 1 layout: Rendering Options

	VBoxContainer *page1_vb = memnew(VBoxContainer);
	page1_vb->set_name(TTR("Rendering Options"));
	main_pages->add_child(page1_vb);

	page1_description = memnew(Label);
	page1_description->set_text(TTR("Select font rendering options:"));
	page1_description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page1_vb->add_child(page1_description);

	HSplitContainer *page1_hb = memnew(HSplitContainer);
	page1_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	page1_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page1_vb->add_child(page1_hb);

	font_preview_label = memnew(Label);
	font_preview_label->add_theme_font_size_override("font_size", 200 * EDSCALE);
	font_preview_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	font_preview_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	font_preview_label->set_autowrap_mode(Label::AUTOWRAP_ARBITRARY);
	font_preview_label->set_clip_text(true);
	font_preview_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	font_preview_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page1_hb->add_child(font_preview_label);

	inspector_general = memnew(EditorInspector);
	inspector_general->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector_general->set_custom_minimum_size(Size2(300 * EDSCALE, 250 * EDSCALE));
	inspector_general->connect("property_edited", callable_mp(this, &DynamicFontImportSettings::_main_prop_changed));
	page1_hb->add_child(inspector_general);

	// Page 2 layout: Configurations
	VBoxContainer *page2_vb = memnew(VBoxContainer);
	page2_vb->set_name(TTR("Sizes and Variations"));
	main_pages->add_child(page2_vb);

	page2_description = memnew(Label);
	page2_description->set_text(TTR("Add font size, variation coordinates, and extra spacing combinations to pre-render:"));
	page2_description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page2_description->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	page2_vb->add_child(page2_description);

	HSplitContainer *page2_hb = memnew(HSplitContainer);
	page2_hb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	page2_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page2_vb->add_child(page2_hb);

	VBoxContainer *page2_side_vb = memnew(VBoxContainer);
	page2_hb->add_child(page2_side_vb);

	HBoxContainer *page2_hb_vars = memnew(HBoxContainer);
	page2_side_vb->add_child(page2_hb_vars);

	label_vars = memnew(Label);
	page2_hb_vars->add_child(label_vars);
	label_vars->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label_vars->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	label_vars->set_text(TTR("Configuration:"));

	add_var = memnew(Button);
	page2_hb_vars->add_child(add_var);
	add_var->set_tooltip(TTR("Add configuration"));
	add_var->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	add_var->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_variation_add));

	vars_list = memnew(Tree);
	page2_side_vb->add_child(vars_list);
	vars_list->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	vars_list->set_hide_root(true);
	vars_list->set_columns(2);
	vars_list->set_column_expand(0, true);
	vars_list->set_column_custom_minimum_width(0, 80 * EDSCALE);
	vars_list->set_column_expand(1, false);
	vars_list->set_column_custom_minimum_width(1, 50 * EDSCALE);
	vars_list->connect("item_selected", callable_mp(this, &DynamicFontImportSettings::_variation_selected));
	vars_list->connect("button_pressed", callable_mp(this, &DynamicFontImportSettings::_variation_remove));
	vars_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	inspector_vars = memnew(EditorInspector);
	inspector_vars->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector_vars->connect("property_edited", callable_mp(this, &DynamicFontImportSettings::_variation_changed));
	page2_hb->add_child(inspector_vars);

	// Page 3 layout: Text to select glyphs
	VBoxContainer *page3_vb = memnew(VBoxContainer);
	page3_vb->set_name(TTR("Glyphs from the Text"));
	main_pages->add_child(page3_vb);

	page3_description = memnew(Label);
	page3_description->set_text(TTR("Enter a text to shape and add all required glyphs to pre-render list:"));
	page3_description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page3_description->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	page3_vb->add_child(page3_description);

	HBoxContainer *ot_hb = memnew(HBoxContainer);
	page3_vb->add_child(ot_hb);
	ot_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *label_ed_ftr = memnew(Label);
	ot_hb->add_child(label_ed_ftr);
	label_ed_ftr->set_text(TTR("OpenType features:"));

	ftr_edit = memnew(LineEdit);
	ot_hb->add_child(ftr_edit);
	ftr_edit->connect("text_changed", callable_mp(this, &DynamicFontImportSettings::_change_text_opts));
	ftr_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	Label *label_ed_lang = memnew(Label);
	ot_hb->add_child(label_ed_lang);
	label_ed_lang->set_text(TTR("Text language:"));

	lang_edit = memnew(LineEdit);
	ot_hb->add_child(lang_edit);
	lang_edit->connect("text_changed", callable_mp(this, &DynamicFontImportSettings::_change_text_opts));
	lang_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	text_edit = memnew(TextEdit);
	page3_vb->add_child(text_edit);
	text_edit->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	text_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	HBoxContainer *text_hb = memnew(HBoxContainer);
	page3_vb->add_child(text_hb);
	text_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	label_glyphs = memnew(Label);
	text_hb->add_child(label_glyphs);
	label_glyphs->set_text(TTR("Preloaded glyphs: ") + itos(0));
	label_glyphs->set_custom_minimum_size(Size2(50 * EDSCALE, 0));

	Button *btn_fill = memnew(Button);
	text_hb->add_child(btn_fill);
	btn_fill->set_text(TTR("Shape text and add glyphs"));
	btn_fill->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_glyph_text_selected));

	Button *btn_clear = memnew(Button);
	text_hb->add_child(btn_clear);
	btn_clear->set_text(TTR("Clear glyph list"));
	btn_clear->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_glyph_clear));

	// Page 4 layout: Character map
	VBoxContainer *page4_vb = memnew(VBoxContainer);
	page4_vb->set_name(TTR("Glyphs from the Character Map"));
	main_pages->add_child(page4_vb);

	page4_description = memnew(Label);
	page4_description->set_text(TTR("Add or remove additional glyphs from the character map to pre-render list:\nNote: Some stylistic alternatives and glyph variants do not have one-to-one correspondence to character, and not shown in this map, use \"Glyphs from the text\" to add these."));
	page4_description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page4_description->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	page4_vb->add_child(page4_description);

	HSplitContainer *glyphs_split = memnew(HSplitContainer);
	glyphs_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	glyphs_split->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page4_vb->add_child(glyphs_split);

	glyph_table = memnew(Tree);
	glyphs_split->add_child(glyph_table);
	glyph_table->set_custom_minimum_size(Size2((30 * 16 + 100) * EDSCALE, 0));
	glyph_table->set_columns(17);
	glyph_table->set_column_expand(0, false);
	glyph_table->set_hide_root(true);
	glyph_table->set_allow_reselect(true);
	glyph_table->set_select_mode(Tree::SELECT_SINGLE);
	glyph_table->connect("item_activated", callable_mp(this, &DynamicFontImportSettings::_glyph_selected));
	glyph_table->set_column_titles_visible(true);
	for (int i = 0; i < 16; i++) {
		glyph_table->set_column_title(i + 1, String::num_int64(i, 16));
	}
	glyph_table->add_theme_style_override("selected", glyph_table->get_theme_stylebox(SNAME("bg")));
	glyph_table->add_theme_style_override("selected_focus", glyph_table->get_theme_stylebox(SNAME("bg")));
	glyph_table->add_theme_constant_override("h_separation", 0);
	glyph_table->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	glyph_table->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	glyph_tree = memnew(Tree);
	glyphs_split->add_child(glyph_tree);
	glyph_tree->set_custom_minimum_size(Size2(300 * EDSCALE, 0));
	glyph_tree->set_columns(2);
	glyph_tree->set_hide_root(true);
	glyph_tree->set_column_expand(0, false);
	glyph_tree->set_column_expand(1, true);
	glyph_tree->set_column_custom_minimum_width(0, 120 * EDSCALE);
	glyph_tree->connect("item_activated", callable_mp(this, &DynamicFontImportSettings::_range_edited));
	glyph_tree->connect("item_selected", callable_mp(this, &DynamicFontImportSettings::_range_selected));
	glyph_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	glyph_root = glyph_tree->create_item();
	for (int i = 0; !unicode_ranges[i].name.is_empty(); i++) {
		_add_glyph_range_item(unicode_ranges[i].start, unicode_ranges[i].end, unicode_ranges[i].name);
	}

	// Page 4 layout: Metadata override
	VBoxContainer *page5_vb = memnew(VBoxContainer);
	page5_vb->set_name(TTR("Metadata Override"));
	main_pages->add_child(page5_vb);

	page5_description = memnew(Label);
	page5_description->set_text(TTR("Add or remove language and script support overrides, to control fallback font selection order:"));
	page5_description->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	page5_description->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	page5_vb->add_child(page5_description);

	HBoxContainer *hb_lang = memnew(HBoxContainer);
	page5_vb->add_child(hb_lang);

	label_langs = memnew(Label);
	label_langs->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label_langs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	label_langs->set_text(TTR("Language support overrides"));
	hb_lang->add_child(label_langs);

	add_lang = memnew(Button);
	hb_lang->add_child(add_lang);
	add_lang->set_tooltip(TTR("Add language override"));
	add_lang->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	add_lang->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_lang_add));

	lang_list = memnew(Tree);
	page5_vb->add_child(lang_list);
	lang_list->set_hide_root(true);
	lang_list->set_columns(3);
	lang_list->set_column_expand(0, false); // Check
	lang_list->set_column_custom_minimum_width(0, 50 * EDSCALE);
	lang_list->set_column_expand(1, true);
	lang_list->set_column_custom_minimum_width(1, 80 * EDSCALE);
	lang_list->set_column_expand(2, false);
	lang_list->set_column_custom_minimum_width(2, 50 * EDSCALE);
	lang_list->connect("button_pressed", callable_mp(this, &DynamicFontImportSettings::_lang_remove));
	lang_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	HBoxContainer *hb_script = memnew(HBoxContainer);
	page5_vb->add_child(hb_script);

	label_script = memnew(Label);
	label_script->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label_script->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	label_script->set_text(TTR("Script support overrides"));
	hb_script->add_child(label_script);

	add_script = memnew(Button);
	hb_script->add_child(add_script);
	add_script->set_tooltip(TTR("Add script override"));
	add_script->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	add_script->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_script_add));

	script_list = memnew(Tree);
	page5_vb->add_child(script_list);
	script_list->set_hide_root(true);
	script_list->set_columns(3);
	script_list->set_column_expand(0, false);
	script_list->set_column_custom_minimum_width(0, 50 * EDSCALE);
	script_list->set_column_expand(1, true);
	script_list->set_column_custom_minimum_width(1, 80 * EDSCALE);
	script_list->set_column_expand(2, false);
	script_list->set_column_custom_minimum_width(2, 50 * EDSCALE);
	script_list->connect("button_pressed", callable_mp(this, &DynamicFontImportSettings::_script_remove));
	script_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	HBoxContainer *hb_ot = memnew(HBoxContainer);
	page5_vb->add_child(hb_ot);

	label_ot = memnew(Label);
	label_ot->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label_ot->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	label_ot->set_text(TTR("OpenType feature overrides"));
	hb_ot->add_child(label_ot);

	add_ot = memnew(Button);
	hb_ot->add_child(add_ot);
	add_ot->set_tooltip(TTR("Add feature override"));
	add_ot->set_icon(add_var->get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
	add_ot->connect("pressed", callable_mp(this, &DynamicFontImportSettings::_ot_add));

	ot_list = memnew(Tree);
	page5_vb->add_child(ot_list);
	ot_list->set_hide_root(true);
	ot_list->set_columns(3);
	ot_list->set_column_expand(0, true);
	ot_list->set_column_custom_minimum_width(0, 80 * EDSCALE);
	ot_list->set_column_expand(1, true);
	ot_list->set_column_custom_minimum_width(1, 80 * EDSCALE);
	ot_list->set_column_expand(2, false);
	ot_list->set_column_custom_minimum_width(2, 50 * EDSCALE);
	ot_list->connect("button_pressed", callable_mp(this, &DynamicFontImportSettings::_ot_remove));
	ot_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	// Common

	import_settings_data.instantiate();
	import_settings_data->owner = this;

	get_ok_button()->set_text(TTR("Reimport"));
	get_cancel_button()->set_text(TTR("Close"));
}
