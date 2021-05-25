/*************************************************************************/
/*  text_server_adv.cpp                                                  */
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

#include "text_server_adv.h"
#include "bitmap_font_adv.h"
#include "dynamic_font_adv.h"

#include "core/string/translation.h"

#ifdef ICU_STATIC_DATA
#include "thirdparty/icu4c/icudata.gen.h"
#endif

_FORCE_INLINE_ bool is_ain(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AIN;
}

_FORCE_INLINE_ bool is_alef(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_ALEF;
}

_FORCE_INLINE_ bool is_beh(char32_t p_chr) {
	int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
	return (prop == U_JG_BEH) || (prop == U_JG_NOON) || (prop == U_JG_AFRICAN_NOON) || (prop == U_JG_NYA) || (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH);
}

_FORCE_INLINE_ bool is_dal(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_DAL;
}

_FORCE_INLINE_ bool is_feh(char32_t p_chr) {
	return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_FEH) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_FEH);
}

_FORCE_INLINE_ bool is_gaf(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_GAF;
}

_FORCE_INLINE_ bool is_heh(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_HEH;
}

_FORCE_INLINE_ bool is_kaf(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_KAF;
}

_FORCE_INLINE_ bool is_lam(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_LAM;
}

_FORCE_INLINE_ bool is_qaf(char32_t p_chr) {
	return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_QAF) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_AFRICAN_QAF);
}

_FORCE_INLINE_ bool is_reh(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_REH;
}

_FORCE_INLINE_ bool is_seen_sad(char32_t p_chr) {
	return (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SAD) || (u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_SEEN);
}

_FORCE_INLINE_ bool is_tah(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TAH;
}

_FORCE_INLINE_ bool is_teh_marbuta(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_TEH_MARBUTA;
}

_FORCE_INLINE_ bool is_yeh(char32_t p_chr) {
	int32_t prop = u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP);
	return (prop == U_JG_YEH) || (prop == U_JG_FARSI_YEH) || (prop == U_JG_YEH_BARREE) || (prop == U_JG_BURUSHASKI_YEH_BARREE) || (prop == U_JG_YEH_WITH_TAIL);
}

_FORCE_INLINE_ bool is_waw(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_GROUP) == U_JG_WAW;
}

_FORCE_INLINE_ bool is_transparent(char32_t p_chr) {
	return u_getIntPropertyValue(p_chr, UCHAR_JOINING_TYPE) == U_JT_TRANSPARENT;
}

_FORCE_INLINE_ bool is_ligature(char32_t p_chr, char32_t p_nchr) {
	return (is_lam(p_chr) && is_alef(p_nchr));
}

_FORCE_INLINE_ bool is_connected_to_prev(char32_t p_chr, char32_t p_pchr) {
	int32_t prop = u_getIntPropertyValue(p_pchr, UCHAR_JOINING_TYPE);
	return (prop != U_JT_RIGHT_JOINING) && (prop != U_JT_NON_JOINING) ? !is_ligature(p_pchr, p_chr) : false;
}

_FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= 0x001f) || (p_char >= 0x007f && p_char <= 0x009F);
}

_FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == 0x0020) || (p_char == 0x00A0) || (p_char == 0x1680) || (p_char >= 0x2000 && p_char <= 0x200a) || (p_char == 0x202f) || (p_char == 0x205f) || (p_char == 0x3000) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char >= 0x0009 && p_char <= 0x000d) || (p_char == 0x0085);
}

_FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= 0x000a && p_char <= 0x000d) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
}

/*************************************************************************/

String TextServerAdvanced::interface_name = "ICU / HarfBuzz / Graphite";
uint32_t TextServerAdvanced::interface_features = FEATURE_BIDI_LAYOUT | FEATURE_VERTICAL_LAYOUT | FEATURE_SHAPING | FEATURE_KASHIDA_JUSTIFICATION | FEATURE_BREAK_ITERATORS | FEATURE_USE_SUPPORT_DATA | FEATURE_FONT_VARIABLE;

bool TextServerAdvanced::has_feature(Feature p_feature) {
	return (interface_features & p_feature) == p_feature;
}

String TextServerAdvanced::get_name() const {
	return interface_name;
}

void TextServerAdvanced::free(RID p_rid) {
	_THREAD_SAFE_METHOD_
	if (font_owner.owns(p_rid)) {
		FontDataAdvanced *fd = font_owner.getornull(p_rid);
		font_owner.free(p_rid);
		memdelete(fd);
	} else if (shaped_owner.owns(p_rid)) {
		ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_rid);
		shaped_owner.free(p_rid);
		memdelete(sd);
	}
}

bool TextServerAdvanced::has(RID p_rid) {
	_THREAD_SAFE_METHOD_
	return font_owner.owns(p_rid) || shaped_owner.owns(p_rid);
}

bool TextServerAdvanced::load_support_data(const String &p_filename) {
	_THREAD_SAFE_METHOD_

#ifdef ICU_STATIC_DATA
	if (icu_data == nullptr) {
		UErrorCode err = U_ZERO_ERROR;
		u_init(&err); // Do not check for errors, since we only load part of the data.
		icu_data = (uint8_t *)&U_ICUDATA_ENTRY_POINT;
	}
#else
	if (icu_data == nullptr) {
		String filename = (p_filename.is_empty()) ? String("res://") + _MKSTR(ICU_DATA_NAME) : p_filename;

		FileAccess *f = FileAccess::open(filename, FileAccess::READ);
		if (!f) {
			return false;
		}

		UErrorCode err = U_ZERO_ERROR;

		// ICU data found.
		uint64_t len = f->get_length();
		icu_data = (uint8_t *)memalloc(len);
		f->get_buffer(icu_data, len);
		f->close();
		memdelete(f);

		udata_setCommonData(icu_data, &err);
		if (U_FAILURE(err)) {
			memfree(icu_data);
			icu_data = nullptr;
			ERR_FAIL_V_MSG(false, u_errorName(err));
		}

		err = U_ZERO_ERROR;
		u_init(&err);
		if (U_FAILURE(err)) {
			memfree(icu_data);
			icu_data = nullptr;
			ERR_FAIL_V_MSG(false, u_errorName(err));
		}
	}
#endif
	return true;
}

#ifdef TOOLS_ENABLED

bool TextServerAdvanced::save_support_data(const String &p_filename) {
	_THREAD_SAFE_METHOD_
#ifdef ICU_STATIC_DATA

	// Store data to the res file if it's available.
	FileAccess *f = FileAccess::open(p_filename, FileAccess::WRITE);
	if (!f) {
		return false;
	}
	f->store_buffer(U_ICUDATA_ENTRY_POINT, U_ICUDATA_SIZE);
	f->close();
	memdelete(f);
	return true;

#else
	return false;
#endif
}

#endif

bool TextServerAdvanced::is_locale_right_to_left(const String &p_locale) {
	String l = p_locale.get_slicec('_', 0);
	if ((l == "ar") || (l == "dv") || (l == "he") || (l == "fa") || (l == "ff") || (l == "ku") || (l == "ur")) {
		return true;
	} else {
		return false;
	}
}

struct FeatureInfo {
	int32_t tag;
	String name;
};

static FeatureInfo feature_set[] = {
	// Registered OpenType feature tags.
	{ HB_TAG('a', 'a', 'l', 't'), "access_all_alternates" },
	{ HB_TAG('a', 'b', 'v', 'f'), "above_base_forms" },
	{ HB_TAG('a', 'b', 'v', 'm'), "above_base_mark_positioning" },
	{ HB_TAG('a', 'b', 'v', 's'), "above_base_substitutions" },
	{ HB_TAG('a', 'f', 'r', 'c'), "alternative_fractions" },
	{ HB_TAG('a', 'k', 'h', 'n'), "akhands" },
	{ HB_TAG('b', 'l', 'w', 'f'), "below_base_forms" },
	{ HB_TAG('b', 'l', 'w', 'm'), "below_base_mark_positioning" },
	{ HB_TAG('b', 'l', 'w', 's'), "below_base_substitutions" },
	{ HB_TAG('c', 'a', 'l', 't'), "contextual_alternates" },
	{ HB_TAG('c', 'a', 's', 'e'), "case_sensitive_forms" },
	{ HB_TAG('c', 'c', 'm', 'p'), "glyph_composition" },
	{ HB_TAG('c', 'f', 'a', 'r'), "conjunct_form_after_ro" },
	{ HB_TAG('c', 'j', 'c', 't'), "conjunct_forms" },
	{ HB_TAG('c', 'l', 'i', 'g'), "contextual_ligatures" },
	{ HB_TAG('c', 'p', 'c', 't'), "centered_cjk_punctuation" },
	{ HB_TAG('c', 'p', 's', 'p'), "capital_spacing" },
	{ HB_TAG('c', 's', 'w', 'h'), "contextual_swash" },
	{ HB_TAG('c', 'u', 'r', 's'), "cursive_positioning" },
	{ HB_TAG('c', 'v', '0', '1'), "character_variant_01" },
	{ HB_TAG('c', 'v', '0', '2'), "character_variant_02" },
	{ HB_TAG('c', 'v', '0', '3'), "character_variant_03" },
	{ HB_TAG('c', 'v', '0', '4'), "character_variant_04" },
	{ HB_TAG('c', 'v', '0', '5'), "character_variant_05" },
	{ HB_TAG('c', 'v', '0', '6'), "character_variant_06" },
	{ HB_TAG('c', 'v', '0', '7'), "character_variant_07" },
	{ HB_TAG('c', 'v', '0', '8'), "character_variant_08" },
	{ HB_TAG('c', 'v', '0', '9'), "character_variant_09" },
	{ HB_TAG('c', 'v', '1', '0'), "character_variant_10" },
	{ HB_TAG('c', 'v', '1', '1'), "character_variant_11" },
	{ HB_TAG('c', 'v', '1', '2'), "character_variant_12" },
	{ HB_TAG('c', 'v', '1', '3'), "character_variant_13" },
	{ HB_TAG('c', 'v', '1', '4'), "character_variant_14" },
	{ HB_TAG('c', 'v', '1', '5'), "character_variant_15" },
	{ HB_TAG('c', 'v', '1', '6'), "character_variant_16" },
	{ HB_TAG('c', 'v', '1', '7'), "character_variant_17" },
	{ HB_TAG('c', 'v', '1', '8'), "character_variant_18" },
	{ HB_TAG('c', 'v', '1', '9'), "character_variant_19" },
	{ HB_TAG('c', 'v', '2', '0'), "character_variant_20" },
	{ HB_TAG('c', 'v', '2', '1'), "character_variant_21" },
	{ HB_TAG('c', 'v', '2', '2'), "character_variant_22" },
	{ HB_TAG('c', 'v', '2', '3'), "character_variant_23" },
	{ HB_TAG('c', 'v', '2', '4'), "character_variant_24" },
	{ HB_TAG('c', 'v', '2', '5'), "character_variant_25" },
	{ HB_TAG('c', 'v', '2', '6'), "character_variant_26" },
	{ HB_TAG('c', 'v', '2', '7'), "character_variant_27" },
	{ HB_TAG('c', 'v', '2', '8'), "character_variant_28" },
	{ HB_TAG('c', 'v', '2', '9'), "character_variant_29" },
	{ HB_TAG('c', 'v', '3', '0'), "character_variant_30" },
	{ HB_TAG('c', 'v', '3', '1'), "character_variant_31" },
	{ HB_TAG('c', 'v', '3', '2'), "character_variant_32" },
	{ HB_TAG('c', 'v', '3', '3'), "character_variant_33" },
	{ HB_TAG('c', 'v', '3', '4'), "character_variant_34" },
	{ HB_TAG('c', 'v', '3', '5'), "character_variant_35" },
	{ HB_TAG('c', 'v', '3', '6'), "character_variant_36" },
	{ HB_TAG('c', 'v', '3', '7'), "character_variant_37" },
	{ HB_TAG('c', 'v', '3', '8'), "character_variant_38" },
	{ HB_TAG('c', 'v', '3', '9'), "character_variant_39" },
	{ HB_TAG('c', 'v', '4', '0'), "character_variant_40" },
	{ HB_TAG('c', 'v', '4', '1'), "character_variant_41" },
	{ HB_TAG('c', 'v', '4', '2'), "character_variant_42" },
	{ HB_TAG('c', 'v', '4', '3'), "character_variant_43" },
	{ HB_TAG('c', 'v', '4', '4'), "character_variant_44" },
	{ HB_TAG('c', 'v', '4', '5'), "character_variant_45" },
	{ HB_TAG('c', 'v', '4', '6'), "character_variant_46" },
	{ HB_TAG('c', 'v', '4', '7'), "character_variant_47" },
	{ HB_TAG('c', 'v', '4', '8'), "character_variant_48" },
	{ HB_TAG('c', 'v', '4', '9'), "character_variant_49" },
	{ HB_TAG('c', 'v', '5', '0'), "character_variant_50" },
	{ HB_TAG('c', 'v', '5', '1'), "character_variant_51" },
	{ HB_TAG('c', 'v', '5', '2'), "character_variant_52" },
	{ HB_TAG('c', 'v', '5', '3'), "character_variant_53" },
	{ HB_TAG('c', 'v', '5', '4'), "character_variant_54" },
	{ HB_TAG('c', 'v', '5', '5'), "character_variant_55" },
	{ HB_TAG('c', 'v', '5', '6'), "character_variant_56" },
	{ HB_TAG('c', 'v', '5', '7'), "character_variant_57" },
	{ HB_TAG('c', 'v', '5', '8'), "character_variant_58" },
	{ HB_TAG('c', 'v', '5', '9'), "character_variant_59" },
	{ HB_TAG('c', 'v', '6', '0'), "character_variant_60" },
	{ HB_TAG('c', 'v', '6', '1'), "character_variant_61" },
	{ HB_TAG('c', 'v', '6', '2'), "character_variant_62" },
	{ HB_TAG('c', 'v', '6', '3'), "character_variant_63" },
	{ HB_TAG('c', 'v', '6', '4'), "character_variant_64" },
	{ HB_TAG('c', 'v', '6', '5'), "character_variant_65" },
	{ HB_TAG('c', 'v', '6', '6'), "character_variant_66" },
	{ HB_TAG('c', 'v', '6', '7'), "character_variant_67" },
	{ HB_TAG('c', 'v', '6', '8'), "character_variant_68" },
	{ HB_TAG('c', 'v', '6', '9'), "character_variant_69" },
	{ HB_TAG('c', 'v', '7', '0'), "character_variant_70" },
	{ HB_TAG('c', 'v', '7', '1'), "character_variant_71" },
	{ HB_TAG('c', 'v', '7', '2'), "character_variant_72" },
	{ HB_TAG('c', 'v', '7', '3'), "character_variant_73" },
	{ HB_TAG('c', 'v', '7', '4'), "character_variant_74" },
	{ HB_TAG('c', 'v', '7', '5'), "character_variant_75" },
	{ HB_TAG('c', 'v', '7', '6'), "character_variant_76" },
	{ HB_TAG('c', 'v', '7', '7'), "character_variant_77" },
	{ HB_TAG('c', 'v', '7', '8'), "character_variant_78" },
	{ HB_TAG('c', 'v', '7', '9'), "character_variant_79" },
	{ HB_TAG('c', 'v', '8', '0'), "character_variant_80" },
	{ HB_TAG('c', 'v', '8', '1'), "character_variant_81" },
	{ HB_TAG('c', 'v', '8', '2'), "character_variant_82" },
	{ HB_TAG('c', 'v', '8', '3'), "character_variant_83" },
	{ HB_TAG('c', 'v', '8', '4'), "character_variant_84" },
	{ HB_TAG('c', 'v', '8', '5'), "character_variant_85" },
	{ HB_TAG('c', 'v', '8', '6'), "character_variant_86" },
	{ HB_TAG('c', 'v', '8', '7'), "character_variant_87" },
	{ HB_TAG('c', 'v', '8', '8'), "character_variant_88" },
	{ HB_TAG('c', 'v', '8', '9'), "character_variant_89" },
	{ HB_TAG('c', 'v', '9', '0'), "character_variant_90" },
	{ HB_TAG('c', 'v', '9', '1'), "character_variant_91" },
	{ HB_TAG('c', 'v', '9', '2'), "character_variant_92" },
	{ HB_TAG('c', 'v', '9', '3'), "character_variant_93" },
	{ HB_TAG('c', 'v', '9', '4'), "character_variant_94" },
	{ HB_TAG('c', 'v', '9', '5'), "character_variant_95" },
	{ HB_TAG('c', 'v', '9', '6'), "character_variant_96" },
	{ HB_TAG('c', 'v', '9', '7'), "character_variant_97" },
	{ HB_TAG('c', 'v', '9', '8'), "character_variant_98" },
	{ HB_TAG('c', 'v', '9', '9'), "character_variant_99" },
	{ HB_TAG('c', '2', 'p', 'c'), "petite_capitals_from_capitals" },
	{ HB_TAG('c', '2', 's', 'c'), "small_capitals_from_capitals" },
	{ HB_TAG('d', 'i', 's', 't'), "distances" },
	{ HB_TAG('d', 'l', 'i', 'g'), "discretionary_ligatures" },
	{ HB_TAG('d', 'n', 'o', 'm'), "denominators" },
	{ HB_TAG('d', 't', 'l', 's'), "dotless_forms" },
	{ HB_TAG('e', 'x', 'p', 't'), "expert_forms" },
	{ HB_TAG('f', 'a', 'l', 't'), "final_glyph_on_line_alternates" },
	{ HB_TAG('f', 'i', 'n', '2'), "terminal_forms_2" },
	{ HB_TAG('f', 'i', 'n', '3'), "terminal_forms_3" },
	{ HB_TAG('f', 'i', 'n', 'a'), "terminal_forms" },
	{ HB_TAG('f', 'l', 'a', 'c'), "flattened_accent_forms" },
	{ HB_TAG('f', 'r', 'a', 'c'), "fractions" },
	{ HB_TAG('f', 'w', 'i', 'd'), "full_widths" },
	{ HB_TAG('h', 'a', 'l', 'f'), "half_forms" },
	{ HB_TAG('h', 'a', 'l', 'n'), "halant_forms" },
	{ HB_TAG('h', 'a', 'l', 't'), "alternate_half_widths" },
	{ HB_TAG('h', 'i', 's', 't'), "historical_forms" },
	{ HB_TAG('h', 'k', 'n', 'a'), "horizontal_kana_alternates" },
	{ HB_TAG('h', 'l', 'i', 'g'), "historical_ligatures" },
	{ HB_TAG('h', 'n', 'g', 'l'), "hangul" },
	{ HB_TAG('h', 'o', 'j', 'o'), "hojo_kanji_forms" },
	{ HB_TAG('h', 'w', 'i', 'd'), "half_widths" },
	{ HB_TAG('i', 'n', 'i', 't'), "initial_forms" },
	{ HB_TAG('i', 's', 'o', 'l'), "isolated_forms" },
	{ HB_TAG('i', 't', 'a', 'l'), "italics" },
	{ HB_TAG('j', 'a', 'l', 't'), "justification_alternates" },
	{ HB_TAG('j', 'p', '7', '8'), "jis78_forms" },
	{ HB_TAG('j', 'p', '8', '3'), "jis83_forms" },
	{ HB_TAG('j', 'p', '9', '0'), "jis90_forms" },
	{ HB_TAG('j', 'p', '0', '4'), "jis2004_forms" },
	{ HB_TAG('k', 'e', 'r', 'n'), "kerning" },
	{ HB_TAG('l', 'f', 'b', 'd'), "left_bounds" },
	{ HB_TAG('l', 'i', 'g', 'a'), "standard_ligatures" },
	{ HB_TAG('l', 'j', 'm', 'o'), "leading_jamo_forms" },
	{ HB_TAG('l', 'n', 'u', 'm'), "lining_figures" },
	{ HB_TAG('l', 'o', 'c', 'l'), "localized_forms" },
	{ HB_TAG('l', 't', 'r', 'a'), "left_to_right_alternates" },
	{ HB_TAG('l', 't', 'r', 'm'), "left_to_right_mirrored_forms" },
	{ HB_TAG('m', 'a', 'r', 'k'), "mark_positioning" },
	{ HB_TAG('m', 'e', 'd', '2'), "medial_forms_2" },
	{ HB_TAG('m', 'e', 'd', 'i'), "medial_forms" },
	{ HB_TAG('m', 'g', 'r', 'k'), "mathematical_greek" },
	{ HB_TAG('m', 'k', 'm', 'k'), "mark_to_mark_positioning" },
	{ HB_TAG('m', 's', 'e', 't'), "mark_positioning_via_substitution" },
	{ HB_TAG('n', 'a', 'l', 't'), "alternate_annotation_forms" },
	{ HB_TAG('n', 'l', 'c', 'k'), "nlc_kanji_forms" },
	{ HB_TAG('n', 'u', 'k', 't'), "nukta_forms" },
	{ HB_TAG('n', 'u', 'm', 'r'), "numerators" },
	{ HB_TAG('o', 'n', 'u', 'm'), "oldstyle_figures" },
	{ HB_TAG('o', 'p', 'b', 'd'), "optical_bounds" },
	{ HB_TAG('o', 'r', 'd', 'n'), "ordinals" },
	{ HB_TAG('o', 'r', 'n', 'm'), "ornaments" },
	{ HB_TAG('p', 'a', 'l', 't'), "proportional_alternate_widths" },
	{ HB_TAG('p', 'c', 'a', 'p'), "petite_capitals" },
	{ HB_TAG('p', 'k', 'n', 'a'), "proportional_kana" },
	{ HB_TAG('p', 'n', 'u', 'm'), "proportional_figures" },
	{ HB_TAG('p', 'r', 'e', 'f'), "pre_base_forms" },
	{ HB_TAG('p', 'r', 'e', 's'), "pre_base_substitutions" },
	{ HB_TAG('p', 's', 't', 'f'), "post_base_forms" },
	{ HB_TAG('p', 's', 't', 's'), "post_base_substitutions" },
	{ HB_TAG('p', 'w', 'i', 'd'), "proportional_widths" },
	{ HB_TAG('q', 'w', 'i', 'd'), "quarter_widths" },
	{ HB_TAG('r', 'a', 'n', 'd'), "randomize" },
	{ HB_TAG('r', 'c', 'l', 't'), "required_contextual_alternates" },
	{ HB_TAG('r', 'k', 'r', 'f'), "rakar_forms" },
	{ HB_TAG('r', 'l', 'i', 'g'), "required_ligatures" },
	{ HB_TAG('r', 'p', 'h', 'f'), "reph_forms" },
	{ HB_TAG('r', 't', 'b', 'd'), "right_bounds" },
	{ HB_TAG('r', 't', 'l', 'a'), "right_to_left_alternates" },
	{ HB_TAG('r', 't', 'l', 'm'), "right_to_left_mirrored_forms" },
	{ HB_TAG('r', 'u', 'b', 'y'), "ruby_notation_forms" },
	{ HB_TAG('r', 'v', 'r', 'n'), "required_variation_alternates" },
	{ HB_TAG('s', 'a', 'l', 't'), "stylistic_alternates" },
	{ HB_TAG('s', 'i', 'n', 'f'), "scientific_inferiors" },
	{ HB_TAG('s', 'i', 'z', 'e'), "optical_size" },
	{ HB_TAG('s', 'm', 'c', 'p'), "small_capitals" },
	{ HB_TAG('s', 'm', 'p', 'l'), "simplified_forms" },
	{ HB_TAG('s', 's', '0', '1'), "stylistic_set_01" },
	{ HB_TAG('s', 's', '0', '2'), "stylistic_set_02" },
	{ HB_TAG('s', 's', '0', '3'), "stylistic_set_03" },
	{ HB_TAG('s', 's', '0', '4'), "stylistic_set_04" },
	{ HB_TAG('s', 's', '0', '5'), "stylistic_set_05" },
	{ HB_TAG('s', 's', '0', '6'), "stylistic_set_06" },
	{ HB_TAG('s', 's', '0', '7'), "stylistic_set_07" },
	{ HB_TAG('s', 's', '0', '8'), "stylistic_set_08" },
	{ HB_TAG('s', 's', '0', '9'), "stylistic_set_09" },
	{ HB_TAG('s', 's', '1', '0'), "stylistic_set_10" },
	{ HB_TAG('s', 's', '1', '1'), "stylistic_set_11" },
	{ HB_TAG('s', 's', '1', '2'), "stylistic_set_12" },
	{ HB_TAG('s', 's', '1', '3'), "stylistic_set_13" },
	{ HB_TAG('s', 's', '1', '4'), "stylistic_set_14" },
	{ HB_TAG('s', 's', '1', '5'), "stylistic_set_15" },
	{ HB_TAG('s', 's', '1', '6'), "stylistic_set_16" },
	{ HB_TAG('s', 's', '1', '7'), "stylistic_set_17" },
	{ HB_TAG('s', 's', '1', '8'), "stylistic_set_18" },
	{ HB_TAG('s', 's', '1', '9'), "stylistic_set_19" },
	{ HB_TAG('s', 's', '2', '0'), "stylistic_set_20" },
	{ HB_TAG('s', 's', 't', 'y'), "math_script_style_alternates" },
	{ HB_TAG('s', 't', 'c', 'h'), "stretching_glyph_decomposition" },
	{ HB_TAG('s', 'u', 'b', 's'), "subscript" },
	{ HB_TAG('s', 'u', 'p', 's'), "superscript" },
	{ HB_TAG('s', 'w', 's', 'h'), "swash" },
	{ HB_TAG('t', 'i', 't', 'l'), "titling" },
	{ HB_TAG('t', 'j', 'm', 'o'), "trailing_jamo_forms" },
	{ HB_TAG('t', 'n', 'a', 'm'), "traditional_name_forms" },
	{ HB_TAG('t', 'n', 'u', 'm'), "tabular_figures" },
	{ HB_TAG('t', 'r', 'a', 'd'), "traditional_forms" },
	{ HB_TAG('t', 'w', 'i', 'd'), "third_widths" },
	{ HB_TAG('u', 'n', 'i', 'c'), "unicase" },
	{ HB_TAG('v', 'a', 'l', 't'), "alternate_vertical_metrics" },
	{ HB_TAG('v', 'a', 't', 'u'), "vattu_variants" },
	{ HB_TAG('v', 'e', 'r', 't'), "vertical_writing" },
	{ HB_TAG('v', 'h', 'a', 'l'), "alternate_vertical_half_metrics" },
	{ HB_TAG('v', 'j', 'm', 'o'), "vowel_jamo_forms" },
	{ HB_TAG('v', 'k', 'n', 'a'), "vertical_kana_alternates" },
	{ HB_TAG('v', 'k', 'r', 'n'), "vertical_kerning" },
	{ HB_TAG('v', 'p', 'a', 'l'), "proportional_alternate_vertical_metrics" },
	{ HB_TAG('v', 'r', 't', '2'), "vertical_alternates_and_rotation" },
	{ HB_TAG('v', 'r', 't', 'r'), "vertical_alternates_for_rotation" },
	{ HB_TAG('z', 'e', 'r', 'o'), "slashed_zero" },
	// Registered OpenType variation tags.
	{ HB_TAG('i', 't', 'a', 'l'), "italic" },
	{ HB_TAG('o', 'p', 's', 'z'), "optical_size" },
	{ HB_TAG('s', 'l', 'n', 't'), "slant" },
	{ HB_TAG('w', 'd', 't', 'h'), "width" },
	{ HB_TAG('w', 'g', 'h', 't'), "weight" },
	{ 0, String() },
};

int32_t TextServerAdvanced::name_to_tag(const String &p_name) {
	for (int i = 0; feature_set[i].tag != 0; i++) {
		if (feature_set[i].name == p_name) {
			return feature_set[i].tag;
		}
	}

	// No readable name, use tag string.
	return hb_tag_from_string(p_name.replace("custom_", "").ascii().get_data(), -1);
}

String TextServerAdvanced::tag_to_name(int32_t p_tag) {
	for (int i = 0; feature_set[i].tag != 0; i++) {
		if (feature_set[i].tag == p_tag) {
			return feature_set[i].name;
		}
	}

	// No readable name, use tag string.
	char name[5];
	memset(name, 0, 5);
	hb_tag_to_string(p_tag, name);
	return String("custom_") + String(name);
}

/*************************************************************************/
/* Font interface */
/*************************************************************************/

RID TextServerAdvanced::create_font_system(const String &p_name, int p_base_size) {
	ERR_FAIL_V_MSG(RID(), "System fonts are not supported by this text server.");
}

RID TextServerAdvanced::create_font_resource(const String &p_filename, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = nullptr;
	if (p_filename.get_extension() == "fnt" || p_filename.get_extension() == "font") {
		fd = memnew(BitmapFontDataAdvanced);
#ifdef MODULE_FREETYPE_ENABLED
	} else if (p_filename.get_extension() == "ttf" || p_filename.get_extension() == "otf" || p_filename.get_extension() == "woff") {
		fd = memnew(DynamicFontDataAdvanced);
#endif
	} else {
		return RID();
	}

	Error err = fd->load_from_file(p_filename, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

RID TextServerAdvanced::create_font_memory(const uint8_t *p_data, size_t p_size, const String &p_type, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = nullptr;
	if (p_type == "fnt" || p_type == "font") {
		fd = memnew(BitmapFontDataAdvanced);
#ifdef MODULE_FREETYPE_ENABLED
	} else if (p_type == "ttf" || p_type == "otf" || p_type == "woff") {
		fd = memnew(DynamicFontDataAdvanced);
#endif
	} else {
		return RID();
	}

	Error err = fd->load_from_memory(p_data, p_size, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

RID TextServerAdvanced::create_font_bitmap(float p_height, float p_ascent, int p_base_size) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = memnew(BitmapFontDataAdvanced);
	Error err = fd->bitmap_new(p_height, p_ascent, p_base_size);
	if (err != OK) {
		memdelete(fd);
		return RID();
	}

	return font_owner.make_rid(fd);
}

void TextServerAdvanced::font_bitmap_add_texture(RID p_font, const Ref<Texture> &p_texture) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_texture(p_texture);
}

void TextServerAdvanced::font_bitmap_add_char(RID p_font, char32_t p_char, int p_texture_idx, const Rect2 &p_rect, const Size2 &p_align, float p_advance) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_char(p_char, p_texture_idx, p_rect, p_align, p_advance);
}

void TextServerAdvanced::font_bitmap_add_kerning_pair(RID p_font, char32_t p_A, char32_t p_B, int p_kerning) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->bitmap_add_kerning_pair(p_A, p_B, p_kerning);
}

float TextServerAdvanced::font_get_height(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_height(p_size);
}

float TextServerAdvanced::font_get_ascent(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_ascent(p_size);
}

float TextServerAdvanced::font_get_descent(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_descent(p_size);
}

float TextServerAdvanced::font_get_underline_position(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_underline_position(p_size);
}

float TextServerAdvanced::font_get_underline_thickness(RID p_font, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_underline_thickness(p_size);
}

int TextServerAdvanced::font_get_spacing_space(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_spacing_space();
}

void TextServerAdvanced::font_set_spacing_space(RID p_font, int p_value) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_spacing_space(p_value);
}

int TextServerAdvanced::font_get_spacing_glyph(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_spacing_glyph();
}

void TextServerAdvanced::font_set_spacing_glyph(RID p_font, int p_value) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_spacing_glyph(p_value);
}

void TextServerAdvanced::font_set_antialiased(RID p_font, bool p_antialiased) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_antialiased(p_antialiased);
}

Dictionary TextServerAdvanced::font_get_feature_list(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Dictionary());
	return fd->get_feature_list();
}

bool TextServerAdvanced::font_get_antialiased(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_antialiased();
}

Dictionary TextServerAdvanced::font_get_variation_list(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Dictionary());
	return fd->get_variation_list();
}

void TextServerAdvanced::font_set_variation(RID p_font, const String &p_name, double p_value) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_variation(p_name, p_value);
}

double TextServerAdvanced::font_get_variation(RID p_font, const String &p_name) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_variation(p_name);
}

void TextServerAdvanced::font_set_distance_field_hint(RID p_font, bool p_distance_field) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_distance_field_hint(p_distance_field);
}

bool TextServerAdvanced::font_get_distance_field_hint(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_distance_field_hint();
}

void TextServerAdvanced::font_set_hinting(RID p_font, TextServer::Hinting p_hinting) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_hinting(p_hinting);
}

TextServer::Hinting TextServerAdvanced::font_get_hinting(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, TextServer::HINTING_NONE);
	return fd->get_hinting();
}

void TextServerAdvanced::font_set_force_autohinter(RID p_font, bool p_enabeld) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->set_force_autohinter(p_enabeld);
}

bool TextServerAdvanced::font_get_force_autohinter(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_force_autohinter();
}

bool TextServerAdvanced::font_has_char(RID p_font, char32_t p_char) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->has_char(p_char);
}

String TextServerAdvanced::font_get_supported_chars(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, String());
	return fd->get_supported_chars();
}

bool TextServerAdvanced::font_has_outline(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->has_outline();
}

float TextServerAdvanced::font_get_base_size(RID p_font) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0.f);
	return fd->get_base_size();
}

bool TextServerAdvanced::font_is_language_supported(RID p_font, const String &p_language) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	if (fd->lang_support_overrides.has(p_language)) {
		return fd->lang_support_overrides[p_language];
	} else {
		Vector<String> tags = p_language.replace("-", "_").split("_");
		if (tags.size() > 0) {
			if (fd->lang_support_overrides.has(tags[0])) {
				return fd->lang_support_overrides[tags[0]];
			}
		}
		return fd->is_lang_supported(p_language);
	}
}

void TextServerAdvanced::font_set_language_support_override(RID p_font, const String &p_language, bool p_supported) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->lang_support_overrides[p_language] = p_supported;
}

bool TextServerAdvanced::font_get_language_support_override(RID p_font, const String &p_language) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->lang_support_overrides[p_language];
}

void TextServerAdvanced::font_remove_language_support_override(RID p_font, const String &p_language) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->lang_support_overrides.erase(p_language);
}

Vector<String> TextServerAdvanced::font_get_language_support_overrides(RID p_font) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector<String>());
	Vector<String> ret;
	for (Map<String, bool>::Element *E = fd->lang_support_overrides.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}
	return ret;
}

bool TextServerAdvanced::font_is_script_supported(RID p_font, const String &p_script) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	if (fd->script_support_overrides.has(p_script)) {
		return fd->script_support_overrides[p_script];
	} else {
		hb_script_t scr = hb_script_from_string(p_script.ascii().get_data(), -1);
		return fd->is_script_supported(scr);
	}
}

void TextServerAdvanced::font_set_script_support_override(RID p_font, const String &p_script, bool p_supported) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->script_support_overrides[p_script] = p_supported;
}

bool TextServerAdvanced::font_get_script_support_override(RID p_font, const String &p_script) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->script_support_overrides[p_script];
}

void TextServerAdvanced::font_remove_script_support_override(RID p_font, const String &p_script) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND(!fd);
	fd->script_support_overrides.erase(p_script);
}

Vector<String> TextServerAdvanced::font_get_script_support_overrides(RID p_font) {
	_THREAD_SAFE_METHOD_
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector<String>());
	Vector<String> ret;
	for (Map<String, bool>::Element *E = fd->script_support_overrides.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}
	return ret;
}

uint32_t TextServerAdvanced::font_get_glyph_index(RID p_font, char32_t p_char, char32_t p_variation_selector) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, 0);
	return fd->get_glyph_index(p_char, p_variation_selector);
}

Vector2 TextServerAdvanced::font_get_glyph_advance(RID p_font, uint32_t p_index, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->get_advance(p_index, p_size);
}

Vector2 TextServerAdvanced::font_get_glyph_kerning(RID p_font, uint32_t p_index_a, uint32_t p_index_b, int p_size) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->get_kerning(p_index_a, p_index_b, p_size);
}

Vector2 TextServerAdvanced::font_draw_glyph(RID p_font, RID p_canvas, int p_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->draw_glyph(p_canvas, p_size, p_pos, p_index, p_color);
}

Vector2 TextServerAdvanced::font_draw_glyph_outline(RID p_font, RID p_canvas, int p_size, int p_outline_size, const Vector2 &p_pos, uint32_t p_index, const Color &p_color) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, Vector2());
	return fd->draw_glyph_outline(p_canvas, p_size, p_outline_size, p_pos, p_index, p_color);
}

bool TextServerAdvanced::font_get_glyph_contours(RID p_font, int p_size, uint32_t p_index, Vector<Vector3> &r_points, Vector<int32_t> &r_contours, bool &r_orientation) const {
	_THREAD_SAFE_METHOD_
	const FontDataAdvanced *fd = font_owner.getornull(p_font);
	ERR_FAIL_COND_V(!fd, false);
	return fd->get_glyph_contours(p_size, p_index, r_points, r_contours, r_orientation);
}

float TextServerAdvanced::font_get_oversampling() const {
	return oversampling;
}

void TextServerAdvanced::font_set_oversampling(float p_oversampling) {
	_THREAD_SAFE_METHOD_
	if (oversampling != p_oversampling) {
		oversampling = p_oversampling;
		List<RID> fonts;
		font_owner.get_owned_list(&fonts);
		for (List<RID>::Element *E = fonts.front(); E; E = E->next()) {
			font_owner.getornull(E->get())->clear_cache();
		}

		List<RID> text_bufs;
		shaped_owner.get_owned_list(&text_bufs);
		for (List<RID>::Element *E = text_bufs.front(); E; E = E->next()) {
			invalidate(shaped_owner.getornull(E->get()));
		}
	}
}

Vector<String> TextServerAdvanced::get_system_fonts() const {
	return Vector<String>();
}

/*************************************************************************/
/* Shaped text buffer interface                                          */
/*************************************************************************/

int TextServerAdvanced::_convert_pos(const ShapedTextDataAdvanced *p_sd, int p_pos) const {
	int32_t limit = p_pos;
	if (p_sd->text.length() != p_sd->utf16.length()) {
		const UChar *data = p_sd->utf16.ptr();
		for (int i = 0; i < p_pos; i++) {
			if (U16_IS_LEAD(data[i])) {
				limit--;
			}
		}
	}
	return limit;
}

int TextServerAdvanced::_convert_pos_inv(const ShapedTextDataAdvanced *p_sd, int p_pos) const {
	int32_t limit = p_pos;
	if (p_sd->text.length() != p_sd->utf16.length()) {
		for (int i = 0; i < p_pos; i++) {
			if (p_sd->text[i] > 0xFFFF) {
				limit++;
			}
		}
	}
	return limit;
}

void TextServerAdvanced::invalidate(TextServerAdvanced::ShapedTextDataAdvanced *p_shaped) {
	p_shaped->valid = false;
	p_shaped->sort_valid = false;
	p_shaped->line_breaks_valid = false;
	p_shaped->justification_ops_valid = false;
	p_shaped->ascent = 0.f;
	p_shaped->descent = 0.f;
	p_shaped->width = 0.f;
	p_shaped->upos = 0.f;
	p_shaped->uthk = 0.f;
	p_shaped->glyphs.clear();
	p_shaped->glyphs_logical.clear();
	p_shaped->utf16 = Char16String();
	if (p_shaped->script_iter != nullptr) {
		memdelete(p_shaped->script_iter);
		p_shaped->script_iter = nullptr;
	}
	for (int i = 0; i < p_shaped->bidi_iter.size(); i++) {
		ubidi_close(p_shaped->bidi_iter[i]);
	}
	p_shaped->bidi_iter.clear();
}

void TextServerAdvanced::full_copy(ShapedTextDataAdvanced *p_shaped) {
	ShapedTextDataAdvanced *parent = shaped_owner.getornull(p_shaped->parent);

	for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = parent->objects.front(); E; E = E->next()) {
		if (E->get().pos >= p_shaped->start && E->get().pos < p_shaped->end) {
			p_shaped->objects[E->key()] = E->get();
		}
	}

	for (int k = 0; k < parent->spans.size(); k++) {
		ShapedTextDataAdvanced::Span span = parent->spans[k];
		if (span.start >= p_shaped->end || span.end <= p_shaped->start) {
			continue;
		}
		span.start = MAX(p_shaped->start, span.start);
		span.end = MIN(p_shaped->end, span.end);
		p_shaped->spans.push_back(span);
	}

	p_shaped->parent = RID();
}

RID TextServerAdvanced::create_shaped_text(TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = memnew(ShapedTextDataAdvanced);
	sd->hb_buffer = hb_buffer_create();
	sd->direction = p_direction;
	sd->orientation = p_orientation;
	return shaped_owner.make_rid(sd);
}

void TextServerAdvanced::shaped_text_clear(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);

	sd->parent = RID();
	sd->start = 0;
	sd->end = 0;
	sd->text = String();
	sd->spans.clear();
	sd->objects.clear();
	sd->bidi_override.clear();
	invalidate(sd);
}

void TextServerAdvanced::shaped_text_set_direction(RID p_shaped, TextServer::Direction p_direction) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);

	if (sd->direction != p_direction) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->direction = p_direction;
		invalidate(sd);
	}
}

TextServer::Direction TextServerAdvanced::shaped_text_get_direction(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, TextServer::DIRECTION_LTR);
	return sd->direction;
}

void TextServerAdvanced::shaped_text_set_bidi_override(RID p_shaped, const Vector<Vector2i> &p_override) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->parent != RID()) {
		full_copy(sd);
	}
	sd->bidi_override = p_override;
	invalidate(sd);
}

void TextServerAdvanced::shaped_text_set_orientation(RID p_shaped, TextServer::Orientation p_orientation) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->orientation != p_orientation) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->orientation = p_orientation;
		invalidate(sd);
	}
}

void TextServerAdvanced::shaped_text_set_preserve_invalid(RID p_shaped, bool p_enabled) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	ERR_FAIL_COND(sd->parent != RID());
	if (sd->preserve_invalid != p_enabled) {
		sd->preserve_invalid = p_enabled;
		invalidate(sd);
	}
}

bool TextServerAdvanced::shaped_text_get_preserve_invalid(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->preserve_invalid;
}

void TextServerAdvanced::shaped_text_set_preserve_control(RID p_shaped, bool p_enabled) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND(!sd);
	if (sd->preserve_control != p_enabled) {
		if (sd->parent != RID()) {
			full_copy(sd);
		}
		sd->preserve_control = p_enabled;
		invalidate(sd);
	}
}

bool TextServerAdvanced::shaped_text_get_preserve_control(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->preserve_control;
}

TextServer::Orientation TextServerAdvanced::shaped_text_get_orientation(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, TextServer::ORIENTATION_HORIZONTAL);
	return sd->orientation;
}

bool TextServerAdvanced::shaped_text_add_string(RID p_shaped, const String &p_text, const Vector<RID> &p_fonts, int p_size, const Dictionary &p_opentype_features, const String &p_language) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(p_size <= 0, false);

	if (p_text.is_empty()) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextDataAdvanced::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_text.length();
	span.fonts = p_fonts; // Do not pre-sort, spans will be divided to subruns later.
	span.font_size = p_size;
	span.language = p_language;
	span.features = p_opentype_features;

	sd->spans.push_back(span);
	sd->text += p_text;
	sd->end += p_text.length();
	invalidate(sd);

	return true;
}

bool TextServerAdvanced::shaped_text_add_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align, int p_length) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(p_key == Variant(), false);
	ERR_FAIL_COND_V(sd->objects.has(p_key), false);

	if (sd->parent != RID()) {
		full_copy(sd);
	}

	ShapedTextDataAdvanced::Span span;
	span.start = sd->text.length();
	span.end = span.start + p_length;
	span.embedded_key = p_key;

	ShapedTextDataAdvanced::EmbeddedObject obj;
	obj.inline_align = p_inline_align;
	obj.rect.size = p_size;
	obj.pos = span.start;

	sd->spans.push_back(span);
	sd->text += String::chr(0xfffc).repeat(p_length);
	sd->end += p_length;
	sd->objects[p_key] = obj;
	invalidate(sd);

	return true;
}

bool TextServerAdvanced::shaped_text_resize_object(RID p_shaped, Variant p_key, const Size2 &p_size, VAlign p_inline_align) {
	_THREAD_SAFE_METHOD_
	ShapedTextData *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	ERR_FAIL_COND_V(!sd->objects.has(p_key), false);
	sd->objects[p_key].rect.size = p_size;
	sd->objects[p_key].inline_align = p_inline_align;
	if (sd->valid) {
		// Recalc string metrics.
		sd->ascent = 0;
		sd->descent = 0;
		sd->width = 0;
		sd->upos = 0;
		sd->uthk = 0;
		int sd_size = sd->glyphs.size();

		for (int i = 0; i < sd_size; i++) {
			Glyph gl = sd->glyphs[i];
			Variant key;
			if (gl.count == 1) {
				for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
					if (E->get().pos == gl.start) {
						key = E->key();
						break;
					}
				}
			}
			if (key != Variant()) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					sd->objects[key].rect.position.x = sd->width;
					sd->width += sd->objects[key].rect.size.x;
					switch (sd->objects[key].inline_align) {
						case VALIGN_TOP: {
							sd->ascent = MAX(sd->ascent, sd->objects[key].rect.size.y);
						} break;
						case VALIGN_CENTER: {
							sd->ascent = MAX(sd->ascent, Math::round(sd->objects[key].rect.size.y / 2));
							sd->descent = MAX(sd->descent, Math::round(sd->objects[key].rect.size.y / 2));
						} break;
						case VALIGN_BOTTOM: {
							sd->descent = MAX(sd->descent, sd->objects[key].rect.size.y);
						} break;
					}
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.x;
				} else {
					sd->objects[key].rect.position.y = sd->width;
					sd->width += sd->objects[key].rect.size.y;
					switch (sd->objects[key].inline_align) {
						case VALIGN_TOP: {
							sd->ascent = MAX(sd->ascent, sd->objects[key].rect.size.x);
						} break;
						case VALIGN_CENTER: {
							sd->ascent = MAX(sd->ascent, Math::round(sd->objects[key].rect.size.x / 2));
							sd->descent = MAX(sd->descent, Math::round(sd->objects[key].rect.size.x / 2));
						} break;
						case VALIGN_BOTTOM: {
							sd->descent = MAX(sd->descent, sd->objects[key].rect.size.x);
						} break;
					}
					sd->glyphs.write[i].advance = sd->objects[key].rect.size.y;
				}
			} else {
				const FontDataAdvanced *fd = font_owner.getornull(gl.font_rid);
				if (fd != nullptr) {
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, MAX(fd->get_ascent(gl.font_size), -gl.y_off));
						sd->descent = MAX(sd->descent, MAX(fd->get_descent(gl.font_size), gl.y_off));
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
						sd->descent = MAX(sd->descent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
					}
					sd->upos = MAX(sd->upos, font_get_underline_position(gl.font_rid, gl.font_size));
					sd->uthk = MAX(sd->uthk, font_get_underline_thickness(gl.font_rid, gl.font_size));
				} else if (sd->preserve_invalid || (sd->preserve_control && is_control(gl.index))) {
					// Glyph not found, replace with hex code box.
					if (sd->orientation == ORIENTATION_HORIZONTAL) {
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.75f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.25f));
					} else {
						sd->ascent = MAX(sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
						sd->descent = MAX(sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
					}
				}
				sd->width += gl.advance * gl.repeat;
			}
		}

		// Align embedded objects to baseline.
		for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
			if ((E->get().pos >= sd->start) && (E->get().pos < sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.y = -sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.y = -(E->get().rect.size.y / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.y = sd->descent - E->get().rect.size.y;
						} break;
					}
				} else {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.x = -sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.x = -(E->get().rect.size.x / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.x = sd->descent - E->get().rect.size.x;
						} break;
					}
				}
			}
		}
	}
	return true;
}

RID TextServerAdvanced::shaped_text_substr(RID p_shaped, int p_start, int p_length) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());
	if (sd->parent != RID()) {
		return shaped_text_substr(sd->parent, p_start, p_length);
	}
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	ERR_FAIL_COND_V(p_start < 0 || p_length < 0, RID());
	ERR_FAIL_COND_V(sd->start > p_start || sd->end < p_start, RID());
	ERR_FAIL_COND_V(sd->end < p_start + p_length, RID());

	ShapedTextDataAdvanced *new_sd = memnew(ShapedTextDataAdvanced);

	new_sd->hb_buffer = hb_buffer_create();
	new_sd->parent = p_shaped;
	new_sd->start = p_start;
	new_sd->end = p_start + p_length;

	new_sd->orientation = sd->orientation;
	new_sd->direction = sd->direction;
	new_sd->para_direction = sd->para_direction;
	new_sd->line_breaks_valid = sd->line_breaks_valid;
	new_sd->justification_ops_valid = sd->justification_ops_valid;
	new_sd->sort_valid = false;
	new_sd->upos = sd->upos;
	new_sd->uthk = sd->uthk;

	if (p_length > 0) {
		new_sd->text = sd->text.substr(p_start, p_length);
		new_sd->utf16 = new_sd->text.utf16();
		new_sd->script_iter = memnew(ScriptIterator(new_sd->text, 0, new_sd->text.length()));

		int sd_size = sd->glyphs.size();
		const Glyph *sd_glyphs = sd->glyphs.ptr();
		const FontDataAdvanced *fd = nullptr;
		RID prev_rid = RID();

		for (int ov = 0; ov < sd->bidi_override.size(); ov++) {
			UErrorCode err = U_ZERO_ERROR;

			if (sd->bidi_override[ov].x >= p_start + p_length || sd->bidi_override[ov].y <= p_start) {
				continue;
			}
			int start = _convert_pos_inv(sd, MAX(0, p_start - sd->bidi_override[ov].x));
			int end = _convert_pos_inv(sd, MIN(p_start + p_length, sd->bidi_override[ov].y) - sd->bidi_override[ov].x);

			ERR_FAIL_COND_V_MSG((start < 0 || end - start > new_sd->utf16.length()), RID(), "Invalid BiDi override range.");

			//Create temporary line bidi & shape
			UBiDi *bidi_iter = ubidi_openSized(end - start, 0, &err);
			ERR_FAIL_COND_V_MSG(U_FAILURE(err), RID(), u_errorName(err));
			ubidi_setLine(sd->bidi_iter[ov], start, end, bidi_iter, &err);
			if (U_FAILURE(err)) {
				ubidi_close(bidi_iter);
				ERR_FAIL_V_MSG(RID(), u_errorName(err));
			}
			new_sd->bidi_iter.push_back(bidi_iter);

			err = U_ZERO_ERROR;
			int bidi_run_count = ubidi_countRuns(bidi_iter, &err);
			ERR_FAIL_COND_V_MSG(U_FAILURE(err), RID(), u_errorName(err));
			for (int i = 0; i < bidi_run_count; i++) {
				int32_t _bidi_run_start = 0;
				int32_t _bidi_run_length = 0;
				ubidi_getVisualRun(bidi_iter, i, &_bidi_run_start, &_bidi_run_length);

				int32_t bidi_run_start = _convert_pos(sd, sd->bidi_override[ov].x + start + _bidi_run_start);
				int32_t bidi_run_end = _convert_pos(sd, sd->bidi_override[ov].x + start + _bidi_run_start + _bidi_run_length);

				for (int j = 0; j < sd_size; j++) {
					if ((sd_glyphs[j].start >= bidi_run_start) && (sd_glyphs[j].end <= bidi_run_end)) {
						// Copy glyphs.
						Glyph gl = sd_glyphs[j];
						Variant key;
						bool find_embedded = false;
						if (gl.count == 1) {
							for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
								if (E->get().pos == gl.start) {
									find_embedded = true;
									key = E->key();
									new_sd->objects[key] = E->get();
									break;
								}
							}
						}
						if (find_embedded) {
							if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
								new_sd->objects[key].rect.position.x = new_sd->width;
								new_sd->width += new_sd->objects[key].rect.size.x;
								switch (new_sd->objects[key].inline_align) {
									case VALIGN_TOP: {
										new_sd->ascent = MAX(new_sd->ascent, new_sd->objects[key].rect.size.y);
									} break;
									case VALIGN_CENTER: {
										new_sd->ascent = MAX(new_sd->ascent, Math::round(new_sd->objects[key].rect.size.y / 2));
										new_sd->descent = MAX(new_sd->descent, Math::round(new_sd->objects[key].rect.size.y / 2));
									} break;
									case VALIGN_BOTTOM: {
										new_sd->descent = MAX(new_sd->descent, new_sd->objects[key].rect.size.y);
									} break;
								}
							} else {
								new_sd->objects[key].rect.position.y = new_sd->width;
								new_sd->width += new_sd->objects[key].rect.size.y;
								switch (new_sd->objects[key].inline_align) {
									case VALIGN_TOP: {
										new_sd->ascent = MAX(new_sd->ascent, new_sd->objects[key].rect.size.x);
									} break;
									case VALIGN_CENTER: {
										new_sd->ascent = MAX(new_sd->ascent, Math::round(new_sd->objects[key].rect.size.x / 2));
										new_sd->descent = MAX(new_sd->descent, Math::round(new_sd->objects[key].rect.size.x / 2));
									} break;
									case VALIGN_BOTTOM: {
										new_sd->descent = MAX(new_sd->descent, new_sd->objects[key].rect.size.x);
									} break;
								}
							}
						} else {
							if (prev_rid != gl.font_rid) {
								fd = font_owner.getornull(gl.font_rid);
								prev_rid = gl.font_rid;
							}
							if (fd != nullptr) {
								if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
									new_sd->ascent = MAX(new_sd->ascent, MAX(fd->get_ascent(gl.font_size), -gl.y_off));
									new_sd->descent = MAX(new_sd->descent, MAX(fd->get_descent(gl.font_size), gl.y_off));
								} else {
									new_sd->ascent = MAX(new_sd->ascent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
									new_sd->descent = MAX(new_sd->descent, Math::round(fd->get_advance(gl.index, gl.font_size).x * 0.5));
								}
							} else if (new_sd->preserve_invalid || (new_sd->preserve_control && is_control(gl.index))) {
								// Glyph not found, replace with hex code box.
								if (new_sd->orientation == ORIENTATION_HORIZONTAL) {
									new_sd->ascent = MAX(new_sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.75f));
									new_sd->descent = MAX(new_sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).y * 0.25f));
								} else {
									new_sd->ascent = MAX(new_sd->ascent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
									new_sd->descent = MAX(new_sd->descent, Math::round(get_hex_code_box_size(gl.font_size, gl.index).x * 0.5f));
								}
							}
							new_sd->width += gl.advance * gl.repeat;
						}
						new_sd->glyphs.push_back(gl);
					}
				}
			}
		}

		// Align embedded objects to baseline.
		for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = new_sd->objects.front(); E; E = E->next()) {
			if ((E->get().pos >= new_sd->start) && (E->get().pos < new_sd->end)) {
				if (sd->orientation == ORIENTATION_HORIZONTAL) {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.y = -new_sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.y = -(E->get().rect.size.y / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.y = new_sd->descent - E->get().rect.size.y;
						} break;
					}
				} else {
					switch (E->get().inline_align) {
						case VALIGN_TOP: {
							E->get().rect.position.x = -new_sd->ascent;
						} break;
						case VALIGN_CENTER: {
							E->get().rect.position.x = -(E->get().rect.size.x / 2);
						} break;
						case VALIGN_BOTTOM: {
							E->get().rect.position.x = new_sd->descent - E->get().rect.size.x;
						} break;
					}
				}
			}
		}
	}

	new_sd->valid = true;

	return shaped_owner.make_rid(new_sd);
}

RID TextServerAdvanced::shaped_text_get_parent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, RID());
	return sd->parent;
}

float TextServerAdvanced::shaped_text_fit_to_width(RID p_shaped, float p_width, uint8_t /*JustificationFlag*/ p_jst_flags) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->justification_ops_valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_update_justification_ops(p_shaped);
	}

	int start_pos = 0;
	int end_pos = sd->glyphs.size() - 1;

	if ((p_jst_flags & JUSTIFICATION_AFTER_LAST_TAB) == JUSTIFICATION_AFTER_LAST_TAB) {
		int start, end, delta;
		if (sd->para_direction == DIRECTION_LTR) {
			start = sd->glyphs.size() - 1;
			end = -1;
			delta = -1;
		} else {
			start = 0;
			end = sd->glyphs.size();
			delta = +1;
		}

		for (int i = start; i != end; i += delta) {
			if ((sd->glyphs[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
				if (sd->para_direction == DIRECTION_LTR) {
					start_pos = i;
					break;
				} else {
					end_pos = i;
					break;
				}
			}
		}
	}

	if ((p_jst_flags & JUSTIFICATION_TRIM_EDGE_SPACES) == JUSTIFICATION_TRIM_EDGE_SPACES) {
		while ((start_pos < end_pos) && ((sd->glyphs[start_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[start_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[start_pos].advance * sd->glyphs[start_pos].repeat;
			sd->glyphs.write[start_pos].advance = 0;
			start_pos += sd->glyphs[start_pos].count;
		}
		while ((start_pos < end_pos) && ((sd->glyphs[end_pos].flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_HARD) == GRAPHEME_IS_BREAK_HARD || (sd->glyphs[end_pos].flags & GRAPHEME_IS_BREAK_SOFT) == GRAPHEME_IS_BREAK_SOFT)) {
			sd->width -= sd->glyphs[end_pos].advance * sd->glyphs[end_pos].repeat;
			sd->glyphs.write[end_pos].advance = 0;
			end_pos -= sd->glyphs[end_pos].count;
		}
	}

	int space_count = 0;
	int elongation_count = 0;
	for (int i = start_pos; i <= end_pos; i++) {
		const Glyph &gl = sd->glyphs[i];
		if (gl.count > 0) {
			if ((gl.flags & GRAPHEME_IS_ELONGATION) == GRAPHEME_IS_ELONGATION) {
				elongation_count++;
			}
			if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
				space_count++;
			}
		}
	}

	if ((elongation_count > 0) && ((p_jst_flags & JUSTIFICATION_KASHIDA) == JUSTIFICATION_KASHIDA)) {
		float delta_width_per_kashida = (p_width - sd->width) / elongation_count;
		for (int i = start_pos; i <= end_pos; i++) {
			Glyph &gl = sd->glyphs.write[i];
			if (gl.count > 0) {
				if (((gl.flags & GRAPHEME_IS_ELONGATION) == GRAPHEME_IS_ELONGATION) && (gl.advance > 0)) {
					int count = delta_width_per_kashida / gl.advance;
					int prev_count = gl.repeat;
					if ((gl.flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) {
						gl.repeat = count;
					} else {
						gl.repeat = count + 1;
					}
					sd->width += (gl.repeat - prev_count) * gl.advance;
				}
			}
		}
	}

	if ((space_count > 0) && ((p_jst_flags & JUSTIFICATION_WORD_BOUND) == JUSTIFICATION_WORD_BOUND)) {
		float delta_width_per_space = (p_width - sd->width) / space_count;
		for (int i = start_pos; i <= end_pos; i++) {
			Glyph &gl = sd->glyphs.write[i];
			if (gl.count > 0) {
				if ((gl.flags & GRAPHEME_IS_SPACE) == GRAPHEME_IS_SPACE) {
					float old_adv = gl.advance;
					if ((gl.flags & GRAPHEME_IS_VIRTUAL) == GRAPHEME_IS_VIRTUAL) {
						gl.advance = Math::round(MAX(gl.advance + delta_width_per_space, 0.f));
					} else {
						gl.advance = Math::round(MAX(gl.advance + delta_width_per_space, 0.05 * gl.font_size));
					}
					sd->width += (gl.advance - old_adv);
				}
			}
		}
	}

	return sd->width;
}

float TextServerAdvanced::shaped_text_tab_align(RID p_shaped, const Vector<float> &p_tab_stops) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_update_breaks(p_shaped);
	}

	int tab_index = 0;
	float off = 0.f;

	int start, end, delta;
	if (sd->para_direction == DIRECTION_LTR) {
		start = 0;
		end = sd->glyphs.size();
		delta = +1;
	} else {
		start = sd->glyphs.size() - 1;
		end = -1;
		delta = -1;
	}

	Glyph *gl = sd->glyphs.ptrw();

	for (int i = start; i != end; i += delta) {
		if ((gl[i].flags & GRAPHEME_IS_TAB) == GRAPHEME_IS_TAB) {
			float tab_off = 0.f;
			while (tab_off <= off) {
				tab_off += p_tab_stops[tab_index];
				tab_index++;
				if (tab_index >= p_tab_stops.size()) {
					tab_index = 0;
				}
			}
			float old_adv = gl[i].advance;
			gl[i].advance = tab_off - off;
			sd->width += gl[i].advance - old_adv;
			off = 0;
			continue;
		}
		off += gl[i].advance * gl[i].repeat;
	}

	return 0.f;
}

bool TextServerAdvanced::shaped_text_update_breaks(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}

	if (sd->line_breaks_valid) {
		return true; // Nothing to do.
	}

	const UChar *data = sd->utf16.ptr();

	HashMap<int, bool> breaks;
	UErrorCode err = U_ZERO_ERROR;
	int i = 0;
	while (i < sd->spans.size()) {
		String language = sd->spans[i].language;
		int r_start = sd->spans[i].start;
		while (i + 1 < sd->spans.size() && language == sd->spans[i + 1].language) {
			i++;
		}
		int r_end = sd->spans[i].end;
		UBreakIterator *bi = ubrk_open(UBRK_LINE, language.ascii().get_data(), data + _convert_pos_inv(sd, r_start), _convert_pos_inv(sd, r_end - r_start), &err);
		if (U_FAILURE(err)) {
			//No data loaded - use fallback.
			for (int j = r_start; j < r_end; j++) {
				char32_t c = sd->text[j - sd->start];
				if (is_whitespace(c)) {
					breaks[j] = false;
				}
				if (is_linebreak(c)) {
					breaks[j] = true;
				}
			}
		} else {
			while (ubrk_next(bi) != UBRK_DONE) {
				int pos = _convert_pos(sd, ubrk_current(bi)) + r_start - 1;
				if (pos != r_end) {
					if ((ubrk_getRuleStatus(bi) >= UBRK_LINE_HARD) && (ubrk_getRuleStatus(bi) < UBRK_LINE_HARD_LIMIT)) {
						breaks[pos] = true;
					} else if ((ubrk_getRuleStatus(bi) >= UBRK_LINE_SOFT) && (ubrk_getRuleStatus(bi) < UBRK_LINE_SOFT_LIMIT)) {
						breaks[pos] = false;
					}
				}
			}
		}
		ubrk_close(bi);
		i++;
	}

	sd->sort_valid = false;
	sd->glyphs_logical.clear();
	int sd_size = sd->glyphs.size();
	const char32_t *ch = sd->text.ptr();
	Glyph *sd_glyphs = sd->glyphs.ptrw();

	for (i = 0; i < sd_size; i++) {
		if (sd_glyphs[i].count > 0) {
			char32_t c = ch[sd_glyphs[i].start - sd->start];
			if (c == 0xfffc) {
				continue;
			}
			if (c == 0x0009 || c == 0x000b) {
				sd_glyphs[i].flags |= GRAPHEME_IS_TAB;
			}
			if (is_whitespace(c)) {
				sd_glyphs[i].flags |= GRAPHEME_IS_SPACE;
			}
			if (u_ispunct(c)) {
				sd_glyphs[i].flags |= GRAPHEME_IS_PUNCTUATION;
			}
			if (breaks.has(sd->glyphs[i].start)) {
				if (breaks[sd->glyphs[i].start]) {
					sd_glyphs[i].flags |= GRAPHEME_IS_BREAK_HARD;
				} else {
					if (is_whitespace(c)) {
						sd_glyphs[i].flags |= GRAPHEME_IS_BREAK_SOFT;
					} else {
						TextServer::Glyph gl;
						gl.start = sd_glyphs[i].start;
						gl.end = sd_glyphs[i].end;
						gl.count = 1;
						gl.font_rid = sd_glyphs[i].font_rid;
						gl.font_size = sd_glyphs[i].font_size;
						gl.flags = GRAPHEME_IS_BREAK_SOFT | GRAPHEME_IS_VIRTUAL;
						sd->glyphs.insert(i + sd_glyphs[i].count, gl); // insert after

						// Update write pointer and size.
						sd_size = sd->glyphs.size();
						sd_glyphs = sd->glyphs.ptrw();

						i += sd_glyphs[i].count;
						continue;
					}
				}
			}

			i += (sd_glyphs[i].count - 1);
		}
	}

	sd->line_breaks_valid = true;

	return sd->line_breaks_valid;
}

_FORCE_INLINE_ int _generate_kashida_justification_opportunies(const String &p_data, int p_start, int p_end) {
	int kashida_pos = -1;
	int8_t priority = 100;
	int i = p_start;

	char32_t pc = 0;

	while ((p_end > p_start) && is_transparent(p_data[p_end - 1])) {
		p_end--;
	}

	while (i < p_end) {
		uint32_t c = p_data[i];

		if (c == 0x0640) {
			kashida_pos = i;
			priority = 0;
		}
		if (priority >= 1 && i < p_end - 1) {
			if (is_seen_sad(c) && (p_data[i + 1] != 0x200C)) {
				kashida_pos = i;
				priority = 1;
			}
		}
		if (priority >= 2 && i > p_start) {
			if (is_teh_marbuta(c) || is_dal(c) || (is_heh(c) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 2;
				}
			}
		}
		if (priority >= 3 && i > p_start) {
			if (is_alef(c) || ((is_lam(c) || is_tah(c) || is_kaf(c) || is_gaf(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 3;
				}
			}
		}
		if (priority >= 4 && i > p_start && i < p_end - 1) {
			if (is_beh(c)) {
				if (is_reh(p_data[i + 1]) || is_yeh(p_data[i + 1])) {
					if (is_connected_to_prev(c, pc)) {
						kashida_pos = i - 1;
						priority = 4;
					}
				}
			}
		}
		if (priority >= 5 && i > p_start) {
			if (is_waw(c) || ((is_ain(c) || is_qaf(c) || is_feh(c)) && i == p_end - 1)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 5;
				}
			}
		}
		if (priority >= 6 && i > p_start) {
			if (is_reh(c)) {
				if (is_connected_to_prev(c, pc)) {
					kashida_pos = i - 1;
					priority = 6;
				}
			}
		}
		if (!is_transparent(c)) {
			pc = c;
		}
		i++;
	}

	return kashida_pos;
}

bool TextServerAdvanced::shaped_text_update_justification_ops(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	if (!sd->valid) {
		shaped_text_shape(p_shaped);
	}
	if (!sd->line_breaks_valid) {
		shaped_text_update_breaks(p_shaped);
	}

	if (sd->justification_ops_valid) {
		return true; // Nothing to do.
	}

	const UChar *data = sd->utf16.ptr();
	int32_t data_size = sd->utf16.length();

	Map<int, bool> jstops;

	// Use ICU word iterator and custom kashida detection.
	UErrorCode err = U_ZERO_ERROR;
	UBreakIterator *bi = ubrk_open(UBRK_WORD, "", data, data_size, &err);
	if (U_FAILURE(err)) {
		// No data - use fallback
		int limit = 0;
		for (int i = 0; i < sd->text.length(); i++) {
			if (is_whitespace(data[i])) {
				int ks = _generate_kashida_justification_opportunies(sd->text, limit, i) + sd->start;
				if (ks != -1) {
					jstops[ks] = true;
				}
				limit = i + 1;
			}
		}
		int ks = _generate_kashida_justification_opportunies(sd->text, limit, sd->text.length()) + sd->start;
		if (ks != -1) {
			jstops[ks] = true;
		}
	} else {
		int limit = 0;
		while (ubrk_next(bi) != UBRK_DONE) {
			if (ubrk_getRuleStatus(bi) != UBRK_WORD_NONE) {
				int i = _convert_pos(sd, ubrk_current(bi));
				jstops[i + sd->start] = false;
				int ks = _generate_kashida_justification_opportunies(sd->text, limit, i);
				if (ks != -1) {
					jstops[ks + sd->start] = true;
				}
				limit = i;
			}
		}
		ubrk_close(bi);
	}

	sd->sort_valid = false;
	sd->glyphs_logical.clear();
	int sd_size = sd->glyphs.size();

	if (jstops.size() > 0) {
		for (int i = 0; i < sd_size; i++) {
			if (sd->glyphs[i].count > 0) {
				if (jstops.has(sd->glyphs[i].start)) {
					char32_t c = sd->text[sd->glyphs[i].start - sd->start];
					if (c == 0xfffc) {
						continue;
					}
					if (jstops[sd->glyphs[i].start]) {
						if (c == 0x0640) {
							sd->glyphs.write[i].flags |= GRAPHEME_IS_ELONGATION;
						} else {
							if (sd->glyphs[i].font_rid != RID()) {
								TextServer::Glyph gl = _shape_single_glyph(sd, 0x0640, HB_SCRIPT_ARABIC, HB_DIRECTION_RTL, sd->glyphs[i].font_rid, sd->glyphs[i].font_size);
								if ((gl.flags & GRAPHEME_IS_VALID) == GRAPHEME_IS_VALID) {
									gl.start = sd->glyphs[i].start;
									gl.end = sd->glyphs[i].end;
									gl.repeat = 0;
									gl.count = 1;
									if (sd->orientation == ORIENTATION_HORIZONTAL) {
										gl.y_off = sd->glyphs[i].y_off;
									} else {
										gl.x_off = sd->glyphs[i].x_off;
									}
									gl.flags |= GRAPHEME_IS_ELONGATION | GRAPHEME_IS_VIRTUAL;
									sd->glyphs.insert(i, gl);
									i++;
								}
							}
						}
					} else if (!is_whitespace(c)) {
						TextServer::Glyph gl;
						gl.start = sd->glyphs[i].start;
						gl.end = sd->glyphs[i].end;
						gl.count = 1;
						gl.font_rid = sd->glyphs[i].font_rid;
						gl.font_size = sd->glyphs[i].font_size;
						gl.flags = GRAPHEME_IS_SPACE | GRAPHEME_IS_VIRTUAL;
						sd->glyphs.insert(i + sd->glyphs[i].count, gl); // insert after
						i += sd->glyphs[i].count;
						continue;
					}
				}
			}
		}
	}

	sd->justification_ops_valid = true;
	return sd->justification_ops_valid;
}

TextServer::Glyph TextServerAdvanced::_shape_single_glyph(ShapedTextDataAdvanced *p_sd, char32_t p_char, hb_script_t p_script, hb_direction_t p_direction, RID p_font, int p_font_size) {
	FontDataAdvanced *fd = font_owner.getornull(p_font);
	hb_font_t *hb_font = fd->get_hb_handle(p_font_size);

	hb_buffer_clear_contents(p_sd->hb_buffer);
	hb_buffer_set_direction(p_sd->hb_buffer, p_direction);
	hb_buffer_set_flags(p_sd->hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT));
	hb_buffer_set_script(p_sd->hb_buffer, p_script);
	hb_buffer_add_utf32(p_sd->hb_buffer, (const uint32_t *)&p_char, 1, 0, 1);

	hb_shape(hb_font, p_sd->hb_buffer, nullptr, 0);

	unsigned int glyph_count = 0;
	hb_glyph_info_t *glyph_info = hb_buffer_get_glyph_infos(p_sd->hb_buffer, &glyph_count);
	hb_glyph_position_t *glyph_pos = hb_buffer_get_glyph_positions(p_sd->hb_buffer, &glyph_count);

	// Process glyphs.
	TextServer::Glyph gl;

	if (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) {
		gl.flags |= TextServer::GRAPHEME_IS_RTL;
	}

	gl.font_rid = p_font;
	gl.font_size = p_font_size;

	if (glyph_count > 0) {
		if (p_sd->orientation == ORIENTATION_HORIZONTAL) {
			gl.advance = Math::round(glyph_pos[0].x_advance / (64.0 / fd->get_font_scale(p_font_size)));
		} else {
			gl.advance = -Math::round(glyph_pos[0].y_advance / (64.0 / fd->get_font_scale(p_font_size)));
		}
		gl.count = 1;

		gl.index = glyph_info[0].codepoint;
		gl.x_off = Math::round(glyph_pos[0].x_offset / (64.0 / fd->get_font_scale(p_font_size)));
		gl.y_off = -Math::round(glyph_pos[0].y_offset / (64.0 / fd->get_font_scale(p_font_size)));

		if ((glyph_info[0].codepoint != 0) || !u_isgraph(p_char)) {
			gl.flags |= GRAPHEME_IS_VALID;
		}
	}
	return gl;
}

void TextServerAdvanced::_shape_run(ShapedTextDataAdvanced *p_sd, int32_t p_start, int32_t p_end, hb_script_t p_script, hb_direction_t p_direction, Vector<RID> p_fonts, int p_span, int p_fb_index) {
	FontDataAdvanced *fd = nullptr;
	if (p_fb_index < p_fonts.size()) {
		fd = font_owner.getornull(p_fonts[p_fb_index]);
	}

	int fs = p_sd->spans[p_span].font_size;
	if (fd == nullptr) {
		// Add fallback glyphs
		for (int i = p_start; i < p_end; i++) {
			if (p_sd->preserve_invalid || (p_sd->preserve_control && is_control(p_sd->text[i]))) {
				TextServer::Glyph gl;
				gl.start = i;
				gl.end = i + 1;
				gl.count = 1;
				gl.index = p_sd->text[i];
				gl.font_size = fs;
				gl.font_rid = RID();
				if (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) {
					gl.flags |= TextServer::GRAPHEME_IS_RTL;
				}
				if (p_sd->orientation == ORIENTATION_HORIZONTAL) {
					gl.advance = get_hex_code_box_size(fs, gl.index).x;
					p_sd->ascent = MAX(p_sd->ascent, Math::round(get_hex_code_box_size(fs, gl.index).y * 0.75f));
					p_sd->descent = MAX(p_sd->descent, Math::round(get_hex_code_box_size(fs, gl.index).y * 0.25f));
				} else {
					gl.advance = get_hex_code_box_size(fs, gl.index).y;
					p_sd->ascent = MAX(p_sd->ascent, Math::round(get_hex_code_box_size(fs, gl.index).x * 0.5f));
					p_sd->descent = MAX(p_sd->descent, Math::round(get_hex_code_box_size(fs, gl.index).x * 0.5f));
				}
				p_sd->width += gl.advance;

				p_sd->glyphs.push_back(gl);
			}
		}
		return;
	}

	hb_font_t *hb_font = fd->get_hb_handle(fs);
	ERR_FAIL_COND(hb_font == nullptr);

	hb_buffer_clear_contents(p_sd->hb_buffer);
	hb_buffer_set_direction(p_sd->hb_buffer, p_direction);
	if (p_sd->preserve_control) {
		hb_buffer_set_flags(p_sd->hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_PRESERVE_DEFAULT_IGNORABLES | (p_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_end == p_sd->text.length() ? HB_BUFFER_FLAG_EOT : 0)));
	} else {
		hb_buffer_set_flags(p_sd->hb_buffer, (hb_buffer_flags_t)(HB_BUFFER_FLAG_DEFAULT | (p_start == 0 ? HB_BUFFER_FLAG_BOT : 0) | (p_end == p_sd->text.length() ? HB_BUFFER_FLAG_EOT : 0)));
	}
	hb_buffer_set_script(p_sd->hb_buffer, p_script);

	if (p_sd->spans[p_span].language != String()) {
		hb_language_t lang = hb_language_from_string(p_sd->spans[p_span].language.ascii().get_data(), -1);
		hb_buffer_set_language(p_sd->hb_buffer, lang);
	}

	hb_buffer_add_utf32(p_sd->hb_buffer, (const uint32_t *)p_sd->text.ptr(), p_sd->text.length(), p_start, p_end - p_start);

	Vector<hb_feature_t> ftrs;
	for (const Variant *ftr = p_sd->spans[p_span].features.next(nullptr); ftr != nullptr; ftr = p_sd->spans[p_span].features.next(ftr)) {
		double values = p_sd->spans[p_span].features[*ftr];
		if (values >= 0) {
			hb_feature_t feature;
			feature.tag = *ftr;
			feature.value = values;
			feature.start = 0;
			feature.end = -1;
			ftrs.push_back(feature);
		}
	}
	hb_shape(hb_font, p_sd->hb_buffer, ftrs.is_empty() ? nullptr : &ftrs[0], ftrs.size());

	unsigned int glyph_count = 0;
	hb_glyph_info_t *glyph_info = hb_buffer_get_glyph_infos(p_sd->hb_buffer, &glyph_count);
	hb_glyph_position_t *glyph_pos = hb_buffer_get_glyph_positions(p_sd->hb_buffer, &glyph_count);

	// Process glyphs.
	if (glyph_count > 0) {
		TextServer::Glyph *w = (TextServer::Glyph *)memalloc(glyph_count * sizeof(TextServer::Glyph));

		int end = (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) ? p_end : 0;
		uint32_t last_cluster_id = UINT32_MAX;
		unsigned int last_cluster_index = 0;
		bool last_cluster_valid = true;

		for (unsigned int i = 0; i < glyph_count; i++) {
			if ((i > 0) && (last_cluster_id != glyph_info[i].cluster)) {
				if (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) {
					end = w[last_cluster_index].start;
				} else {
					for (unsigned int j = last_cluster_index; j < i; j++) {
						w[j].end = glyph_info[i].cluster;
					}
				}
				if (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) {
					w[last_cluster_index].flags |= TextServer::GRAPHEME_IS_RTL;
				}
				if (last_cluster_valid) {
					w[last_cluster_index].flags |= GRAPHEME_IS_VALID;
				}
				w[last_cluster_index].count = i - last_cluster_index;
				last_cluster_index = i;
				last_cluster_valid = true;
			}

			last_cluster_id = glyph_info[i].cluster;

			TextServer::Glyph &gl = w[i];
			gl = TextServer::Glyph();

			gl.start = glyph_info[i].cluster;
			gl.end = end;
			gl.count = 0;

			gl.font_rid = p_fonts[p_fb_index];
			gl.font_size = fs;

			gl.index = glyph_info[i].codepoint;
			if (gl.index != 0) {
				if (p_sd->orientation == ORIENTATION_HORIZONTAL) {
					gl.advance = Math::round(glyph_pos[i].x_advance / (64.0 / fd->get_font_scale(fs)));
				} else {
					gl.advance = -Math::round(glyph_pos[i].y_advance / (64.0 / fd->get_font_scale(fs)));
				}
				gl.x_off = Math::round(glyph_pos[i].x_offset / (64.0 / fd->get_font_scale(fs)));
				gl.y_off = -Math::round(glyph_pos[i].y_offset / (64.0 / fd->get_font_scale(fs)));
			}
			if (fd->get_spacing_space() && is_whitespace(p_sd->text[glyph_info[i].cluster])) {
				gl.advance += fd->get_spacing_space();
			} else {
				gl.advance += fd->get_spacing_glyph();
			}

			if (p_sd->preserve_control) {
				last_cluster_valid = last_cluster_valid && ((glyph_info[i].codepoint != 0) || is_whitespace(p_sd->text[glyph_info[i].cluster]) || is_linebreak(p_sd->text[glyph_info[i].cluster]));
			} else {
				last_cluster_valid = last_cluster_valid && ((glyph_info[i].codepoint != 0) || !u_isgraph(p_sd->text[glyph_info[i].cluster]));
			}
		}
		if (p_direction == HB_DIRECTION_LTR || p_direction == HB_DIRECTION_TTB) {
			for (unsigned int j = last_cluster_index; j < glyph_count; j++) {
				w[j].end = p_end;
			}
		}
		w[last_cluster_index].count = glyph_count - last_cluster_index;
		if (p_direction == HB_DIRECTION_RTL || p_direction == HB_DIRECTION_BTT) {
			w[last_cluster_index].flags |= TextServer::GRAPHEME_IS_RTL;
		}
		if (last_cluster_valid) {
			w[last_cluster_index].flags |= GRAPHEME_IS_VALID;
		}

		//Fallback.
		int failed_subrun_start = p_end + 1;
		int failed_subrun_end = p_start;

		for (unsigned int i = 0; i < glyph_count; i++) {
			if ((w[i].flags & GRAPHEME_IS_VALID) == GRAPHEME_IS_VALID) {
				if (failed_subrun_start != p_end + 1) {
					_shape_run(p_sd, failed_subrun_start, failed_subrun_end, p_script, p_direction, p_fonts, p_span, p_fb_index + 1);
					failed_subrun_start = p_end + 1;
					failed_subrun_end = p_start;
				}
				for (int j = 0; j < w[i].count; j++) {
					if (p_sd->orientation == ORIENTATION_HORIZONTAL) {
						p_sd->ascent = MAX(p_sd->ascent, -w[i + j].y_off);
						p_sd->descent = MAX(p_sd->descent, w[i + j].y_off);
					} else {
						p_sd->ascent = MAX(p_sd->ascent, Math::round(fd->get_advance(w[i + j].index, fs).x * 0.5));
						p_sd->descent = MAX(p_sd->descent, Math::round(fd->get_advance(w[i + j].index, fs).x * 0.5));
					}
					p_sd->width += w[i + j].advance;
					p_sd->glyphs.push_back(w[i + j]);
				}
			} else {
				if (failed_subrun_start >= w[i].start) {
					failed_subrun_start = w[i].start;
				}
				if (failed_subrun_end <= w[i].end) {
					failed_subrun_end = w[i].end;
				}
			}
			i += w[i].count - 1;
		}
		memfree(w);
		if (failed_subrun_start != p_end + 1) {
			_shape_run(p_sd, failed_subrun_start, failed_subrun_end, p_script, p_direction, p_fonts, p_span, p_fb_index + 1);
		}
		p_sd->ascent = MAX(p_sd->ascent, fd->get_ascent(fs));
		p_sd->descent = MAX(p_sd->descent, fd->get_descent(fs));
		p_sd->upos = MAX(p_sd->upos, fd->get_underline_position(fs));
		p_sd->uthk = MAX(p_sd->uthk, fd->get_underline_thickness(fs));
	}
}

bool TextServerAdvanced::shaped_text_shape(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);

	if (sd->valid) {
		return true;
	}

	if (sd->parent != RID()) {
		full_copy(sd);
	}
	invalidate(sd);

	if (sd->text.length() == 0) {
		sd->valid = true;
		return true;
	}

	sd->utf16 = sd->text.utf16();
	const UChar *data = sd->utf16.ptr();

	// Create script iterator.
	if (sd->script_iter == nullptr) {
		sd->script_iter = memnew(ScriptIterator(sd->text, 0, sd->text.length()));
	}

	if (sd->bidi_override.is_empty()) {
		sd->bidi_override.push_back(Vector2i(0, sd->end));
	}

	for (int ov = 0; ov < sd->bidi_override.size(); ov++) {
		// Create BiDi iterator.
		int start = _convert_pos_inv(sd, sd->bidi_override[ov].x);
		int end = _convert_pos_inv(sd, sd->bidi_override[ov].y);

		ERR_FAIL_COND_V_MSG((start < 0 || end - start > sd->utf16.length()), false, "Invalid BiDi override range.");

		UErrorCode err = U_ZERO_ERROR;
		UBiDi *bidi_iter = ubidi_openSized(end, 0, &err);
		ERR_FAIL_COND_V_MSG(U_FAILURE(err), false, u_errorName(err));

		switch (sd->direction) {
			case DIRECTION_LTR: {
				ubidi_setPara(bidi_iter, data + start, end - start, UBIDI_LTR, nullptr, &err);
				sd->para_direction = DIRECTION_LTR;
			} break;
			case DIRECTION_RTL: {
				ubidi_setPara(bidi_iter, data + start, end - start, UBIDI_RTL, nullptr, &err);
				sd->para_direction = DIRECTION_RTL;
			} break;
			case DIRECTION_AUTO: {
				UBiDiDirection direction = ubidi_getBaseDirection(data + start, end - start);
				if (direction != UBIDI_NEUTRAL) {
					ubidi_setPara(bidi_iter, data + start, end - start, direction, nullptr, &err);
					sd->para_direction = (direction == UBIDI_RTL) ? DIRECTION_RTL : DIRECTION_LTR;
				} else {
					ubidi_setPara(bidi_iter, data + start, end - start, UBIDI_DEFAULT_LTR, nullptr, &err);
					sd->para_direction = DIRECTION_LTR;
				}
			} break;
		}
		ERR_FAIL_COND_V_MSG(U_FAILURE(err), false, u_errorName(err));
		sd->bidi_iter.push_back(bidi_iter);

		err = U_ZERO_ERROR;
		int bidi_run_count = ubidi_countRuns(bidi_iter, &err);
		ERR_FAIL_COND_V_MSG(U_FAILURE(err), false, u_errorName(err));
		for (int i = 0; i < bidi_run_count; i++) {
			int32_t _bidi_run_start = 0;
			int32_t _bidi_run_length = 0;
			hb_direction_t bidi_run_direction = HB_DIRECTION_INVALID;
			bool is_rtl = (ubidi_getVisualRun(bidi_iter, i, &_bidi_run_start, &_bidi_run_length) == UBIDI_LTR);
			switch (sd->orientation) {
				case ORIENTATION_HORIZONTAL: {
					if (is_rtl) {
						bidi_run_direction = HB_DIRECTION_LTR;
					} else {
						bidi_run_direction = HB_DIRECTION_RTL;
					}
				} break;
				case ORIENTATION_VERTICAL: {
					if (is_rtl) {
						bidi_run_direction = HB_DIRECTION_TTB;
					} else {
						bidi_run_direction = HB_DIRECTION_BTT;
					}
				}
			}

			int32_t bidi_run_start = _convert_pos(sd, sd->bidi_override[ov].x + _bidi_run_start);
			int32_t bidi_run_end = _convert_pos(sd, sd->bidi_override[ov].x + _bidi_run_start + _bidi_run_length);

			// Shape runs.

			int scr_from = (is_rtl) ? 0 : sd->script_iter->script_ranges.size() - 1;
			int scr_to = (is_rtl) ? sd->script_iter->script_ranges.size() : -1;
			int scr_delta = (is_rtl) ? +1 : -1;

			for (int j = scr_from; j != scr_to; j += scr_delta) {
				if ((sd->script_iter->script_ranges[j].start < bidi_run_end) && (sd->script_iter->script_ranges[j].end > bidi_run_start)) {
					int32_t script_run_start = MAX(sd->script_iter->script_ranges[j].start, bidi_run_start);
					int32_t script_run_end = MIN(sd->script_iter->script_ranges[j].end, bidi_run_end);
					char scr_buffer[5] = { 0, 0, 0, 0, 0 };
					hb_tag_to_string(hb_script_to_iso15924_tag(sd->script_iter->script_ranges[j].script), scr_buffer);
					String script = String(scr_buffer);

					int spn_from = (is_rtl) ? 0 : sd->spans.size() - 1;
					int spn_to = (is_rtl) ? sd->spans.size() : -1;
					int spn_delta = (is_rtl) ? +1 : -1;

					for (int k = spn_from; k != spn_to; k += spn_delta) {
						const ShapedTextDataAdvanced::Span &span = sd->spans[k];
						if (span.start >= script_run_end || span.end <= script_run_start) {
							continue;
						}
						if (span.embedded_key != Variant()) {
							// Embedded object.
							if (sd->orientation == ORIENTATION_HORIZONTAL) {
								sd->objects[span.embedded_key].rect.position.x = sd->width;
								sd->width += sd->objects[span.embedded_key].rect.size.x;
								switch (sd->objects[span.embedded_key].inline_align) {
									case VALIGN_TOP: {
										sd->ascent = MAX(sd->ascent, sd->objects[span.embedded_key].rect.size.y);
									} break;
									case VALIGN_CENTER: {
										sd->ascent = MAX(sd->ascent, Math::round(sd->objects[span.embedded_key].rect.size.y / 2));
										sd->descent = MAX(sd->descent, Math::round(sd->objects[span.embedded_key].rect.size.y / 2));
									} break;
									case VALIGN_BOTTOM: {
										sd->descent = MAX(sd->descent, sd->objects[span.embedded_key].rect.size.y);
									} break;
								}
							} else {
								sd->objects[span.embedded_key].rect.position.y = sd->width;
								sd->width += sd->objects[span.embedded_key].rect.size.y;
								switch (sd->objects[span.embedded_key].inline_align) {
									case VALIGN_TOP: {
										sd->ascent = MAX(sd->ascent, sd->objects[span.embedded_key].rect.size.x);
									} break;
									case VALIGN_CENTER: {
										sd->ascent = MAX(sd->ascent, Math::round(sd->objects[span.embedded_key].rect.size.x / 2));
										sd->descent = MAX(sd->descent, Math::round(sd->objects[span.embedded_key].rect.size.x / 2));
									} break;
									case VALIGN_BOTTOM: {
										sd->descent = MAX(sd->descent, sd->objects[span.embedded_key].rect.size.x);
									} break;
								}
							}
							Glyph gl;
							gl.start = span.start;
							gl.end = span.end;
							gl.count = 1;
							gl.flags = GRAPHEME_IS_VALID | GRAPHEME_IS_VIRTUAL;
							if (sd->orientation == ORIENTATION_HORIZONTAL) {
								gl.advance = sd->objects[span.embedded_key].rect.size.x;
							} else {
								gl.advance = sd->objects[span.embedded_key].rect.size.y;
							}
							sd->glyphs.push_back(gl);
						} else {
							Vector<RID> fonts;
							// Push fonts with the language and script support first.
							for (int l = 0; l < span.fonts.size(); l++) {
								if ((font_is_language_supported(span.fonts[l], span.language)) && (font_is_script_supported(span.fonts[l], script))) {
									fonts.push_back(sd->spans[k].fonts[l]);
								}
							}
							// Push fonts with the script support.
							for (int l = 0; l < sd->spans[k].fonts.size(); l++) {
								if (!(font_is_language_supported(span.fonts[l], span.language)) && (font_is_script_supported(span.fonts[l], script))) {
									fonts.push_back(sd->spans[k].fonts[l]);
								}
							}
							// Push the rest valid fonts.
							for (int l = 0; l < sd->spans[k].fonts.size(); l++) {
								if (!(font_is_language_supported(span.fonts[l], span.language)) && !(font_is_script_supported(span.fonts[l], script))) {
									fonts.push_back(sd->spans[k].fonts[l]);
								}
							}
							_shape_run(sd, MAX(sd->spans[k].start, script_run_start), MIN(sd->spans[k].end, script_run_end), sd->script_iter->script_ranges[j].script, bidi_run_direction, fonts, k, 0);
						}
					}
				}
			}
		}
	}

	// Align embedded objects to baseline.
	for (Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
		if (sd->orientation == ORIENTATION_HORIZONTAL) {
			switch (E->get().inline_align) {
				case VALIGN_TOP: {
					E->get().rect.position.y = -sd->ascent;
				} break;
				case VALIGN_CENTER: {
					E->get().rect.position.y = -(E->get().rect.size.y / 2);
				} break;
				case VALIGN_BOTTOM: {
					E->get().rect.position.y = sd->descent - E->get().rect.size.y;
				} break;
			}
		} else {
			switch (E->get().inline_align) {
				case VALIGN_TOP: {
					E->get().rect.position.x = -sd->ascent;
				} break;
				case VALIGN_CENTER: {
					E->get().rect.position.x = -(E->get().rect.size.x / 2);
				} break;
				case VALIGN_BOTTOM: {
					E->get().rect.position.x = sd->descent - E->get().rect.size.x;
				} break;
			}
		}
	}

	sd->valid = true;
	return sd->valid;
}

bool TextServerAdvanced::shaped_text_is_ready(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, false);
	return sd->valid;
}

Vector<TextServer::Glyph> TextServerAdvanced::shaped_text_get_glyphs(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector<TextServer::Glyph>());
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->glyphs;
}

Vector2i TextServerAdvanced::shaped_text_get_range(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector2i());
	return Vector2(sd->start, sd->end);
}

Vector<TextServer::Glyph> TextServerAdvanced::shaped_text_sort_logical(RID p_shaped) {
	_THREAD_SAFE_METHOD_
	ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Vector<TextServer::Glyph>());
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}

	if (!sd->sort_valid) {
		sd->glyphs_logical = sd->glyphs;
		sd->glyphs_logical.sort_custom<TextServer::GlyphCompare>();
		sd->sort_valid = true;
	}

	return sd->glyphs_logical;
}

Array TextServerAdvanced::shaped_text_get_objects(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	Array ret;
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, ret);
	for (const Map<Variant, ShapedTextData::EmbeddedObject>::Element *E = sd->objects.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}

	return ret;
}

Rect2 TextServerAdvanced::shaped_text_get_object_rect(RID p_shaped, Variant p_key) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Rect2());
	ERR_FAIL_COND_V(!sd->objects.has(p_key), Rect2());
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->objects[p_key].rect;
}

Size2 TextServerAdvanced::shaped_text_get_size(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, Size2());
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	if (sd->orientation == TextServer::ORIENTATION_HORIZONTAL) {
		return Size2(sd->width, sd->ascent + sd->descent);
	} else {
		return Size2(sd->ascent + sd->descent, sd->width);
	}
}

float TextServerAdvanced::shaped_text_get_ascent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->ascent;
}

float TextServerAdvanced::shaped_text_get_descent(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->descent;
}

float TextServerAdvanced::shaped_text_get_width(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}
	return sd->width;
}

float TextServerAdvanced::shaped_text_get_underline_position(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->upos;
}

float TextServerAdvanced::shaped_text_get_underline_thickness(RID p_shaped) const {
	_THREAD_SAFE_METHOD_
	const ShapedTextDataAdvanced *sd = shaped_owner.getornull(p_shaped);
	ERR_FAIL_COND_V(!sd, 0.f);
	if (!sd->valid) {
		const_cast<TextServerAdvanced *>(this)->shaped_text_shape(p_shaped);
	}

	return sd->uthk;
}

struct num_system_data {
	String lang;
	String digits;
	String percent_sign;
	String exp;
};

static num_system_data num_systems[]{
	{ "ar,ar_AR,ar_BH,ar_DJ,ar_EG,ar_ER,ar_IL,ar_IQ,ar_JO,ar_KM,ar_KW,ar_LB,ar_MR,ar_OM,ar_PS,ar_QA,ar_SA,ar_SD,ar_SO,ar_SS,ar_SY,ar_TD,ar_YE", U"٠١٢٣٤٥٦٧٨٩٫", U"٪", U"اس" },
	{ "fa,ks,pa_Arab,ps,ug,ur_IN,ur,uz_Arab", U"۰۱۲۳۴۵۶۷۸۹٫", U"٪", U"اس" },
	{ "as,bn,mni", U"০১২৩৪৫৬৭৮৯.", U"%", U"e" },
	{ "mr,ne", U"०१२३४५६७८९.", U"%", U"e" },
	{ "dz", U"༠༡༢༣༤༥༦༧༨༩.", U"%", U"e" },
	{ "sat", U"᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙.", U"%", U"e" },
	{ "my", U"၀၁၂၃၄၅၆၇၈၉.", U"%", U"e" },
	{ String(), String(), String(), String() },
};

String TextServerAdvanced::format_number(const String &p_string, const String &p_language) const {
	_THREAD_SAFE_METHOD_
	String lang = (p_language == "") ? TranslationServer::get_singleton()->get_tool_locale() : p_language;

	String res = p_string;
	for (int i = 0; num_systems[i].lang != String(); i++) {
		Vector<String> langs = num_systems[i].lang.split(",");
		if (langs.has(lang)) {
			if (num_systems[i].digits == String()) {
				return p_string;
			}
			res.replace("e", num_systems[i].exp);
			res.replace("E", num_systems[i].exp);
			char32_t *data = res.ptrw();
			for (int j = 0; j < res.size(); j++) {
				if (data[j] >= 0x30 && data[j] <= 0x39) {
					data[j] = num_systems[i].digits[data[j] - 0x30];
				} else if (data[j] == '.' || data[j] == ',') {
					data[j] = num_systems[i].digits[10];
				}
			}
			break;
		}
	}
	return res;
}

String TextServerAdvanced::parse_number(const String &p_string, const String &p_language) const {
	_THREAD_SAFE_METHOD_
	String lang = (p_language == "") ? TranslationServer::get_singleton()->get_tool_locale() : p_language;

	String res = p_string;
	for (int i = 0; num_systems[i].lang != String(); i++) {
		Vector<String> langs = num_systems[i].lang.split(",");
		if (langs.has(lang)) {
			if (num_systems[i].digits == String()) {
				return p_string;
			}
			res.replace(num_systems[i].exp, "e");
			char32_t *data = res.ptrw();
			for (int j = 0; j < res.size(); j++) {
				if (data[j] == num_systems[i].digits[10]) {
					data[j] = '.';
				} else {
					for (int k = 0; k < 10; k++) {
						if (data[j] == num_systems[i].digits[k]) {
							data[j] = 0x30 + k;
						}
					}
				}
			}
			break;
		}
	}
	return res;
}

String TextServerAdvanced::percent_sign(const String &p_language) const {
	_THREAD_SAFE_METHOD_
	String lang = (p_language == "") ? TranslationServer::get_singleton()->get_tool_locale() : p_language;

	for (int i = 0; num_systems[i].lang != String(); i++) {
		Vector<String> langs = num_systems[i].lang.split(",");
		if (langs.has(lang)) {
			if (num_systems[i].percent_sign == String()) {
				return "%";
			}
			return num_systems[i].percent_sign;
		}
	}
	return "%";
}

TextServer *TextServerAdvanced::create_func(Error &r_error, void *p_user_data) {
	r_error = OK;
	return memnew(TextServerAdvanced());
}

void TextServerAdvanced::register_server() {
	TextServerManager::register_create_function(interface_name, interface_features, create_func, nullptr);
}

TextServerAdvanced::TextServerAdvanced() {
	hb_bmp_create_font_funcs();
}

TextServerAdvanced::~TextServerAdvanced() {
	hb_bmp_free_font_funcs();
	u_cleanup();
#ifndef ICU_STATIC_DATA
	if (icu_data != nullptr) {
		memfree(icu_data);
		icu_data = nullptr;
	}
#endif
}
