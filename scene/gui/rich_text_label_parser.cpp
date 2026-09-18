/**************************************************************************/
/*  rich_text_label_parser.cpp                                            */
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

#include "rich_text_label_parser.h"

#include "scene/gui/rich_text_effect.h"

const Vector<String> parameterless_tags = {
	"b",
	"i",
	"u",
	"s",
	"code",
	"center",
	"left",
	"right",
	"fill",
	"indent",
};

const Vector<String> self_closing_tags = {
	"br",
	"lb",
	"rb",
	"lrm",
	"rlm",
	"rle",
	"lro",
	"rlo",
	"pdf",
	"alm",
	"lri",
	"rli",
	"fsi",
	"pdi",
	"zwj",
	"zwnj",
	"wj",
	"shy",
};

const Vector<String> main_parameter_tags = {
	"char",
	"color",
	"url",
	"hint",
	"font_size",
	"opentype_features",
	"lang",
	"bgcolor",
	"fgcolor",
	"outline_size",
	"outline_color",
	"table",
};

const Vector<String> complex_tags = {
	"p",
	"img",
	"font",
	"dropcap",
	"cell",
	"ul",
	"ol",
	"pulse",
	"fade",
	"shake",
	"wave",
	"tornado",
	"rainbow",
};

static Vector<String> make_all_known_tags() {
	Vector<String> tags;
	tags.append_array(parameterless_tags);
	tags.append_array(self_closing_tags);
	tags.append_array(main_parameter_tags);
	tags.append_array(complex_tags);
	return tags;
}

const Vector<String> all_known_tags = make_all_known_tags();

Dictionary RichTextLabelBBCodeParser::_validate_tag(const String &p_tag, const Dictionary &p_parameters) const {
	Dictionary result;
	Error err = OK;

	if (parameterless_tags.has(p_tag)) {
		if (!p_parameters.is_empty()) {
			err = ERR_INVALID_PARAMETER;
		}
	} else if (self_closing_tags.has(p_tag)) {
		if (!p_parameters.is_empty()) {
			err = ERR_INVALID_PARAMETER;
		}
		result["self_closing"] = true;
	} else if (main_parameter_tags.has(p_tag)) {
		// Main parameter is required for every tag except URL.
		if (p_parameters.is_empty() && p_tag != "url") {
			err = ERR_INVALID_PARAMETER;
		}
		// Too many parameters.
		if (p_parameters.size() > 1) {
			err = ERR_INVALID_PARAMETER;
		}
		if (!p_parameters.is_empty()) {
			// Parameter may not be name=value syntax.
			const String &key = p_parameters.get_key_at_index(0);
			const Variant &value = p_parameters[key];
			if (!value.is_null()) {
				err = ERR_INVALID_PARAMETER;
			}
		}
		// TODO: data validation, e.g. check for valid (fg|bg|outline)color, font_size being a number, etc.
	} else if (complex_tags.has(p_tag)) {
		// TODO: parameter syntax & data validation
	} else {
		bool valid = false;
		for (int i = 0; i < custom_effects.size(); i++) {
			Ref<RichTextEffect> effect = custom_effects[i];
			if (effect.is_null()) {
				continue;
			}

			if (effect->get_bbcode() == p_tag) {
				valid = true;
				break;
			}
		}

		if (!valid) {
			// Unknown tag.
			err = ERR_DOES_NOT_EXIST;
		}
	}

	result.set("error", err);
	return result;
}

void RichTextLabelBBCodeParser::add_custom_effect(const Ref<RichTextEffect> &p_effect) {
	ERR_FAIL_COND_MSG(p_effect.is_null(), "Invalid RichTextEffect resource.");
	custom_effects.append(p_effect);
}

TypedArray<RichTextEffect> RichTextLabelBBCodeParser::get_custom_effects() {
	return custom_effects;
}

void RichTextLabelBBCodeParser::set_custom_effects(const TypedArray<RichTextEffect> &p_effects) {
	custom_effects = p_effects;
}

void RichTextLabelBBCodeParser::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_custom_effect", "effect"), &RichTextLabelBBCodeParser::add_custom_effect);
	ClassDB::bind_method(D_METHOD("get_custom_effects"), &RichTextLabelBBCodeParser::get_custom_effects);
	ClassDB::bind_method(D_METHOD("set_custom_effects", "effects"), &RichTextLabelBBCodeParser::set_custom_effects);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "custom_effects", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("RichTextEffect")), "set_custom_effects", "get_custom_effects");
}

RichTextLabelBBCodeParser::RichTextLabelBBCodeParser() {
	set_escape_brackets(ESCAPE_BRACKETS_ABBREVIATION);

	// For backwards-compatibility:
	set_backslash_escape_quotes(false);
	set_error_handling(ERROR_HANDLING_PARSE_AS_TEXT);
}
