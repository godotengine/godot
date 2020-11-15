/*************************************************************************/
/*  color_uppercase_name_transformation.inc                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

// Names from https://en.wikipedia.org/wiki/X11_color_names
// and https://docs.microsoft.com/en-us/dotnet/api/system.windows.media.colors?view=netcore-3.1
#include "core/string/ustring.h"
#include "core/templates/vector.h"

static Vector<String> _common_colors;
static Vector<String> _common_color_adjectives;

static void _populate_uppercase_token_lists() {
	if (_common_colors.empty()) {
		_common_colors.append("AQUAMARINE");
		_common_colors.append("BLACK");
		_common_colors.append("BLUE");
		_common_colors.append("BROWN");
		_common_colors.append("CORAL");
		_common_colors.append("CYAN");
		_common_colors.append("GRAY");
		_common_colors.append("GREEN");
		_common_colors.append("GOLDENROD");
		_common_colors.append("KHAKI");
		_common_colors.append("MAROON");
		_common_colors.append("ORANGE");
		_common_colors.append("ORCHID");
		_common_colors.append("PINK");
		_common_colors.append("PURPLE");
		_common_colors.append("RED");
		_common_colors.append("TURQUOISE");
		_common_colors.append("WHITE");
		_common_colors.append("YELLOW");
	}

	if (_common_color_adjectives.empty()) {
		_common_color_adjectives.append("DARK");
		_common_color_adjectives.append("MEDIUM");
		_common_color_adjectives.append("LIGHT");
		_common_color_adjectives.append("PALE");
		_common_color_adjectives.append("SEA");
		_common_color_adjectives.append("SKY");
		_common_color_adjectives.append("SLATE");
		_common_color_adjectives.append("SPRING");
		_common_color_adjectives.append("STEEL");
		_common_color_adjectives.append("WEB");
	}
}

static String _insert_underscores(const String &name, const Vector<String> &tokens) {
	String name_with_underscores(name);

	for (int i = 0; i < tokens.size(); ++i) {
		const String &token = tokens[i];

		// Skip the following steps if the word is the same as the whole name
		if (token == name_with_underscores) {
			continue;
		}

		int begin = name_with_underscores.find(token, 0);

		// Skip the following steps if the word isn't found in the name
		if (begin < 0) {
			continue;
		}

		int end = begin + token.length() - 1;

		// If the word isn't at the beginning
		if (begin > 0) {
			// If the letter before it isn't already an underscore
			if (name_with_underscores[begin - 1] != '_') {
				name_with_underscores = name_with_underscores.insert(begin, "_");
			}
		}

		// If the word isn't at the end
		if (end < name_with_underscores.length() - 2) {
			// If the letter after it isn't already an underscore
			if (name_with_underscores[end + 1] != '_') {
				name_with_underscores = name_with_underscores.insert(end + 1, "_");
			}
		}
	}

	return name_with_underscores;
}

static String _to_uppercase_color_name(const String &name) {
	// Edge cases where the color names and their respective adjectives don't appear in the cached lists.
	// These names are unique enough to warrant not caching them in the lists.
	if (name == "blanchedalmond") {
		return "BLANCHED_ALMOND";
	} else if (name == "burlywood") {
		return "BURLY_WOOD";
	} else if (name == "deepskyblue") {
		// This case behaves weirdly unless kept here since "DEEP" is not in the cached lists.
		return "DEEP_SKY_BLUE";
	} else if (name == "lavenderblush") {
		return "LAVENDER_BLUSH";
	} else if (name == "lemonchiffon") {
		return "LEMON_CHIFFON";
	} else if (name == "mintcream") {
		return "MINT_CREAM";
	} else if (name == "mistyrose") {
		return "MISTY_ROSE";
	} else if (name == "oldlace") {
		return "OLD_LACE";
	} else if (name == "olivedrab") {
		return "OLIVE_DRAB";
	} else if (name == "papayawhip") {
		return "PAPAYA_WHIP";
	} else if (name == "peachpuff") {
		return "PEACH_PUFF";
	} else if (name == "seashell") {
		return "SEA_SHELL";
	}

	// All other cases
	_populate_uppercase_token_lists();
	String uppercase_name = name.to_upper();
	uppercase_name = _insert_underscores(uppercase_name, _common_colors);
	uppercase_name = _insert_underscores(uppercase_name, _common_color_adjectives);
	return uppercase_name;
}
