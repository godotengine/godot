// Names from https://en.wikipedia.org/wiki/X11_color_names
// and https://docs.microsoft.com/en-us/dotnet/api/system.windows.media.colors?view=netcore-3.1
#include "core/ustring.h"
#include "core/vector.h"

static Vector<String> _modifiable_colors;
static Vector<String> _color_descriptors;
static void _populate_uppercase_token_lists() {
	if (_modifiable_colors.empty()) {
		_modifiable_colors.append("AQUAMARINE");
		_modifiable_colors.append("BLACK");
		_modifiable_colors.append("BLUE");
		_modifiable_colors.append("BROWN");
		_modifiable_colors.append("CORAL");
		_modifiable_colors.append("CYAN");
		_modifiable_colors.append("GRAY");
		_modifiable_colors.append("GREEN");
		_modifiable_colors.append("GOLDENROD");
		_modifiable_colors.append("KHAKI");
		_modifiable_colors.append("MAROON");
		_modifiable_colors.append("ORANGE");
		_modifiable_colors.append("ORCHID");
		_modifiable_colors.append("PINK");
		_modifiable_colors.append("PURPLE");
		_modifiable_colors.append("RED");
		_modifiable_colors.append("TURQUOISE");
		_modifiable_colors.append("WHITE");
		_modifiable_colors.append("YELLOW");
	}

	if (_color_descriptors.empty()) {
		_color_descriptors.append("DARK");
		_color_descriptors.append("MEDIUM");
		_color_descriptors.append("LIGHT");
		_color_descriptors.append("OLIVE");
		_color_descriptors.append("PALE");
		_color_descriptors.append("SEA");
		_color_descriptors.append("SKY");
		_color_descriptors.append("SLATE");
		_color_descriptors.append("SPRING");
		_color_descriptors.append("STEEL");
		_color_descriptors.append("WEB");
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

static String _to_uppercase_name(const String &name) {
	// Edge cases
	if (name == "blanchedalmond") {
		return "BLANCHED_ALMOND";
	} else if (name == "burlywood") {
		return "BURLY_WOOD";
	} else if (name == "deepskyblue") {
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
	uppercase_name = _insert_underscores(uppercase_name, _modifiable_colors);
	uppercase_name = _insert_underscores(uppercase_name, _color_descriptors);
	return uppercase_name;
}
