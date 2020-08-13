// Names from https://en.wikipedia.org/wiki/X11_color_names
// and https://docs.microsoft.com/en-us/dotnet/api/system.windows.media.colors?view=netcore-3.1
#include "core/map.h"
#include "core/vector.h"

static Map<String, Color> _named_colors;
static void _populate_named_colors() {
	if (!_named_colors.empty()) {
		return;
	}
	_named_colors.insert("aliceblue", Color(0.94f, 0.97f, 1.00f));
	_named_colors.insert("antiquewhite", Color(0.98f, 0.92f, 0.84f));
	_named_colors.insert("aqua", Color(0.00f, 1.00f, 1.00f));
	_named_colors.insert("aquamarine", Color(0.50f, 1.00f, 0.83f));
	_named_colors.insert("azure", Color(0.94f, 1.00f, 1.00f));
	_named_colors.insert("beige", Color(0.96f, 0.96f, 0.86f));
	_named_colors.insert("bisque", Color(1.00f, 0.89f, 0.77f));
	_named_colors.insert("black", Color(0.00f, 0.00f, 0.00f));
	_named_colors.insert("blanchedalmond", Color(1.00f, 0.92f, 0.80f));
	_named_colors.insert("blue", Color(0.00f, 0.00f, 1.00f));
	_named_colors.insert("blueviolet", Color(0.54f, 0.17f, 0.89f));
	_named_colors.insert("brown", Color(0.65f, 0.16f, 0.16f));
	_named_colors.insert("burlywood", Color(0.87f, 0.72f, 0.53f));
	_named_colors.insert("cadetblue", Color(0.37f, 0.62f, 0.63f));
	_named_colors.insert("chartreuse", Color(0.50f, 1.00f, 0.00f));
	_named_colors.insert("chocolate", Color(0.82f, 0.41f, 0.12f));
	_named_colors.insert("coral", Color(1.00f, 0.50f, 0.31f));
	_named_colors.insert("cornflowerblue", Color(0.39f, 0.58f, 0.93f));
	_named_colors.insert("cornsilk", Color(1.00f, 0.97f, 0.86f));
	_named_colors.insert("crimson", Color(0.86f, 0.08f, 0.24f));
	_named_colors.insert("cyan", Color(0.00f, 1.00f, 1.00f));
	_named_colors.insert("darkblue", Color(0.00f, 0.00f, 0.55f));
	_named_colors.insert("darkcyan", Color(0.00f, 0.55f, 0.55f));
	_named_colors.insert("darkgoldenrod", Color(0.72f, 0.53f, 0.04f));
	_named_colors.insert("darkgray", Color(0.66f, 0.66f, 0.66f));
	_named_colors.insert("darkgreen", Color(0.00f, 0.39f, 0.00f));
	_named_colors.insert("darkkhaki", Color(0.74f, 0.72f, 0.42f));
	_named_colors.insert("darkmagenta", Color(0.55f, 0.00f, 0.55f));
	_named_colors.insert("darkolivegreen", Color(0.33f, 0.42f, 0.18f));
	_named_colors.insert("darkorange", Color(1.00f, 0.55f, 0.00f));
	_named_colors.insert("darkorchid", Color(0.60f, 0.20f, 0.80f));
	_named_colors.insert("darkred", Color(0.55f, 0.00f, 0.00f));
	_named_colors.insert("darksalmon", Color(0.91f, 0.59f, 0.48f));
	_named_colors.insert("darkseagreen", Color(0.56f, 0.74f, 0.56f));
	_named_colors.insert("darkslateblue", Color(0.28f, 0.24f, 0.55f));
	_named_colors.insert("darkslategray", Color(0.18f, 0.31f, 0.31f));
	_named_colors.insert("darkturquoise", Color(0.00f, 0.81f, 0.82f));
	_named_colors.insert("darkviolet", Color(0.58f, 0.00f, 0.83f));
	_named_colors.insert("deeppink", Color(1.00f, 0.08f, 0.58f));
	_named_colors.insert("deepskyblue", Color(0.00f, 0.75f, 1.00f));
	_named_colors.insert("dimgray", Color(0.41f, 0.41f, 0.41f));
	_named_colors.insert("dodgerblue", Color(0.12f, 0.56f, 1.00f));
	_named_colors.insert("firebrick", Color(0.70f, 0.13f, 0.13f));
	_named_colors.insert("floralwhite", Color(1.00f, 0.98f, 0.94f));
	_named_colors.insert("forestgreen", Color(0.13f, 0.55f, 0.13f));
	_named_colors.insert("fuchsia", Color(1.00f, 0.00f, 1.00f));
	_named_colors.insert("gainsboro", Color(0.86f, 0.86f, 0.86f));
	_named_colors.insert("ghostwhite", Color(0.97f, 0.97f, 1.00f));
	_named_colors.insert("gold", Color(1.00f, 0.84f, 0.00f));
	_named_colors.insert("goldenrod", Color(0.85f, 0.65f, 0.13f));
	_named_colors.insert("gray", Color(0.75f, 0.75f, 0.75f));
	_named_colors.insert("green", Color(0.00f, 1.00f, 0.00f));
	_named_colors.insert("greenyellow", Color(0.68f, 1.00f, 0.18f));
	_named_colors.insert("honeydew", Color(0.94f, 1.00f, 0.94f));
	_named_colors.insert("hotpink", Color(1.00f, 0.41f, 0.71f));
	_named_colors.insert("indianred", Color(0.80f, 0.36f, 0.36f));
	_named_colors.insert("indigo", Color(0.29f, 0.00f, 0.51f));
	_named_colors.insert("ivory", Color(1.00f, 1.00f, 0.94f));
	_named_colors.insert("khaki", Color(0.94f, 0.90f, 0.55f));
	_named_colors.insert("lavender", Color(0.90f, 0.90f, 0.98f));
	_named_colors.insert("lavenderblush", Color(1.00f, 0.94f, 0.96f));
	_named_colors.insert("lawngreen", Color(0.49f, 0.99f, 0.00f));
	_named_colors.insert("lemonchiffon", Color(1.00f, 0.98f, 0.80f));
	_named_colors.insert("lightblue", Color(0.68f, 0.85f, 0.90f));
	_named_colors.insert("lightcoral", Color(0.94f, 0.50f, 0.50f));
	_named_colors.insert("lightcyan", Color(0.88f, 1.00f, 1.00f));
	_named_colors.insert("lightgoldenrod", Color(0.98f, 0.98f, 0.82f));
	_named_colors.insert("lightgray", Color(0.83f, 0.83f, 0.83f));
	_named_colors.insert("lightgreen", Color(0.56f, 0.93f, 0.56f));
	_named_colors.insert("lightpink", Color(1.00f, 0.71f, 0.76f));
	_named_colors.insert("lightsalmon", Color(1.00f, 0.63f, 0.48f));
	_named_colors.insert("lightseagreen", Color(0.13f, 0.70f, 0.67f));
	_named_colors.insert("lightskyblue", Color(0.53f, 0.81f, 0.98f));
	_named_colors.insert("lightslategray", Color(0.47f, 0.53f, 0.60f));
	_named_colors.insert("lightsteelblue", Color(0.69f, 0.77f, 0.87f));
	_named_colors.insert("lightyellow", Color(1.00f, 1.00f, 0.88f));
	_named_colors.insert("lime", Color(0.00f, 1.00f, 0.00f));
	_named_colors.insert("limegreen", Color(0.20f, 0.80f, 0.20f));
	_named_colors.insert("linen", Color(0.98f, 0.94f, 0.90f));
	_named_colors.insert("magenta", Color(1.00f, 0.00f, 1.00f));
	_named_colors.insert("maroon", Color(0.69f, 0.19f, 0.38f));
	_named_colors.insert("mediumaquamarine", Color(0.40f, 0.80f, 0.67f));
	_named_colors.insert("mediumblue", Color(0.00f, 0.00f, 0.80f));
	_named_colors.insert("mediumorchid", Color(0.73f, 0.33f, 0.83f));
	_named_colors.insert("mediumpurple", Color(0.58f, 0.44f, 0.86f));
	_named_colors.insert("mediumseagreen", Color(0.24f, 0.70f, 0.44f));
	_named_colors.insert("mediumslateblue", Color(0.48f, 0.41f, 0.93f));
	_named_colors.insert("mediumspringgreen", Color(0.00f, 0.98f, 0.60f));
	_named_colors.insert("mediumturquoise", Color(0.28f, 0.82f, 0.80f));
	_named_colors.insert("mediumvioletred", Color(0.78f, 0.08f, 0.52f));
	_named_colors.insert("midnightblue", Color(0.10f, 0.10f, 0.44f));
	_named_colors.insert("mintcream", Color(0.96f, 1.00f, 0.98f));
	_named_colors.insert("mistyrose", Color(1.00f, 0.89f, 0.88f));
	_named_colors.insert("moccasin", Color(1.00f, 0.89f, 0.71f));
	_named_colors.insert("navajowhite", Color(1.00f, 0.87f, 0.68f));
	_named_colors.insert("navyblue", Color(0.00f, 0.00f, 0.50f));
	_named_colors.insert("oldlace", Color(0.99f, 0.96f, 0.90f));
	_named_colors.insert("olive", Color(0.50f, 0.50f, 0.00f));
	_named_colors.insert("olivedrab", Color(0.42f, 0.56f, 0.14f));
	_named_colors.insert("orange", Color(1.00f, 0.65f, 0.00f));
	_named_colors.insert("orangered", Color(1.00f, 0.27f, 0.00f));
	_named_colors.insert("orchid", Color(0.85f, 0.44f, 0.84f));
	_named_colors.insert("palegoldenrod", Color(0.93f, 0.91f, 0.67f));
	_named_colors.insert("palegreen", Color(0.60f, 0.98f, 0.60f));
	_named_colors.insert("paleturquoise", Color(0.69f, 0.93f, 0.93f));
	_named_colors.insert("palevioletred", Color(0.86f, 0.44f, 0.58f));
	_named_colors.insert("papayawhip", Color(1.00f, 0.94f, 0.84f));
	_named_colors.insert("peachpuff", Color(1.00f, 0.85f, 0.73f));
	_named_colors.insert("peru", Color(0.80f, 0.52f, 0.25f));
	_named_colors.insert("pink", Color(1.00f, 0.75f, 0.80f));
	_named_colors.insert("plum", Color(0.87f, 0.63f, 0.87f));
	_named_colors.insert("powderblue", Color(0.69f, 0.88f, 0.90f));
	_named_colors.insert("purple", Color(0.63f, 0.13f, 0.94f));
	_named_colors.insert("rebeccapurple", Color(0.40f, 0.20f, 0.60f));
	_named_colors.insert("red", Color(1.00f, 0.00f, 0.00f));
	_named_colors.insert("rosybrown", Color(0.74f, 0.56f, 0.56f));
	_named_colors.insert("royalblue", Color(0.25f, 0.41f, 0.88f));
	_named_colors.insert("saddlebrown", Color(0.55f, 0.27f, 0.07f));
	_named_colors.insert("salmon", Color(0.98f, 0.50f, 0.45f));
	_named_colors.insert("sandybrown", Color(0.96f, 0.64f, 0.38f));
	_named_colors.insert("seagreen", Color(0.18f, 0.55f, 0.34f));
	_named_colors.insert("seashell", Color(1.00f, 0.96f, 0.93f));
	_named_colors.insert("sienna", Color(0.63f, 0.32f, 0.18f));
	_named_colors.insert("silver", Color(0.75f, 0.75f, 0.75f));
	_named_colors.insert("skyblue", Color(0.53f, 0.81f, 0.92f));
	_named_colors.insert("slateblue", Color(0.42f, 0.35f, 0.80f));
	_named_colors.insert("slategray", Color(0.44f, 0.50f, 0.56f));
	_named_colors.insert("snow", Color(1.00f, 0.98f, 0.98f));
	_named_colors.insert("springgreen", Color(0.00f, 1.00f, 0.50f));
	_named_colors.insert("steelblue", Color(0.27f, 0.51f, 0.71f));
	_named_colors.insert("tan", Color(0.82f, 0.71f, 0.55f));
	_named_colors.insert("teal", Color(0.00f, 0.50f, 0.50f));
	_named_colors.insert("thistle", Color(0.85f, 0.75f, 0.85f));
	_named_colors.insert("tomato", Color(1.00f, 0.39f, 0.28f));
	_named_colors.insert("turquoise", Color(0.25f, 0.88f, 0.82f));
	_named_colors.insert("transparent", Color(1.00f, 1.00f, 1.00f, 0.00));
	_named_colors.insert("violet", Color(0.93f, 0.51f, 0.93f));
	_named_colors.insert("webgray", Color(0.50f, 0.50f, 0.50f));
	_named_colors.insert("webgreen", Color(0.00f, 0.50f, 0.00f));
	_named_colors.insert("webmaroon", Color(0.50f, 0.00f, 0.00f));
	_named_colors.insert("webpurple", Color(0.50f, 0.00f, 0.50f));
	_named_colors.insert("wheat", Color(0.96f, 0.87f, 0.70f));
	_named_colors.insert("white", Color(1.00f, 1.00f, 1.00f));
	_named_colors.insert("whitesmoke", Color(0.96f, 0.96f, 0.96f));
	_named_colors.insert("yellow", Color(1.00f, 1.00f, 0.00f));
	_named_colors.insert("yellowgreen", Color(0.60f, 0.80f, 0.20f));
}

static Vector<String> _modifiable_colors;
static Vector<String> _color_descriptors;
static void _populate_uppercase_helper_lists() {
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
		_color_descriptors.append("BLUSH");
		_color_descriptors.append("DARK");
		_color_descriptors.append("DRAB");
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

static void _insert_underscores(String &out_name, const Vector<String> &helper_list) {
	for (int i = 0; i < helper_list.size(); ++i) {
		const String &word = helper_list[i];
		int word_begins_at = out_name.find(word, 0);
		int word_ends_at = word_begins_at + word.size() - 2;
		if (word_begins_at >= 0) {
			ERR_PRINT("Found " + word + " in " + out_name + " between [" + itos(word_begins_at) + "] and [" + itos(word_ends_at) + "].");
		}

		if (word_begins_at > 0) {
			if (out_name[word_begins_at - 1] != '_') {
				out_name.insert(word_begins_at - 1, "_");
			}
		}

		if (word_ends_at < out_name.size()) {
			if (out_name[word_ends_at + 1] != '_') {
				out_name.insert(word_ends_at + 1, "_");
			}
		}
	}
}

static String _to_uppercase_name(const String &name) {
	// Edge cases
	if (name == "blanchedalmond") {
		return "BLANCHED_ALMOND";
	} else if (name == "burlywood") {
		return "BURLY_WOOD";
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

	_populate_uppercase_helper_lists();
	// Cases with standard color names
	String uppercase_name = name.to_upper();
	_insert_underscores(uppercase_name, _modifiable_colors);
	_insert_underscores(uppercase_name, _color_descriptors);
	return uppercase_name;
}