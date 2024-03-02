#*
#* code_edit.gd
#* =============================================================================
#* Copyright 2024 Serhii Snitsaruk
#*
#* Use of this source code is governed by an MIT-style
#* license that can be found in the LICENSE file or at
#* https://opensource.org/licenses/MIT.
#* =============================================================================
#*

extends CodeEdit

const RESERVED_WORDS := [
		# Control flow.
		"break",
		"continue",
		"elif",
		"else",
		"for",
		"if",
		"match",
		"pass",
		"return",
		"when",
		"while",
		# Declarations.
		"class",
		"class_name",
		"const",
		"enum",
		"extends",
		"func",
		"namespace",
		"signal",
		"static",
		"trait",
		"var",
		# Other keywords.
		"await",
		"breakpoint",
		"self",
		"super",
		"yield",
		# Operators.
		"and",
		"as",
		"in",
		"is",
		"not",
		"or",
		# Special values.
		"false",
		"null",
		"true",
		# Constants.
		"INF",
		"NAN",
		"PI",
		"TAU",
		# Functions.
		"assert",
		"preload",
]

const TYPE_WORDS := [
	"bool",
	"int",
	"float",
	"void",
	"String",
	"Vector2",
	"Vector2i",
	"Rect2",
	"Rect2i",
	"Vector3",
	"Vector3i",
	"Transform2D",
	"Vector4",
	"Vector4i",
	"Plane",
	"Quaternion",
	"AABB",
	"Basis",
	"Transform3D",
	"Projection",
	"Color",
	"StringName",
	"NodePath",
	"RID",
	"Callable",
	"Signal",
	"Dictionary",
	"Array",
	"PackedByteArray",
	"PackedInt32Array",
	"PackedInt64Array",
	"PackedFloat32Array",
	"PackedFloat64Array",
	"PackedStringArray",
	"PackedVector2Array",
	"PackedVector3Array",
	"PackedColorArray",
	# Other types
	"Status",
]


func _ready() -> void:
	var highlighter := CodeHighlighter.new()
	syntax_highlighter = highlighter
	highlighter.number_color = Color.AQUAMARINE
	highlighter.symbol_color = Color.CORNFLOWER_BLUE
	highlighter.function_color = Color.DEEP_SKY_BLUE
	highlighter.member_variable_color = Color.LIGHT_BLUE

	# Engine types
	for c in ClassDB.get_class_list():
		syntax_highlighter.add_keyword_color(c, Color.AQUAMARINE)

	syntax_highlighter.add_color_region("#", "", Color.DIM_GRAY, true)
	syntax_highlighter.add_color_region("@", " ", Color.GOLDENROD)
	syntax_highlighter.add_color_region("\"", "\"", Color.GOLD)

	for keyword in RESERVED_WORDS:
		syntax_highlighter.add_keyword_color(keyword, Color.INDIAN_RED)

	for typeword in TYPE_WORDS:
		syntax_highlighter.add_keyword_color(typeword, Color.AQUAMARINE)


func set_source_code(source_code: String) -> void:
	# Hide license header
	var idx: int = source_code.find("#*")
	while idx != - 1:
		source_code = source_code.substr(0, idx) + source_code.substr(source_code.findn("\n", idx) + 1)
		idx = source_code.findn("#*", idx)

	text = "" # Workaround
	text = source_code
