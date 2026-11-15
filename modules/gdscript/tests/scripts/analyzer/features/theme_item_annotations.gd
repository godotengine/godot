class_name TestThemeItemAnnotations
extends Control

@theme_color var glow_color: Color = Color.GOLD
@theme_constant("glow_px") var glow_size: int = 4
@theme_font var title_font: Font
@theme_font_size var title_font_size: int = 24
@theme_icon var badge: Resource
@theme_stylebox var frame: StyleBox

func test():
	# Not in a tree, so no theme has been read yet: each property holds its initializer.
	prints("glow_color", glow_color == Color.GOLD)
	prints("glow_size", glow_size)
	prints("title_font", title_font)
	prints("title_font_size", title_font_size)
	prints("badge", badge)
	prints("frame", frame)
