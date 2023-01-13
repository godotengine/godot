# `Variant::can_convert_strict()` allows `String -> Color`, `but doesn't allow `Color -> String`.
# This test is to ensure that such asymmetric conversions are handled correctly for `Variant` types.

func test():
	var c: Color = Color.RED # Test non-constant.
	var s: String = c
