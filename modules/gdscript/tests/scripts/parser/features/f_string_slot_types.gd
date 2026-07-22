# Verifies that an f-string slot stringifies its value the same way as str().
#
# Unlike the scalar literals in f_strings.gd (bool/int/float/enum/String, which
# are constant-folded at analysis time), these non-scalar values are not
# constant-folded, so every slot below exercises the runtime "".join() path.
# Each line prints str(value) and the equivalent f-string slot side by side; the
# two columns must match.

class Stringable:
	func _to_string() -> String:
		return "custom-to-string"

func test():
	print(str(null), " | ", f"{null}")
	print(str([1, 2, 3]), " | ", f"{[1, 2, 3]}")
	print(str([[1, 2], [3, 4]]), " | ", f"{[[1, 2], [3, 4]]}")
	print(str({"a": 1, "b": 2}), " | ", f"{ {"a": 1, "b": 2} }")
	print(str(Vector2(0.25, 1)), " | ", f"{Vector2(0.25, 1)}")
	print(str(Vector3i(4, 5, 6)), " | ", f"{Vector3i(4, 5, 6)}")
	print(str(Color(1, 0, 0, 1)), " | ", f"{Color(1, 0, 0, 1)}")
	print(str(PackedInt32Array([7, 8, 9])), " | ", f"{PackedInt32Array([7, 8, 9])}")
	print(str(&"some_name"), " | ", f"{&"some_name"}")
	print(str(^"Path/To/Node"), " | ", f"{^"Path/To/Node"}")

	var obj := Stringable.new()
	print(str(obj), " | ", f"{obj}")

	# The same values through variables must stringify identically to the literals above.
	var arr := [1, 2, 3]
	var dict := {"a": 1, "b": 2}
	print(str(arr), " | ", f"{arr}")
	print(str(dict), " | ", f"{dict}")
