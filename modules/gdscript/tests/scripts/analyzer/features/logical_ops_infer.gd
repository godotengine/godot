func test():
	var x: Variant = 42

	var a := not x
	var b := x and x
	var c := x or x

	var _a_is_bool: bool = a
	var _b_is_bool: bool = b
	var _c_is_bool: bool = c
