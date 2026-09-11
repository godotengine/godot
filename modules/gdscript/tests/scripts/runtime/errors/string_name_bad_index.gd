func test():
	var s := String("abcd")
	s[1] = "z" # OK: can assign to String index, as it is mutable

	var sn := StringName(&"abcd")
	sn[1] = "z" # Error: cannot modify a read-only type
