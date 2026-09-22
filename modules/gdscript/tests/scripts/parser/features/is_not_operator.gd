func test():
	var i: Variant = 123
	var s: Variant = "str"
	prints(i is int, i is not int)
	prints(s is int, s is not int)

	var a: Variant = false
	var b: Variant = true
	prints(a == b is int, a == b is not int)
	prints(a == (b is int), a == (b is not int))
	prints((a == b) is int, (a == b) is not int)
