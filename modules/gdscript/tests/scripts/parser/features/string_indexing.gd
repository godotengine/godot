func test():
	var s := String("abcd")
	print(s[3])
	print(s[0] + s[1] + s[2] + s[3])
	print(s[3] == s[-1])

	var sn := StringName(&"abcd")
	print(sn[3])
	print(sn[0] + sn[1] + sn[2] + sn[3])
	print(sn[3] == sn[-1])
