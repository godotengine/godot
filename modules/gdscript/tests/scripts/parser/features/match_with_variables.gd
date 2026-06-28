func test():
	var a = 1
	match 1:
		a:
			print("reach 1")

	var dict = { b = 2 }
	match 2:
		dict.b:
			print("reach 2")

	var nested_dict = {
		sub = { c = 3 }
	}
	match 3:
		nested_dict.sub.c:
			print("reach 3")

	var sub_pattern = { d = 4 }
	match [4]:
		[sub_pattern.d]:
			print("reach 4")
