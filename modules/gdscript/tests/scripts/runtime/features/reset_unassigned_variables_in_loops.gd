# GH-56223, GH-76569

func test():
	for i in 3:
		var a
		if true:
			var b
			if true:
				var c
				@warning_ignore("unassigned_variable")
				prints("Begin:", i, a, b, c)
				a = 1
				b = 1
				c = 1
				prints("End:", i, a, b, c)
	print("===")
	var j := 0
	while j < 3:
		var a
		if true:
			var b
			if true:
				var c
				@warning_ignore("unassigned_variable")
				prints("Begin:", j, a, b, c)
				a = 1
				b = 1
				c = 1
				prints("End:", j, a, b, c)
		j += 1
