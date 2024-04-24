# GH-89958

func test():
	if true:
		@warning_ignore("unused_variable")
		var a = 1
		@warning_ignore("unused_variable")
		var b := 1
		@warning_ignore("unused_variable")
		var c := 1

	if true:
		@warning_ignore("unassigned_variable")
		var a
		print(a)
		@warning_ignore("unassigned_variable")
		var b
		print(b)
		@warning_ignore("unassigned_variable")
		var c: Object
		print(c)
