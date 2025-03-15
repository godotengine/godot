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
		var a
		@warning_ignore("unassigned_variable")
		print(a)
		var b
		@warning_ignore("unassigned_variable")
		print(b)
		var c: Object
		@warning_ignore("unassigned_variable")
		print(c)
