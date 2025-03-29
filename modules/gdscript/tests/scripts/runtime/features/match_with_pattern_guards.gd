var global := 0

func test():
	var a = 0
	var b = 1

	match a:
		0 when b == 0:
			print("does not run" if true else "")
		0 when b == 1:
			print("guards work")
		_:
			print("does not run")

	match a:
		var a_bind when b == 0:
			prints("a is", a_bind, "and b is 0")
		var a_bind when b == 1:
			prints("a is", a_bind, "and b is 1")
		_:
			print("does not run")

	match a:
		var a_bind when a_bind < 0:
			print("a is less than zero")
		var a_bind when a_bind == 0:
			print("a is equal to zero")
		_:
			print("a is more than zero")

	match [1, 2, 3]:
		[1, 2, var element] when element == 0:
			print("does not run")
		[1, 2, var element] when element == 3:
			print("3rd element is 3")

	match a:
		_ when b == 0:
			print("does not run")
		_ when b == 1:
			print("works with wildcard too.")
		_:
			print("does not run")

	match a:
		0, 1 when b == 0:
			print("does not run")
		0, 1 when b == 1:
			print("guard with multiple patterns")
		_:
			print("does not run")

	match a:
		0 when b == 0:
			print("does not run")
		0:
			print("regular pattern after guard mismatch")

	match a:
		1 when side_effect():
			print("should not run the side effect call")
		0 when side_effect():
			print("will run the side effect call, but not this")
		_:
			Utils.check(global == 1)
			print("side effect only ran once")

func side_effect():
	print("side effect")
	global += 1
	return false
