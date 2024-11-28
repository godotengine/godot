func foo(n: int) -> bool:
	print("foo")
	return n > 0

func bar(n: int) -> int:
	print("bar")
	return n

func add(a: int, b: int) -> int:
	print("add")
	return a + b

func test_if(f: bool, a: int, b: int) -> void:
	print("--")
	if f, foo(a), var n := bar(b):
		print("t:%s" % n)
	elif var n = add(a, b), n >= 1:
		print("tt:%s" % n)
	else:
		print("f")

func test():
	if var x := 100:
		if x >= 1, var y := "ttt":
			print(y)
		print(100 + signi(x))

	test_if(false, 0, 0);
	test_if(false, 0, 1);
	test_if(false, 1, 0);
	test_if(false, 1, 1);
	test_if(true, 0, 0);
	test_if(true, 0, 1);
	test_if(true, 0, 2);
	test_if(true, 1, 0);
	test_if(true, 1, 1);
	test_if(true, 1, 2);
