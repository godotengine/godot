func foo(x):
	match x:
		1:
			print("1")
		2:
			print("2")
		[1, 2]:
			print("[1, 2]")
		3 or 4:
			print("3 or 4")
		4:
			print("4")
		{1 : 2, 2 : 3}:
			print("{1 : 2, 2 : 3}")
		_:
			print("wildcard")

func test():
	foo(0)
	foo(1)
	foo(2)
	foo([1, 2])
	foo(3)
	foo(4)
	foo([4,4])
	foo({1 : 2, 2 : 3})
	foo({1 : 2, 4 : 3})
