func foo(x):
	match x:
		1 + 1:
			print("1+1")
		[1,2,[1,{1:2,2:var z,..}]]:
			print("[1,2,[1,{1:2,2:var z,..}]]")
			print(z)
		1 if true else 2:
			print("1 if true else 2")
		1 < 2:
			print("1 < 2")
		1 or 2 and 1:
			print("1 or 2 and 1")
		6 | 1:
			print("1 | 1")
		1 >> 1:
			print("1 >> 1")
		1, 2 or 3, 4:
			print("1, 2 or 3, 4")
		_:
			print("wildcard")

func test():
	foo(6 | 1)
	foo(1 >> 1)
	foo(2)
	foo(1)
	foo(1+1)
	foo(1 < 2)
	foo([2, 1])
	foo(4)
	foo([1, 2, [1, {1 : 2, 2:3}]])
	foo([1, 2, [1, {1 : 2, 2:[1,3,5, "123"], 4:2}]])
	foo([1, 2, [1, {1 : 2}]])
