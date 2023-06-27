func foo(x):
	match x:
		1, [2]:
			print('1, [2]')
		_:
			print('wildcard')

func bar(x):
	match x:
		[1], [2], [3]:
			print('[1], [2], [3]')
		[4]:
			print('[4]')
		_:
			print('wildcard')

func test():
	foo(1)
	foo([2])
	foo(2)
	bar([1])
	bar([2])
	bar([3])
	bar([4])
	bar([5])

