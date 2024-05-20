func test():
	print(f1())
	print(f2())

static func f1(p := f2()) -> int:
	return 1

static func f2(p := f1()) -> int:
	return 2
