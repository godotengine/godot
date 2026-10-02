enum MyEnum { A, B }

func get_enum() -> Self.MyEnum:
	return MyEnum.A

func test():
	var e: Self.MyEnum = get_enum()
	print(int(e) == 0)
