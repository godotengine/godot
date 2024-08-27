enum MyEnum {A, B, C}
const ARRAY = [1, 2]
@static_assert(len(MyEnum) == len(ARRAY), "Custom message.")

func test():
	pass
