# Enum is equivalent to int for comparisons and operations.
enum MyEnum {
	ZERO,
	ONE,
	TWO,
}

enum OtherEnum {
	ZERO,
	ONE,
	TWO,
}

func test():
	print(MyEnum.ZERO == OtherEnum.ZERO)
	print(MyEnum.ZERO == 1)
	print(MyEnum.ZERO != OtherEnum.ONE)
	print(MyEnum.ZERO != 0)

	print(MyEnum.ONE + OtherEnum.TWO)
	print(2 - MyEnum.ONE)
