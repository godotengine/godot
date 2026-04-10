const OtherClass = preload("./static_assert.notest.gd")
@static_assert(len(MyEnum) == len(OtherClass.OTHER_ARRAY))

class InnerClass:
	const INNER_ARRAY = [1, 2, 3]
	@static_assert(len(MyEnum) == len(INNER_ARRAY))

	func method():
		var _lambda = func ():
			const LAMBDA_ARRAY = [1, 2, 3]
			@static_assert(len(MyEnum) == len(LAMBDA_ARRAY))

enum MyEnum {A, B, C}
const ARRAY = [1, 2, 3]
@static_assert(len(MyEnum) == len(ARRAY))

func test():
	const LOCAL_ARRAY = [1, 2, 3]
	@static_assert(len(MyEnum) == len(LOCAL_ARRAY))
