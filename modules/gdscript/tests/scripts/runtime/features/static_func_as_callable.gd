# GH-41919

class_name TestStaticFuncAsCallable

class InnerClass:
	static func inner_my_func():
		print("inner_my_func")

static func my_func():
		print("my_func")

var a: Callable = TestStaticFuncAsCallable.my_func
var b: Callable = InnerClass.inner_my_func

func test():
	a.call()
	b.call()
