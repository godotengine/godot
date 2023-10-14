# GH-79521

class_name TestStaticMethodAsCallable

static func static_func() -> String:
	return "Test"

func test():
	var a: Callable = TestStaticMethodAsCallable.static_func
	var b: Callable = static_func
	prints(a.call(), a.is_valid())
	prints(b.call(), b.is_valid())
