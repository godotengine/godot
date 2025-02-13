# GH-79521, GH-86032

class_name TestStaticMethodAsCallable

static func static_func() -> String:
	return "Test"

static func another_static_func():
	prints("another_static_func:", static_func.call(), static_func.is_valid())

func test():
	var a: Callable = TestStaticMethodAsCallable.static_func
	var b: Callable = static_func
	prints(a.call(), a.is_valid())
	prints(b.call(), b.is_valid())
	@warning_ignore("static_called_on_instance")
	another_static_func()
	TestStaticMethodAsCallable.another_static_func()
