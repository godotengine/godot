func foo(x):
	return x + 1

func test():
	@warning_ignore("unsafe_call_argument")
	print(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(foo(0)))))))))))))))))))))))))
