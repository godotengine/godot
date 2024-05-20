class_name TestStaticCalledOnInstance

class Inner:
	static func static_func():
		pass

static func static_func():
	pass

func test():
	print(String.num_uint64(8589934592))
	var some_string := String()
	print(some_string.num_uint64(8589934592)) # Warning.

	TestStaticCalledOnInstance.static_func()
	static_func()
	self.static_func()
	var other := TestStaticCalledOnInstance.new()
	other.static_func() # Warning.

	Inner.static_func()
	var inner := Inner.new()
	inner.static_func() # Warning.
