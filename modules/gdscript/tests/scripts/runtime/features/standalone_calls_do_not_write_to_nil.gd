# https://github.com/godotengine/godot/issues/70964

func test():
	test_construct(0, false)
	test_utility(0, false)
	test_builtin_call(Vector2.UP, false)
	test_builtin_call_validated(Vector2.UP, false)
	test_object_call(RefCounted.new(), false)
	test_object_call_method_bind(Resource.new(), false)
	test_object_call_method_bind_validated(RefCounted.new(), false)

	print("end")

func test_construct(v, f):
	@warning_ignore("unsafe_call_argument")
	Vector2(v, v) # Built-in type construct.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_utility(v, f):
	abs(v) # Utility function.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_builtin_call(v, f):
	@warning_ignore("unsafe_method_access")
	v.angle() # Built-in method call.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_builtin_call_validated(v: Vector2, f):
	@warning_ignore("return_value_discarded")
	v.abs() # Built-in method call validated.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_object_call(v, f):
	@warning_ignore("unsafe_method_access")
	v.get_reference_count() # Native type method call.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_object_call_method_bind(v: Resource, f):
	@warning_ignore("return_value_discarded")
	v.duplicate() # Native type method call with MethodBind.
	Utils.check(not f) # Test unary operator reading from `nil`.

func test_object_call_method_bind_validated(v: RefCounted, f):
	@warning_ignore("return_value_discarded")
	v.get_reference_count() # Native type method call with validated MethodBind.
	Utils.check(not f) # Test unary operator reading from `nil`.
