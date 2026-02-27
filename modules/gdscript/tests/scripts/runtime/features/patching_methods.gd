class TestClass:
	static var max_id: int
	var id: int

	func _init() -> void:
		max_id += 1
		id = max_id

	func _to_string() -> String:
		return "<TestClass#%d>" % id

	static func static_test_func(a: int, b: int = 0, ...rest: Array) -> int:
		prints("static_test_func", null, a, b, rest)
		return 0

	func test_func(a: int, b: int = 0, ...rest: Array) -> int:
		prints("test_func", self, a, b, rest)
		return 0

	signal test_signal()

	func test_await(x: int) -> void:
		prints("test_await", x, "before await")
		await test_signal
		prints("test_await", x, "after await")

func add_one(instance: Object, args: Array, previous: Callable) -> int:
	var result: int = previous.callv(args)
	prints("add_one", instance, args, previous, result)
	return result + 1

func add_ten(instance: Object, args: Array, previous: Callable) -> int:
	var result: int = previous.callv(args)
	prints("add_ten", instance, args, previous, result)
	return result + 10

func modify_async(instance: Object, args: Array, previous: Callable) -> void:
	await previous.call(2)
	prints("modify_async", instance, args, previous)

func test():
	print((TestClass as GDScript).patch_method(&"static_test_func", add_one))
	print((TestClass as GDScript).patch_method(&"static_test_func", add_ten))
	print((TestClass as GDScript).patch_method(&"test_func", add_one))
	print((TestClass as GDScript).patch_method(&"test_func", add_ten))

	print("===")
	print(TestClass.static_test_func(1))
	print("---")
	print(TestClass.static_test_func(1, 2))
	print("---")
	print(TestClass.static_test_func(1, 2, 3))

	var t := TestClass.new()

	print("===")
	print(t.test_func(1))
	print("---")
	print(t.test_func(1, 2))
	print("---")
	print(t.test_func(1, 2, 3))

	print("===")
	@warning_ignore("missing_await")
	t.test_await(1)
	# The replacement does not affect existing coroutines.
	print((TestClass as GDScript).patch_method(&"test_await", modify_async))
	t.test_signal.emit()

	print("---")
	@warning_ignore("missing_await")
	t.test_await(1)
	t.test_signal.emit()
