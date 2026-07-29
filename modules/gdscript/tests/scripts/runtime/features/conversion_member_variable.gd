class Foo:
	const NAME: StringName = String("Barrel T.")

	static var foo: StringName = String("1")
	var bar: String = &"1"

	func _init():
		Utils.check(typeof(NAME) == TYPE_STRING_NAME)
		Utils.check(NAME == &"Barrel T.")

		Utils.check(typeof(foo) == TYPE_STRING_NAME)
		Utils.check(foo == &"1")

		foo = String("2")
		Utils.check(typeof(foo) == TYPE_STRING_NAME)
		Utils.check(foo == &"2")

		Utils.check(typeof(bar) == TYPE_STRING)
		Utils.check(bar == "1")

		bar = &"2"
		Utils.check(typeof(bar) == TYPE_STRING)
		Utils.check(bar == "2")


func test_external_property_typed(f: Foo):
	f.foo = String("3")
	Utils.check(typeof(f.foo) == TYPE_STRING_NAME)
	Utils.check(f.foo == &"3")

	f.bar = NodePath("3")
	Utils.check(typeof(f.bar) == TYPE_STRING)
	Utils.check(f.bar == "3")


func test_external_property_untyped(f):
	f.foo = "4"
	Utils.check(typeof(f.foo) == TYPE_STRING_NAME)
	Utils.check(f.foo == &"4")

	f.bar = StringName("4")
	Utils.check(typeof(f.bar) == TYPE_STRING)
	Utils.check(f.bar == "4")


func test_static(cls):
	Foo.foo = String("5")
	Utils.check(typeof(Foo.foo) == TYPE_STRING_NAME)
	Utils.check(Foo.foo == &"5")

	cls.foo = String("6")
	Utils.check(typeof(cls.foo) == TYPE_STRING_NAME)
	Utils.check(cls.foo == &"6")

func test():
	var foo := Foo.new()

	test_external_property_typed(foo)
	test_external_property_untyped(foo)
	test_static(Foo)

	print('ok')
