class Base:
	var value = 10

	func get_value():
		return value

	func describe() -> String:
		return "base"


class WithInit:
	var name: String

	func _init(p_name: String):
		name = p_name

	func greet():
		return "hello from " + name


class Outer:
	class Inner:
		func describe() -> String:
			return "inner"

	func wrap() -> Object:
		return null


class InnerWrap:
	func name_of() -> String:
		return "inner"


func test():
	# Basic override.
	var obj = Base.new():
		func describe() -> String:
			return "anonymous"
	print(obj.get_value())
	print(obj.describe())

	# Constructor arguments forwarded to parent _init.
	var obj2 = WithInit.new("anon"):
		func greet():
			return "overridden greet from " + name
	print(obj2.greet())

	# Adding new member variables.
	var obj3 = Base.new():
		var extra = 42
		func describe() -> String:
			return "extra=" + str(extra)
	print(obj3.describe())

	# Multiple anonymous classes from same base.
	var a = Base.new():
		func describe() -> String:
			return "first"
	var b = Base.new():
		func describe() -> String:
			return "second"
	print(a.describe())
	print(b.describe())

	# Typed variable assignment.
	var typed: Base = Base.new():
		func describe() -> String:
			return "typed"
	print(typed.describe())

	# Dotted base class.
	var dotted = Outer.Inner.new():
		func describe() -> String:
			return "overridden"
	print(dotted.describe())

	# Nested anonymous classes.
	var o = Outer.new():
		func wrap() -> Object:
			return InnerWrap.new():
				func name_of() -> String:
					return "nested"
	print(o.wrap().name_of())

	# Preloaded script as base.
	var pre = preload("anonymous_class_preload_base.notest.gd").new():
		func describe():
			return "preload-anon value=" + str(value)
	print(pre.describe())
