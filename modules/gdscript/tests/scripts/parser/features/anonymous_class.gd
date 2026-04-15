class Base:
	var value = 10

	func get_value():
		return value

	func describe():
		return "base"


class WithInit:
	var name: String

	func _init(p_name: String):
		name = p_name

	func greet():
		return "hello from " + name


func test():
	# Basic override.
	var obj = Base.new():
		func describe():
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
		func describe():
			return "extra=" + str(extra)
	print(obj3.describe())

	# Multiple anonymous classes from same base.
	var a = Base.new():
		func describe():
			return "first"
	var b = Base.new():
		func describe():
			return "second"
	print(a.describe())
	print(b.describe())
