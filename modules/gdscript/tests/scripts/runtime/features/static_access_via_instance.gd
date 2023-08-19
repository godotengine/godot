# GH-77098 p.4

@static_unload

class A:
	class InnerClass:
		pass

	enum NamedEnum { VALUE = 111 }
	enum { UNNAMED_ENUM_VALUE = 222 }
	const CONSTANT = 333
	static var static_var := 1

	static func static_func() -> int:
		return 444

class B extends A:
	func test_self():
		print(self.InnerClass is GDScript)
		print(self.NamedEnum)
		print(self.NamedEnum.VALUE)
		print(self.UNNAMED_ENUM_VALUE)
		print(self.CONSTANT)
		@warning_ignore("static_called_on_instance")
		print(self.static_func())

		prints("test_self before:", self.static_var)
		self.static_var = 2
		prints("test_self after:", self.static_var)

func test():
	var hard := B.new()
	hard.test_self()

	print(hard.InnerClass is GDScript)
	print(hard.NamedEnum)
	print(hard.NamedEnum.VALUE)
	print(hard.UNNAMED_ENUM_VALUE)
	print(hard.CONSTANT)
	@warning_ignore("static_called_on_instance")
	print(hard.static_func())

	prints("hard before:", hard.static_var)
	hard.static_var = 3
	prints("hard after:", hard.static_var)

	var weak: Variant = B.new()
	print(weak.InnerClass is GDScript)
	print(weak.NamedEnum)
	print(weak.NamedEnum.VALUE)
	print(weak.UNNAMED_ENUM_VALUE)
	print(weak.CONSTANT)
	@warning_ignore("unsafe_method_access")
	print(weak.static_func())

	prints("weak before:", weak.static_var)
	weak.static_var = 4
	prints("weak after:", weak.static_var)
