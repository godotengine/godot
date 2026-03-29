# GH-117912

class A:
	static func overload_me() -> void:
		pass

class B extends A:
	var overload_me

func test():
	pass
