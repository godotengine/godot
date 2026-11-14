class MyClass:
	static func my_static() -> void:
		print("success")

func test():
	print(MyClass["my_static"])
