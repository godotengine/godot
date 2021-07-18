class InnerClass:
	var val := "OK"
	static func create_instance() -> InnerClass:
		return new()

func create_inner_instance() -> InnerClass:
	return InnerClass.create_instance()

func test():
	var instance = create_inner_instance()
	print(instance.val)
