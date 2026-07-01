class InnerClass:
	static func test():
		Utils.check(load("load_relative_path.notest.gd").get(&"TEST") == 123)

func get_path() -> String:
	return "load_relative_path.notest.gd"

func load(path: String) -> Resource:
	print(path)
	return null

func test():
	Utils.check(load("load_relative_path.notest.gd").get(&"TEST") == 123)
	Utils.check(load("./load_relative_path.notest.gd").get(&"TEST") == 123)
	Utils.check(load("../features/load_relative_path.notest.gd").get(&"TEST") == 123)
	Utils.check(load(get_path()).get(&"TEST") == 123)

	InnerClass.test()

	# Path transformation does not apply to custom `load()` functions.
	var _temp := self.load("load_relative_path.notest.gd")
