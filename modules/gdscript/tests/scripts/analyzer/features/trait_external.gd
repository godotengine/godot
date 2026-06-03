class Impl extends RefCounted implements ExternalSystem:
	func get_value() -> int:
		return 99

func use_it(s: ExternalSystem) -> int:
	return s.get_value()

func test():
	print(use_it(Impl.new()))
