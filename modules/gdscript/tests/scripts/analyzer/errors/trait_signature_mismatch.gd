class Something extends RefCounted implements SimSystem:
	func declared_func() -> bool:
		return true

trait SimSystem:
	func declared_func(value: int) -> bool

func test():
	pass
