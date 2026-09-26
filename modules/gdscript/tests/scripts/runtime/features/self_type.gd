class FluentInt:
	var value:int

	static func make(x:int) -> Self:
		return Self.new(x)

	func _init(x:int):
		value = x
	func add_one() -> Self:
		value += 1
		return self
	func add_ten() -> Self:
		value += 10
		return self

func test():
	print(FluentInt.make(100).add_one().add_ten().value)
