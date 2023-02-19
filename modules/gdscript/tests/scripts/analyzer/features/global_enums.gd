func test():
	var type: Variant.Type
	type = Variant.Type.TYPE_INT
	print(type)
	type = TYPE_FLOAT
	print(type)

	var direction: ClockDirection
	direction = ClockDirection.CLOCKWISE
	print(direction)
	direction = COUNTERCLOCKWISE
	print(direction)

	var duper := Duper.new()
	duper.set_type(Variant.Type.TYPE_INT)
	duper.set_type(TYPE_FLOAT)
	duper.set_direction(ClockDirection.CLOCKWISE)
	duper.set_direction(COUNTERCLOCKWISE)

class Super:
	func set_type(type: Variant.Type) -> void:
		print(type)
	func set_direction(dir: ClockDirection) -> void:
		print(dir)

class Duper extends Super:
	func set_type(type: Variant.Type) -> void:
		print(type)
	func set_direction(dir: ClockDirection) -> void:
		print(dir)
