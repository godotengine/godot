const BEFORE = 1

@export_range(-10, 10) var a = 0
@export_range(1 + 2, absi(-10) + 1) var b = 5
@export_range(BEFORE + 1, BEFORE + AFTER + 1) var c = 5

const AFTER = 10

func test():
	pass
