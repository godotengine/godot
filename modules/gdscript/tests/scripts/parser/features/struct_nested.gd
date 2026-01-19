struct Position:
	var x: float = 0.0
	var y: float = 0.0
	var z: float = 0.0

struct Entity:
	var id: int
	var pos: Position
	var health: int = 100

func test():
	var e = Entity(1, Position(10.0, 20.0, 30.0), 150)
	pass
