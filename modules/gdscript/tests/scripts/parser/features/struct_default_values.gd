struct Enemy:
	var health: int = 100
	var position: Vector2 = Vector2.ZERO
	var attacking: bool = false

struct Config:
	var data  # Untyped member

func test():
	print("Struct with default values and untyped members parsed")
