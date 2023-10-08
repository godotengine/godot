class Player:
	var x = 3

static func box(x : int, y : int, z : int, x_max : int, y_max : int) -> int:
	return (z * x_max * y_max) + (y * x_max) + x
	
static func boxv(v : Vector3i, x_max : int, y_max : int) -> int:
	return box(v.x, v.y, v.z, x_max, y_max)

func test():
	# These should not emit a warning.
	var _player = Player.new()
	print(String.num_uint64(8589934592)) # 2 ^ 33

	# This should emit a warning.
	var some_string = String()
	print(some_string.num_uint64(8589934592)) # 2 ^ 33

	# These should not emit a warning.
	Globals.boxv(Vector3i(0, 0, 0), 10, 10)
