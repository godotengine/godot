class Player:
	var x = 3

func test():
	# These should not emit a warning.
	var _player = Player.new()
	print(String.num_uint64(8589934592)) # 2 ^ 33

	# This should emit a warning.
	var some_string = String()
	print(some_string.num_uint64(8589934592)) # 2 ^ 33
