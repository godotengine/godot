enum HasZero { A = 0, B = 1 }
enum HasNoZero { A = 1, B = 2 }
var has_zero: HasZero # No warning, because the default `0` is valid.
var has_no_zero: HasNoZero # Warning, because there is no `0` in the enum.


func test():
	print(has_zero)
	print(has_no_zero)


# GH-94634. A parameter is either mandatory or has a default value.
func test_no_exec(param: HasNoZero) -> void:
	print(param)

	# Loop iterator always has a value.
	for i: HasNoZero in HasNoZero.values():
		print(i)

	match param:
		# Pattern bind always has a value.
		var x:
			print(x)
