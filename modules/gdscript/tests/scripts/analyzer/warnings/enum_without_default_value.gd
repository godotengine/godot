enum HasZero { A = 0, B = 1 }
enum HasNoZero { A = 1, B = 2 }
var has_zero: HasZero # No warning, because the default `0` is valid.
var has_no_zero: HasNoZero # Warning, because there is no `0` in the enum.


func test():
	print(has_zero)
	print(has_no_zero)
