class Inner:
	var val := "OK"

func make_inner() -> Self.Inner:
	return Self.Inner.new()

func test():
	var i: Self.Inner = make_inner()
	print(i.val)
