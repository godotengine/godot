
var base: Array[float] = [0.1, 200.2, 30.5];

func test():
	# Error
	# Typed conversion constructor checks if base and the provided type are compatible.
	var convert = Array[Node](base);
