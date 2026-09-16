func test():
	# GH-114299
	var a1: Array[PackedInt32Array] = [[1]]
	var a2 = [[2]] as Array[PackedInt32Array]
	print(var_to_str(a1))
	print(var_to_str(a2))
