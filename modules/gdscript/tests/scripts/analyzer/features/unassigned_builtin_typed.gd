# GH-88117, GH-85796

func test():
	var array: Array
	# Should not emit unassigned warning because the Array type has a default value.
	array.assign([1, 2, 3])
	print(array)
