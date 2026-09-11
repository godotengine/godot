class Inner:
	pass


func test() -> void:
	# A script class reference is itself an `Object` instance, so it is a
	# valid key for `Dictionary[Object, ...]`.
	var dict: Dictionary[Object, String] = {}
	dict[Inner] = "class reference key"
	print(dict[Inner])
