class MyIterator:
	var count: int

	func _init(p_count: int) -> void:
		count = p_count

	func _iter_init(arg: Array) -> bool:
		prints("_iter_init", arg)
		arg[0] = 0
		return arg[0] < count

	func _iter_next(arg: Array) -> bool:
		prints("_iter_next", arg)
		arg[0] += 1
		return arg[0] < count

	func _iter_get(arg: Variant) -> Variant:
		prints("_iter_get", arg)
		return arg

func test():
	var container := PackedDataContainer.new()
	var _err := container.pack([{
		id = 123,
		node_path = ^"/some/path",
		data = PackedByteArray(),
	}])

	for ref: PackedDataContainerRef in container:
		for key: String in ref:
			print(key)

	print("===")

	for ref: Variant in container:
		for key: String in ref:
			print(key)

	print("===")

	var hard_custom := MyIterator.new(3)
	for x in hard_custom:
		print(x)

	print("===")

	var weak_custom: Variant = MyIterator.new(3)
	for x in weak_custom:
		print(x)
