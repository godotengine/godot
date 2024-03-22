# GH-77666

func test():
	var ref := RefCounted.new()
	print(ref.get_reference_count())

	if true:
		var _temp := ref

	print(ref.get_reference_count())
