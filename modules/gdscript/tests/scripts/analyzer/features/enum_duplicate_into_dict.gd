enum Enum {V1, V2}

func test():
	var enumAsDict: Dictionary = Enum.duplicate()
	var enumAsVariant = Enum.duplicate()
	print(Enum.has("V1"))
	print(enumAsDict.has("V1"))
	print(enumAsVariant.has("V1"))
	enumAsDict.clear()
	enumAsVariant.clear()
	print(Enum.has("V1"))
	print(enumAsDict.has("V1"))
	print(enumAsVariant.has("V1"))
