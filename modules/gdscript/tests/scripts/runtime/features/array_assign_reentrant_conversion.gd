class RidClearer extends RefCounted:
	var source: Array

	func get_rid() -> RID:
		if source != null:
			source.clear()
			source = []
		return RID()

func test():
	# GH-125402: Conversion can re-enter script code and mutate the source array.
	var source: Array = []
	var clearer := RidClearer.new()
	clearer.source = source
	source.append(clearer)
	source.append(RefCounted.new())

	var typed: Array[RID] = []
	typed.assign(source)
	print(typed.size())
