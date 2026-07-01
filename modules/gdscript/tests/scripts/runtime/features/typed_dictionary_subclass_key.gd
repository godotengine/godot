class ThingA:
	pass

class ThingB extends ThingA:
	pass

func test() -> void:
	var dict: Dictionary[ThingA, String] = {}
	var a := ThingA.new()
	var b := ThingB.new()
	dict[a] = "yes"
	dict[b] = "no"
	print(dict[a])
	print(dict[b])
	print("ok")
