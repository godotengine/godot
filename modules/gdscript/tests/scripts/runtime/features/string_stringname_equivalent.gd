# https://github.com/godotengine/godot/issues/64171
# https://github.com/godotengine/godot/issues/60145

var s = "abc"
var sn = &"abc"

func test():
	print("Compare ==: ", "abc" == &"abc")
	print("Compare ==: ", &"abc" == "abc")
	print("Compare !=: ", "abc" != &"abc")
	print("Compare !=: ", &"abc" != "abc")

	print("Concat: ", "abc" + &"def")
	print("Concat: ", &"abc" + "def")
	print("Concat: ", &"abc" + &"def")

	print("literal StringName is String: ", &"abc" is String)
	print("StringName is String: ", sn is String)

	for c in sn:
		print(c)

	print(sn[1])
	print(sn[-1])

	match &"abc":
		"abc":
			print("String-like Match")
		_:
			print("no Match")

	match "abc":
		sn:
			print("String matched StringName")
		_:
			print("no match")

	match &"abc":
		s:
			print("StringName matched String")
		_:
			print("no match")
