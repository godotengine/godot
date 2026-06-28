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

	match "abc":
		&"abc":
			print("String matched StringName literal")
		_:
			print("no Match")

	match &"abc":
		"abc":
			print("StringName matched String literal")
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
