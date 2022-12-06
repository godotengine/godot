# https://github.com/godotengine/godot/issues/60145

func test():
	match "abc":
		&"abc":
			print("String matched StringName")
		_:
			print("no match")

	match &"abc":
		"abc":
			print("StringName matched String")
		_:
			print("no match")
