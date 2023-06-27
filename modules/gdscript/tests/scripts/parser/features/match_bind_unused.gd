# https://github.com/godotengine/godot/pull/61666

func test():
	var dict := {"key": "value"}
	match dict:
		{"key": var value}:
			print(value) # used, no warning
	match dict:
		{"key": var value}:
			pass # unused, warning
	match dict:
		{"key": var _value}:
			pass # unused, suppressed warning from underscore
