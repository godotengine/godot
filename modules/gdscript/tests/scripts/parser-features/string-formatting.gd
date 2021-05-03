func test():
	assert("hello %s" % "world" == "hello world")
	assert("hello %s" % true == "hello True")
	assert("hello %s" % false == "hello False")

	assert("hello %d" % 25 == "hello 25")
	assert("hello %d %d" % [25, 42] == "hello 25 42")
	# Pad with spaces.
	assert("hello %3d" % 25 == "hello  25")
	# Pad with zeroes.
	assert("hello %03d" % 25 == "hello 025")

	assert("hello %.02f" % 0.123456 == "hello 0.12")

	# Dynamic padding:
	# <https://docs.godotengine.org/en/stable/getting_started/scripting/gdscript/gdscript_format_string.html#dynamic-padding>
	assert("hello %*.*f" % [7, 3, 0.123456] == "hello   0.123")
	assert("hello %0*.*f" % [7, 3, 0.123456] == "hello 000.123")
