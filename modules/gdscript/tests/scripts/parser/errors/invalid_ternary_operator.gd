func test():
	var amount = 50
	# C-style ternary operator is invalid in GDScript.
	# The valid syntax is `"yes" if amount < 60 else "no"`, like in Python.
	var ternary = amount < 60 ? "yes" : "no"
