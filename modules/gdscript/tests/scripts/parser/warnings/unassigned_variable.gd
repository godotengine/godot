func test():
	var unassigned
	print(unassigned)
	unassigned = "something" # Assigned only after use.

	var a
	print(a) # Unassigned, warn.
	if a: # Still unassigned, warn.
		a = 1
		print(a) # Assigned (dead code), don't warn.
	print(a) # "Maybe" assigned, don't warn.
