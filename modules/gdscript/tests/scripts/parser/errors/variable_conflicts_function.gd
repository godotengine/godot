var test = 25

# Error here. The difference with `variable-conflicts-function.gd` is that here,
# the function is defined *before* the variable.
func test():
	pass
