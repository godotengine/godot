func test():
	print("warn")

func get(_property: StringName) -> Variant: # Native method
	return null

func char(): # @GDScript function
	return null

func log(): # @GlobalScope function
	return null

func unique(): # A unique name
	return null

# Note: Named lambdas do not override anything.

var named_lambda_1 = func get(): # Native method
	return null

var named_lambda_2 = func char(): # @GDScript function
	return null

var named_lambda_3 = func log(): # @GlobalScope function
	return null

var named_lambda_4 = func unique(): # A unique name
	return null
