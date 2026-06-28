func i_take_lambda(lambda: Callable, param: String):
	lambda.call(param)


func test():
	var my_lambda := func this_is_lambda(x):
		print("Hello")
		print("This is %s" % x)

	i_take_lambda(my_lambda, "a lambda")
