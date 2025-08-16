var member = "foo"

func bar():
	print("bar")

func test():
	var lambda1 = func():
		print(member)
	lambda1.call()

	var lambda2 = func():
		var nested = func():
			print(member)
		nested.call()
	lambda2.call()

	var lambda3 = func():
		bar()
	lambda3.call()

	var lambda4 = func():
		return self
	print(lambda4.call() == self)
