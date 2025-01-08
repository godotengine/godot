class InnerClass:
	pass

func test():
	var x : InnerClass.DoesNotExist
	print("FAIL")
