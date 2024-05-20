func test():
	print(InnerA.new())

class InnerA extends InnerB:
	pass

class InnerB extends InnerA:
	pass
