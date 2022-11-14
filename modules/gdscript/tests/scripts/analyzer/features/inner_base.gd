extends InnerA

func test():
	super.test()

class InnerA extends InnerAB:
	func test():
		print("InnerA.test")
		super.test()

	class InnerAB extends InnerB:
		func test():
			print("InnerA.InnerAB.test")
			super.test()

class InnerB:
	func test():
		print("InnerB.test")
