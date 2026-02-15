extends InnerA

@override
func test():
	super.test()

class InnerA extends InnerAB:
	@override
	func test():
		print("InnerA.test")
		super.test()

	class InnerAB extends InnerB:
		@override
		func test():
			print("InnerA.InnerAB.test")
			super.test()

class InnerB:
	func test():
		print("InnerB.test")
