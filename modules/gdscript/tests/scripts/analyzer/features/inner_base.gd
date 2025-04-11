extends InnerA

@override func test():
	super.test()

class InnerA extends InnerAB:
	@virtual @override func test():
		print("InnerA.test")
		super.test()

	class InnerAB extends InnerB:
		@virtual @override func test():
			print("InnerA.InnerAB.test")
			super.test()

class InnerB:
	@virtual func test():
		print("InnerB.test")
