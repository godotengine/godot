signal hello

func get_signal() -> Signal:
	return hello

class A:
	signal hello

	func get_signal() -> Signal:
		return hello

	class B:
		signal hello

		func get_signal() -> Signal:
			return hello

class C extends A.B:
	func get_signal() -> Signal:
		return hello

func test():
	var a: = A.new()
	var b: = A.B.new()
	var c: = C.new()

	var hello_a_result: = hello == a.get_signal()
	var hello_b_result: = hello == b.get_signal()
	var hello_c_result: = hello == c.get_signal()
	var a_b_result: = a.get_signal() == b.get_signal()
	var a_c_result: = a.get_signal() == c.get_signal()
	var b_c_result: = b.get_signal() == c.get_signal()
	var c_c_result: = c.get_signal() == c.get_signal()

	print("hello == A.hello? %s" % hello_a_result)
	print("hello == A.B.hello? %s" % hello_b_result)
	print("hello == C.hello? %s" % hello_c_result)
	print("A.hello == A.B.hello? %s" % a_b_result)
	print("A.hello == C.hello? %s" % a_c_result)
	print("A.B.hello == C.hello? %s" % b_c_result)
	print("C.hello == C.hello? %s" % c_c_result)
