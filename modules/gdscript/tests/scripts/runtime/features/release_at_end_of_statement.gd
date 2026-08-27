class TestRefCounted:
	extends RefCounted
	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("    TestRefCounted released")

func test() -> void:
	test1()
	test2()
	test3()

func test1(_a: Array = []) -> void:
	print("test1 (untyped Array default)")
	var _count = TestRefCounted.new().get_reference_count()
	print("    end of statement")

func test2(_a: Array[String] = []) -> void:
	print("test2 (typed Array[String] default)")
	var _count = TestRefCounted.new().get_reference_count()
	print("    end of statement")

func returns_param(param):
	return param

func test3() -> void:
	print("test3 (function call)")
	returns_param(TestRefCounted.new())
	print("    end of statement")
