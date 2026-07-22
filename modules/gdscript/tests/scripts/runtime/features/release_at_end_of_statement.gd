class TestRefCounted:
	extends RefCounted
	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("    TestRefCounted released")

func test() -> void:
	test1()
	test2()

func test1(_a: Array = []) -> void:
	print("test1 (untyped Array default)")
	var _count = TestRefCounted.new().get_reference_count()
	print("    end of statement")

func test2(_a: Array[String] = []) -> void:
	print("test2 (typed Array[String] default)")
	var _count = TestRefCounted.new().get_reference_count()
	print("    end of statement")
