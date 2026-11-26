class Instance:
	func _init() -> void:
		print("Instance _init")

	func _to_string() -> String:
		return "<Instance>"

	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("Instance predelete")

# GH-116706

class LocalOwner:
	signal never_emitted()

	func _init() -> void:
		print("LocalOwner _init")

	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("LocalOwner predelete")

	func interrupted_coroutine() -> void:
		print("interrupted_coroutine begin")
		var _instance := Instance.new()
		await never_emitted
		print("interrupted_coroutine end")

func subtest_order():
	print("subtest_order begin")
	var local_owner := LocalOwner.new()
	@warning_ignore("missing_await")
	local_owner.interrupted_coroutine()
	local_owner = null
	print("subtest_order end")

# GH-117049

signal tick()

func await_before_and_after() -> void:
	await tick
	var packed_array: PackedStringArray = ["abc"]
	var instance := Instance.new()
	await tick
	prints(packed_array, instance)

func await_two_after() -> void:
	var packed_array: PackedStringArray = ["abc"]
	var instance := Instance.new()
	await tick
	await tick
	prints(packed_array, instance)

func subtest_resume():
	print("subtest_resume begin")

	@warning_ignore("missing_await")
	await_before_and_after()
	tick.emit()
	tick.emit()

	print("---")

	@warning_ignore("missing_await")
	await_two_after()
	tick.emit()
	tick.emit()

	print("subtest_resume end")

# ===

func test():
	subtest_order()
	print("===")
	subtest_resume()
