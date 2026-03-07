# GH-116706

class Instance:
	func _init() -> void:
		print("Instance _init")

	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("Instance predelete")

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

func test():
	print("test begin")
	var local_owner := LocalOwner.new()
	@warning_ignore("missing_await")
	local_owner.interrupted_coroutine()
	local_owner = null
	print("test end")
