@abstract class_name ABC extends RefCounted

@abstract func simulate() -> void

func coroutine() -> void:
	@warning_ignore("redundant_await")
	await 0

func other_method() -> void:
	await simulate()
