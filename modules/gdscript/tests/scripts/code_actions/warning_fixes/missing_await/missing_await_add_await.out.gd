extends Node

func coroutine() -> void:
	@warning_ignore("redundant_await")
	await 0

func _ready() -> void:
	await coroutine()
