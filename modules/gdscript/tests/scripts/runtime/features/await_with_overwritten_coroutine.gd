class BaseClass:
	signal release_bool

	func get_bool() -> bool:
		return true

class InheritedClass extends BaseClass:
	func get_bool() -> bool:
		await release_bool
		return false


var instance: BaseClass = InheritedClass.new()

func test() -> void:
	@warning_ignore("missing_await")
	await_with_overwritten_coroutine()
	instance.release_bool.emit()

func await_with_overwritten_coroutine():
	var res_v := await instance.get_bool()
	print(res_v == true)
	var result: bool = bool(res_v)
	print(result == true)
