class RefInstance extends RefCounted:
	signal remove_self_ref()
	var some_state := 0

	func emit_remove_self_ref_signal() -> void:
		var before := get_reference_count()
		remove_self_ref.emit()
		var diff := before - get_reference_count()
		prints("Did drop itself?:", diff == 1)
		some_state += 1

	func _notification(what: int) -> void:
		if what == NOTIFICATION_PREDELETE:
			print("===")
			print("Entered NOTIFICATION_PREDELETE")
			call_1()

	func call_1() -> void:
		prints("call_1: ok. Count:", get_reference_count())
		call_2()

	func call_2() -> void:
		var self_copy := self
		prints("call_2:", self_copy, self)
		prints("self == null:", self == null)
		prints("self_copy == null:", self_copy == null)
		prints("self_copy == self:", self_copy == self)
		prints("Some state:", self.some_state, self["some_state"])
		call_3(self)

	func call_3(self_arg: RefInstance) -> void:
		prints("call_3 self arg:", self_arg)

var instance: RefInstance

func test():
	print("===")
	instance = RefInstance.new()
	@warning_ignore("return_value_discarded")
	instance.remove_self_ref.connect(func ():
		instance = null
	)
	instance.emit_remove_self_ref_signal()
	print("===")
