class A:
	func _notification(what: int) -> void:
		if what == 10:
			print("A")

	func _get_property_list() -> Array[Dictionary]:
		var ret: Array[Dictionary]
		ret.append({ "name": "password", "type": TYPE_INT })
		return ret

	func _set(property: StringName, _value: Variant) -> bool:
		if property == &"password":
			# Should not be reached.
			Utils.check(false)
		return false

	func _get(property: StringName) -> Variant:
		if property == &"surname":
			# Should not be reached.
			Utils.check(false)
		elif property == &"password":
			return 2137
		return null

	func _validate_property(property: Dictionary) -> void:
		if property["name"] == "exported_property":
			property["usage"] |= PROPERTY_USAGE_ARRAY
		elif property["name"] == "surname":
			# Dynamic property, should not be validated.
			Utils.check(false)

	func _property_can_revert(property: StringName) -> bool:
		if property == &"fake":
			Utils.check(false)
		return false

	func _property_get_revert(property: StringName) -> Variant:
		if property  == &"fake":
			Utils.check(false)
		return null

class B extends A:
	func _notification(what: int) -> void:
		if what == 15:
			print("B")

	func _get_property_list() -> Array[Dictionary]:
		var ret: Array[Dictionary]
		return ret

	func _set(property: StringName, value: Variant) -> bool:
		if property == &"password":
			prints("Setting password to:", value)
			return true
		return false

	func _get(property: StringName) -> Variant:
		if property == &"surname":
			return "B-man"
		return null

	func _validate_property(_property: Dictionary) -> void:
		pass

	func _property_can_revert(property: StringName) -> bool:
		return property == &"fake"

	func _property_get_revert(property: StringName) -> Variant:
		if property == &"fake":
			return "fake"
		return null

class C extends B:
	@export var exported_property: int

	func _notification(what: int) -> void:
		if what == 10:
			print("C")

	func _get_property_list() -> Array[Dictionary]:
		var ret: Array[Dictionary]
		ret.append({ "name": "surname", "type": TYPE_STRING })
		return ret

	func _set(_property: StringName, _value: Variant) -> bool:
		return false

	func _get(_property: StringName) -> Variant:
		# Returning null allows to call super method.
		return null

	func _validate_property(property: Dictionary) -> void:
		if property["name"] == "exported_property":
			property["type"] = TYPE_ARRAY


	func _property_can_revert(_property: StringName) -> bool:
		return false

	func _property_get_revert(_property: StringName) -> Variant:
		print("Get revert C")
		return null

func test():
	var c := C.new()

	c.notification(10)
	c.notification(15)

	# Uses var_to_str() for better formatting.
	for property in c.get_property_list():
		if not (property["usage"] & PROPERTY_USAGE_CATEGORY):
			Utils.print_property_extended_info(property)

	prints("surname:", c.get("surname"))
	prints("password:", c.get("password"))
	c.set("password", 420)

	print(c.property_can_revert(&"fake"))
	print(c.property_can_revert(&"fake2"))
	print(c.property_get_revert(&"fake"))
	print(c.property_get_revert(&"fake2"))
