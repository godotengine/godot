const PrivateMembers = preload("./private_member_access.notest.gd").PrivateMembers

signal _own_private_signal


class Subclass extends PrivateMembers:
	func access_through_other_instance(other: PrivateMembers) -> void:
		print(other._private_variable)
		other._private_method()
		print(self._private_variable)


class SameFileInner:
	var _inner_variable := 1


class SameFileOtherInner:
	func access_sibling(sibling: SameFileInner) -> void:
		print(sibling._inner_variable)


func access_external_private_members(other: PrivateMembers, subclass_instance: Subclass) -> void:
	print(other._private_variable)
	other._private_variable = 2
	other._private_method()
	print(other._private_method)
	print(other._private_signal)
	print(PrivateMembers._PRIVATE_CONSTANT)
	print(PrivateMembers._PrivateEnum.VALUE)
	print(PrivateMembers._private_static_variable)
	PrivateMembers._private_static_method()
	print(PrivateMembers._PrivateInner)
	print(subclass_instance._private_variable)
	var outside_lambda := func() -> void: other._private_method()
	outside_lambda.call()


func access_allowed_members(other: PrivateMembers, untyped_other, inner: SameFileInner) -> void:
	print(other.public_variable)
	other._ready()
	print(other._to_string())
	print(untyped_other._private_variable)
	var inferred_other = other
	print(inferred_other._private_variable)
	print(inner._inner_variable)
	print(other.get("_private_variable"))
	other.get_private_method().call()


func _own_private_method() -> void:
	pass


func use_own_private_members_inside_class() -> void:
	@warning_ignore("return_value_discarded")
	_own_private_signal.connect(_own_private_method)
	var inside_lambda := func() -> void: _own_private_method()
	inside_lambda.call()


func access_ignored_member(other: PrivateMembers) -> void:
	@warning_ignore("private_member_access")
	print(other._private_variable)


func test():
	pass
