class_name ShadowedClass
extends ShadowingBase

var member: int = 0

var print_debug := 'print_debug'
@warning_ignore('shadowed_global_identifier')
var print := 'print'

@warning_ignore_start('unused_variable', 'unused_local_constant')
func test():
	var Array := 'Array'
	var Node := 'Node'
	var is_same := 'is_same'
	var sqrt := 'sqrt'
	var member := 'member'
	var reference := 'reference'
	var ShadowedClass := 'ShadowedClass'
	var base_variable_member
	const base_function_member = 1
	var base_const_member

	print('warn')

func char(_code :int) -> Variant:
	return null

func log(_x: float) -> Variant:
	return null

func method1(_arg: float) -> Variant:
	return null

func method2(..._arg: Array) -> Variant:
	return null

func method3() -> void:
	var sinh: int = 1

	for char in []:
		pass

	var k = 1
	match k:
		var abs:
			pass

var named_lambda = func log(): # Note: Named lambdas do not override anything.
	return null

class floor extends Node:
	var property1: Variant = null

var sin: Variant = null

const cos: Variant = null

enum tan {
	min = 333 # Note: Named enum key do not shadow anything, because it can only be accessed as NamedEnum.KEY.
}

enum {
	clamp,
	clampi = 1
}

@warning_ignore("unused_signal")
signal sqrt

@warning_ignore("unused_signal")
signal s(pow: Variant) # Note: Signal parameter do not shadow anything.
