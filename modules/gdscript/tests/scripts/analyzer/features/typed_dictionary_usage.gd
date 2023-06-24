class A: pass
class B extends A: pass

enum E { E0 = 391, E1 = 193 }

func floats_identity(floats: Dictionary[float, float]): return floats

class Members:
	var one: Dictionary[int, int] = { 104: 401 }
	var two: Dictionary[int, int] = one

	func check_passing() -> bool:
		assert(str(one) == '{ 104: 401 }')
		assert(str(two) == '{ 104: 401 }')
		two[582] = 285
		assert(str(one) == '{ 104: 401, 582: 285 }')
		assert(str(two) == '{ 104: 401, 582: 285 }')
		two = { 486: 684 }
		assert(str(one) == '{ 104: 401, 582: 285 }')
		assert(str(two) == '{ 486: 684 }')
		return true


@warning_ignore("unsafe_method_access")
@warning_ignore("assert_always_true")
@warning_ignore("return_value_discarded")
func test():
	var untyped_basic = { 459: 954 }
	assert(str(untyped_basic) == '{ 459: 954 }')
	assert(untyped_basic.get_typed_key_builtin() == TYPE_NIL)
	assert(untyped_basic.get_typed_value_builtin() == TYPE_NIL)

	var inferred_basic := { 366: 663 }
	assert(str(inferred_basic) == '{ 366: 663 }')
	assert(inferred_basic.get_typed_key_builtin() == TYPE_NIL)
	assert(inferred_basic.get_typed_value_builtin() == TYPE_NIL)

	var typed_basic: Dictionary = { 521: 125 }
	assert(str(typed_basic) == '{ 521: 125 }')
	assert(typed_basic.get_typed_key_builtin() == TYPE_NIL)
	assert(typed_basic.get_typed_value_builtin() == TYPE_NIL)


	var empty_floats: Dictionary[float, float] = {}
	assert(str(empty_floats) == '{  }')
	assert(empty_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(empty_floats.get_typed_value_builtin() == TYPE_FLOAT)

	untyped_basic = empty_floats
	assert(untyped_basic.get_typed_key_builtin() == TYPE_FLOAT)
	assert(untyped_basic.get_typed_value_builtin() == TYPE_FLOAT)

	inferred_basic = empty_floats
	assert(inferred_basic.get_typed_key_builtin() == TYPE_FLOAT)
	assert(inferred_basic.get_typed_value_builtin() == TYPE_FLOAT)

	typed_basic = empty_floats
	assert(typed_basic.get_typed_key_builtin() == TYPE_FLOAT)
	assert(typed_basic.get_typed_value_builtin() == TYPE_FLOAT)

	empty_floats[705.0] = 507.0
	untyped_basic[430.0] = 34.0
	inferred_basic[263.0] = 362.0
	typed_basic[518.0] = 815.0
	assert(str(empty_floats) == '{ 705: 507, 430: 34, 263: 362, 518: 815 }')
	assert(str(untyped_basic) == '{ 705: 507, 430: 34, 263: 362, 518: 815 }')
	assert(str(inferred_basic) == '{ 705: 507, 430: 34, 263: 362, 518: 815 }')
	assert(str(typed_basic) == '{ 705: 507, 430: 34, 263: 362, 518: 815 }')


	const constant_float := 950.0
	const constant_int := 170
	var typed_float := 954.0
	var filled_floats: Dictionary[float, float] = { constant_float: constant_int, typed_float: empty_floats[430.0] + empty_floats[263.0] }
	assert(str(filled_floats) == '{ 950: 170, 954: 396 }')
	assert(filled_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(filled_floats.get_typed_value_builtin() == TYPE_FLOAT)

	var casted_floats := { empty_floats[263.0] * 2: empty_floats[263.0] / 2 } as Dictionary[float, float]
	assert(str(casted_floats) == '{ 724: 181 }')
	assert(casted_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(casted_floats.get_typed_value_builtin() == TYPE_FLOAT)

	var returned_floats = (func () -> Dictionary[float, float]: return { 554: 455 }).call()
	assert(str(returned_floats) == '{ 554: 455 }')
	assert(returned_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(returned_floats.get_typed_value_builtin() == TYPE_FLOAT)

	var passed_floats = floats_identity({ 663.0 if randf() > 0.5 else 663.0: 366.0 if randf() <= 0.5 else 366.0 })
	assert(str(passed_floats) == '{ 663: 366 }')
	assert(passed_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(passed_floats.get_typed_value_builtin() == TYPE_FLOAT)

	var default_floats = (func (floats: Dictionary[float, float] = { 364.0: 463.0 }): return floats).call()
	assert(str(default_floats) == '{ 364: 463 }')
	assert(default_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(default_floats.get_typed_value_builtin() == TYPE_FLOAT)

	var typed_int := 556
	var converted_floats: Dictionary[float, float] = { typed_int: typed_int }
	converted_floats[498.0] = 894
	assert(str(converted_floats) == '{ 556: 556, 498: 894 }')
	assert(converted_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(converted_floats.get_typed_value_builtin() == TYPE_FLOAT)


	const constant_basic = { 228: 822 }
	assert(str(constant_basic) == '{ 228: 822 }')
	assert(constant_basic.get_typed_key_builtin() == TYPE_NIL)
	assert(constant_basic.get_typed_value_builtin() == TYPE_NIL)

	const constant_floats: Dictionary[float, float] = { constant_float - constant_basic[228] - constant_int: constant_float + constant_basic[228] + constant_int }
	assert(str(constant_floats) == '{ -42: 1942 }')
	assert(constant_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(constant_floats.get_typed_value_builtin() == TYPE_FLOAT)


	var source_floats: Dictionary[float, float] = { 999.74: 47.999 }
	untyped_basic = source_floats
	var destination_floats: Dictionary[float, float] = untyped_basic
	destination_floats[999.74] -= 0.999
	assert(str(source_floats) == '{ 999.74: 47 }')
	assert(str(untyped_basic) == '{ 999.74: 47 }')
	assert(str(destination_floats) == '{ 999.74: 47 }')
	assert(destination_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(destination_floats.get_typed_value_builtin() == TYPE_FLOAT)


	var duplicated_floats := empty_floats.duplicate()
	duplicated_floats.erase(705.0)
	duplicated_floats.erase(430.0)
	duplicated_floats.erase(518.0)
	duplicated_floats[263.0] *= 3
	assert(str(duplicated_floats) == '{ 263: 1086 }')
	assert(duplicated_floats.get_typed_key_builtin() == TYPE_FLOAT)
	assert(duplicated_floats.get_typed_value_builtin() == TYPE_FLOAT)


	var b_objects: Dictionary[int, B] = { 0: B.new(), 1: B.new() as A, 2: null }
	assert(b_objects.size() == 3)
	assert(b_objects.get_typed_value_builtin() == TYPE_OBJECT)
	assert(b_objects.get_typed_value_script() == B)

	var a_objects: Dictionary[int, A] = { 0: A.new(), 1: B.new(), 2: null, 3: b_objects[0] }
	assert(a_objects.size() == 4)
	assert(a_objects.get_typed_value_builtin() == TYPE_OBJECT)
	assert(a_objects.get_typed_value_script() == A)

	var a_passed = (func check_a_passing(a_objects: Dictionary[int, A]): return a_objects.size()).call(a_objects)
	assert(a_passed == 4)

	var b_passed = (func check_b_passing(basic: Dictionary): return basic[0] != null).call(b_objects)
	assert(b_passed == true)


	var empty_strings: Dictionary[String, String] = {}
	var empty_bools: Dictionary[bool, bool] = {}
	var empty_basic_one := {}
	var empty_basic_two := {}
	assert(empty_strings == empty_bools)
	assert(empty_basic_one == empty_basic_two)
	assert(empty_strings.hash() == empty_bools.hash())
	assert(empty_basic_one.hash() == empty_basic_two.hash())


	var assign_source: Dictionary[int, int] = { 527: 725 }
	var assign_target: Dictionary[int, int] = {}
	assign_target.assign(assign_source)
	assert(str(assign_source) == '{ 527: 725 }')
	assert(str(assign_target) == '{ 527: 725 }')
	assign_source[657] = 756
	assert(str(assign_source) == '{ 527: 725, 657: 756 }')
	assert(str(assign_target) == '{ 527: 725 }')


	var defaults_passed = (func check_defaults_passing(one: Dictionary[int, int] = {}, two := one):
		one[887] = 788
		two[198] = 891
		assert(str(one) == '{ 887: 788, 198: 891 }')
		assert(str(two) == '{ 887: 788, 198: 891 }')
		two = {130: 31}
		assert(str(one) == '{ 887: 788, 198: 891 }')
		assert(str(two) == '{ 130: 31 }')
		return true
	).call()
	assert(defaults_passed == true)


	var members := Members.new()
	var members_passed := members.check_passing()
	assert(members_passed == true)


	var typed_enums: Dictionary[E, E] = {}
	typed_enums[E.E0] = E.E1
	assert(str(typed_enums) == '{ 391: 193 }')
	assert(typed_enums.get_typed_key_builtin() == TYPE_INT)
	assert(typed_enums.get_typed_value_builtin() == TYPE_INT)

	const const_enums: Dictionary[E, E] = {}
	assert(const_enums.get_typed_key_builtin() == TYPE_INT)
	assert(const_enums.get_typed_key_class_name() == &'')
	assert(const_enums.get_typed_value_builtin() == TYPE_INT)
	assert(const_enums.get_typed_value_class_name() == &'')


	var a := A.new()
	var b := B.new()
	var typed_natives: Dictionary[RefCounted, RefCounted] = { a: b }
	var typed_scripts = Dictionary(typed_natives, TYPE_OBJECT, "RefCounted", A, TYPE_OBJECT, "RefCounted", B)
	assert(typed_scripts[a] == b)


	print('ok')
