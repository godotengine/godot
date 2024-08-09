class A: pass
class B extends A: pass

enum E { E0 = 391 }

func floats_identity(floats: Set[float]): return floats

class Members:
	var one: Set[int] = {104}
	var two: Set[int] = one

	func check_passing() -> bool:
		assert(str(one) == '{104}')
		assert(str(two) == '{104}')
		two.add(582)
		assert(str(one) == '{104, 582}')
		assert(str(two) == '{104, 582}')
		two = {486}
		assert(str(one) == '{104, 582}')
		assert(str(two) == '{486}')
		return true


@warning_ignore("unsafe_method_access")
@warning_ignore("assert_always_true")
@warning_ignore("return_value_discarded")
func test():
	var untyped_basic = {459}
	assert(str(untyped_basic) == '{459}')
	assert(untyped_basic.get_typed_builtin() == TYPE_NIL)

	var inferred_basic := {366}
	assert(str(inferred_basic) == '{366}')
	assert(inferred_basic.get_typed_builtin() == TYPE_NIL)

	var typed_basic: Set = {521}
	assert(str(typed_basic) == '{521}')
	assert(typed_basic.get_typed_builtin() == TYPE_NIL)


	var empty_floats: Set[float] = Set()
	assert(str(empty_floats) == 'Set()')
	assert(empty_floats.get_typed_builtin() == TYPE_FLOAT)

	untyped_basic = empty_floats
	assert(untyped_basic.get_typed_builtin() == TYPE_FLOAT)

	inferred_basic = empty_floats
	assert(inferred_basic.get_typed_builtin() == TYPE_FLOAT)

	typed_basic = empty_floats
	assert(typed_basic.get_typed_builtin() == TYPE_FLOAT)

	empty_floats.add(705.0)
	untyped_basic.add(430.0)
	inferred_basic.add(263.0)
	typed_basic.add(518.0)
	assert(str(empty_floats) == '{705, 430, 263, 518}')
	assert(str(untyped_basic) == '{705, 430, 263, 518}')
	assert(str(inferred_basic) == '{705, 430, 263, 518}')
	assert(str(typed_basic) == '{705, 430, 263, 518}')


	const constant_float := 950.0
	const constant_int := 170
	var typed_float := 954.0
	var filled_floats: Set[float] = {constant_float, constant_int, typed_float, empty_floats.intersected({430.0}).values()[0] + empty_floats.intersected({263.0}).values()[0]}
	assert(str(filled_floats) == '{950, 170, 954, 693}')
	assert(filled_floats.get_typed_builtin() == TYPE_FLOAT)

	var casted_floats := {empty_floats.intersected({263.0}).values()[0] * 2} as Set[float]
	assert(str(casted_floats) == '{526}')
	assert(casted_floats.get_typed_builtin() == TYPE_FLOAT)

	var returned_floats = (func () -> Set[float]: return {554}).call()
	assert(str(returned_floats) == '{554}')
	assert(returned_floats.get_typed_builtin() == TYPE_FLOAT)

	var passed_floats = floats_identity({663.0 if randf() > 0.5 else 663.0})
	assert(str(passed_floats) == '{663}')
	assert(passed_floats.get_typed_builtin() == TYPE_FLOAT)

	var default_floats = (func (floats: Set[float] = {364.0}): return floats).call()
	assert(str(default_floats) == '{364}')
	assert(default_floats.get_typed_builtin() == TYPE_FLOAT)

	var typed_int := 556
	var converted_floats: Set[float] = {typed_int}
	converted_floats.add(498)
	assert(str(converted_floats) == '{556, 498}')
	assert(converted_floats.get_typed_builtin() == TYPE_FLOAT)


	# const constant_basic = {228}
	# assert(str(constant_basic) == '{228}')
	# assert(constant_basic.get_typed_builtin() == TYPE_NIL)

	# const constant_floats: Set[float] = {constant_float - constant_basic.values()[0] - constant_int}
	# assert(str(constant_floats) == '{552}')
	# assert(constant_floats.get_typed_builtin() == TYPE_FLOAT)


	var source_floats: Set[float] = {999.74}
	untyped_basic = source_floats
	var destination_floats: Set[float] = untyped_basic
	assert(str(source_floats) == '{999.74}')
	assert(str(untyped_basic) == '{999.74}')
	assert(str(destination_floats) == '{999.74}')
	assert(destination_floats.get_typed_builtin() == TYPE_FLOAT)

	var b_objects: Set[B] = {B.new(), B.new() as A, null}
	assert(b_objects.size() == 3)
	assert(b_objects.get_typed_builtin() == TYPE_OBJECT)
	assert(b_objects.get_typed_script() == B)

	var a_objects: Set[A] = {A.new(), B.new(), null, b_objects.values()[0]}
	assert(a_objects.size() == 4)
	assert(a_objects.get_typed_builtin() == TYPE_OBJECT)
	assert(a_objects.get_typed_script() == A)

	var a_passed = (func check_a_passing(p_objects: Set[A]): return p_objects.size()).call(a_objects)
	assert(a_passed == 4)

	var b_passed = (func check_b_passing(basic: Set): return basic.values()[0] != null).call(b_objects)
	assert(b_passed == true)


	var empty_strings: Set[String] = Set()
	var empty_bools: Set[bool] = Set()
	var empty_basic_one := Set()
	var empty_basic_two := Set()
	assert(empty_strings == empty_bools)
	assert(empty_basic_one == empty_basic_two)
	assert(empty_strings.hash() == empty_bools.hash())
	assert(empty_basic_one.hash() == empty_basic_two.hash())


	var assign_source: Set[int] = {527}
	var assign_target: Set[int] = Set()
	assign_target.assign(assign_source)
	assert(str(assign_source) == '{527}')
	assert(str(assign_target) == '{527}')
	assign_source.add(657)
	assert(str(assign_source) == '{527, 657}')
	assert(str(assign_target) == '{527}')


	var defaults_passed = (func check_defaults_passing(one: Set[int] = Set(), two := one):
		one.add(887)
		two.add(198)
		assert(str(one) == '{887, 198}')
		assert(str(two) == '{887, 198}')
		two = {130}
		assert(str(one) == '{887, 198}')
		assert(str(two) == '{130}')
		return true
	).call()
	assert(defaults_passed == true)


	var members := Members.new()
	var members_passed := members.check_passing()
	assert(members_passed == true)

	# const const_enums: Set[E] = Set()
	# assert(const_enums.get_typed_builtin() == TYPE_INT)
	# assert(const_enums.get_typed_class_name() == &'')

	var a := A.new()
	var typed_natives: Set[RefCounted] = {a}
	var typed_scripts = Set(typed_natives, TYPE_OBJECT, "RefCounted", A)
	assert(typed_scripts.values()[0] == a)

	print('ok')
