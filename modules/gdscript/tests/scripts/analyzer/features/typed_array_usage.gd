class A: pass
class B extends A: pass

enum E { E0 = 391 }

func floats_identity(floats: Array[float]): return floats

class Members:
	var one: Array[int] = [104]
	var two: Array[int] = one

	func check_passing() -> bool:
		assert(str(one) == '[104]')
		assert(str(two) == '[104]')
		two.push_back(582)
		assert(str(one) == '[104, 582]')
		assert(str(two) == '[104, 582]')
		two = [486]
		assert(str(one) == '[104, 582]')
		assert(str(two) == '[486]')
		return true


@warning_ignore("unsafe_method_access")
@warning_ignore("assert_always_true")
@warning_ignore("return_value_discarded")
func test():
	var untyped_basic = [459]
	assert(str(untyped_basic) == '[459]')
	assert(untyped_basic.get_typed_builtin() == TYPE_NIL)

	var inferred_basic := [366]
	assert(str(inferred_basic) == '[366]')
	assert(inferred_basic.get_typed_builtin() == TYPE_NIL)

	var typed_basic: Array = [521]
	assert(str(typed_basic) == '[521]')
	assert(typed_basic.get_typed_builtin() == TYPE_NIL)


	var empty_floats: Array[float] = []
	assert(str(empty_floats) == '[]')
	assert(empty_floats.get_typed_builtin() == TYPE_FLOAT)

	untyped_basic = empty_floats
	assert(untyped_basic.get_typed_builtin() == TYPE_FLOAT)

	inferred_basic = empty_floats
	assert(inferred_basic.get_typed_builtin() == TYPE_FLOAT)

	typed_basic = empty_floats
	assert(typed_basic.get_typed_builtin() == TYPE_FLOAT)

	empty_floats.push_back(705.0)
	untyped_basic.push_back(430.0)
	inferred_basic.push_back(263.0)
	typed_basic.push_back(518.0)
	assert(str(empty_floats) == '[705, 430, 263, 518]')
	assert(str(untyped_basic) == '[705, 430, 263, 518]')
	assert(str(inferred_basic) == '[705, 430, 263, 518]')
	assert(str(typed_basic) == '[705, 430, 263, 518]')


	const constant_float := 950.0
	const constant_int := 170
	var typed_float := 954.0
	var filled_floats: Array[float] = [constant_float, constant_int, typed_float, empty_floats[1] + empty_floats[2]]
	assert(str(filled_floats) == '[950, 170, 954, 693]')
	assert(filled_floats.get_typed_builtin() == TYPE_FLOAT)

	var casted_floats := [empty_floats[2] * 2] as Array[float]
	assert(str(casted_floats) == '[526]')
	assert(casted_floats.get_typed_builtin() == TYPE_FLOAT)

	var returned_floats = (func () -> Array[float]: return [554]).call()
	assert(str(returned_floats) == '[554]')
	assert(returned_floats.get_typed_builtin() == TYPE_FLOAT)

	var passed_floats = floats_identity([663.0 if randf() > 0.5 else 663.0])
	assert(str(passed_floats) == '[663]')
	assert(passed_floats.get_typed_builtin() == TYPE_FLOAT)

	var default_floats = (func (floats: Array[float] = [364.0]): return floats).call()
	assert(str(default_floats) == '[364]')
	assert(default_floats.get_typed_builtin() == TYPE_FLOAT)

	var typed_int := 556
	var converted_floats: Array[float] = [typed_int]
	converted_floats.push_back(498)
	assert(str(converted_floats) == '[556, 498]')
	assert(converted_floats.get_typed_builtin() == TYPE_FLOAT)


	const constant_basic = [228]
	assert(str(constant_basic) == '[228]')
	assert(constant_basic.get_typed_builtin() == TYPE_NIL)

	const constant_floats: Array[float] = [constant_float - constant_basic[0] - constant_int]
	assert(str(constant_floats) == '[552]')
	assert(constant_floats.get_typed_builtin() == TYPE_FLOAT)


	var source_floats: Array[float] = [999.74]
	untyped_basic = source_floats
	var destination_floats: Array[float] = untyped_basic
	destination_floats[0] -= 0.74
	assert(str(source_floats) == '[999]')
	assert(str(untyped_basic) == '[999]')
	assert(str(destination_floats) == '[999]')
	assert(destination_floats.get_typed_builtin() == TYPE_FLOAT)


	var duplicated_floats := empty_floats.duplicate().slice(2, 3)
	duplicated_floats[0] *= 3
	assert(str(duplicated_floats) == '[789]')
	assert(duplicated_floats.get_typed_builtin() == TYPE_FLOAT)


	var b_objects: Array[B] = [B.new(), B.new() as A, null]
	assert(b_objects.size() == 3)
	assert(b_objects.get_typed_builtin() == TYPE_OBJECT)
	assert(b_objects.get_typed_script() == B)

	var a_objects: Array[A] = [A.new(), B.new(), null, b_objects[0]]
	assert(a_objects.size() == 4)
	assert(a_objects.get_typed_builtin() == TYPE_OBJECT)
	assert(a_objects.get_typed_script() == A)

	var a_passed = (func check_a_passing(p_objects: Array[A]): return p_objects.size()).call(a_objects)
	assert(a_passed == 4)

	var b_passed = (func check_b_passing(basic: Array): return basic[0] != null).call(b_objects)
	assert(b_passed == true)


	var empty_strings: Array[String] = []
	var empty_bools: Array[bool] = []
	var empty_basic_one := []
	var empty_basic_two := []
	assert(empty_strings == empty_bools)
	assert(empty_basic_one == empty_basic_two)
	assert(empty_strings.hash() == empty_bools.hash())
	assert(empty_basic_one.hash() == empty_basic_two.hash())


	var assign_source: Array[int] = [527]
	var assign_target: Array[int] = []
	assign_target.assign(assign_source)
	assert(str(assign_source) == '[527]')
	assert(str(assign_target) == '[527]')
	assign_source.push_back(657)
	assert(str(assign_source) == '[527, 657]')
	assert(str(assign_target) == '[527]')


	var defaults_passed = (func check_defaults_passing(one: Array[int] = [], two := one):
		one.push_back(887)
		two.push_back(198)
		assert(str(one) == '[887, 198]')
		assert(str(two) == '[887, 198]')
		two = [130]
		assert(str(one) == '[887, 198]')
		assert(str(two) == '[130]')
		return true
	).call()
	assert(defaults_passed == true)


	var members := Members.new()
	var members_passed := members.check_passing()
	assert(members_passed == true)


	var resized_basic: Array = []
	resized_basic.resize(1)
	assert(typeof(resized_basic[0]) == TYPE_NIL)
	assert(resized_basic[0] == null)

	var resized_ints: Array[int] = []
	resized_ints.resize(1)
	assert(typeof(resized_ints[0]) == TYPE_INT)
	assert(resized_ints[0] == 0)

	var resized_arrays: Array[Array] = []
	resized_arrays.resize(1)
	assert(typeof(resized_arrays[0]) == TYPE_ARRAY)
	resized_arrays[0].resize(1)
	resized_arrays[0][0] = 523
	assert(str(resized_arrays) == '[[523]]')

	var resized_objects: Array[Object] = []
	resized_objects.resize(1)
	assert(typeof(resized_objects[0]) == TYPE_NIL)
	assert(resized_objects[0] == null)


	var typed_enums: Array[E] = []
	typed_enums.resize(1)
	assert(str(typed_enums) == '[0]')
	typed_enums[0] = E.E0
	assert(str(typed_enums) == '[391]')
	assert(typed_enums.get_typed_builtin() == TYPE_INT)

	const const_enums: Array[E] = []
	assert(const_enums.get_typed_builtin() == TYPE_INT)
	assert(const_enums.get_typed_class_name() == &'')


	var a := A.new()
	var typed_natives: Array[RefCounted] = [a]
	var typed_scripts = Array(typed_natives, TYPE_OBJECT, "RefCounted", A)
	assert(typed_scripts[0] == a)


	print('ok')
