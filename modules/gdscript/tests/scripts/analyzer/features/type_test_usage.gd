class A extends RefCounted:
	pass

class B extends A:
	pass

@warning_ignore("assert_always_true")
func test():
	var builtin: Variant = 3
	assert((builtin is Variant) == true)
	assert((builtin is int) == true)
	assert(is_instance_of(builtin, TYPE_INT) == true)
	assert((builtin is float) == false)
	assert(is_instance_of(builtin, TYPE_FLOAT) == false)

	const const_builtin: Variant = 3
	assert((const_builtin is Variant) == true)
	assert((const_builtin is int) == true)
	assert(is_instance_of(const_builtin, TYPE_INT) == true)
	assert((const_builtin is float) == false)
	assert(is_instance_of(const_builtin, TYPE_FLOAT) == false)

	var int_array: Variant = [] as Array[int]
	assert((int_array is Variant) == true)
	assert((int_array is Array) == true)
	assert(is_instance_of(int_array, TYPE_ARRAY) == true)
	assert((int_array is Array[int]) == true)
	assert((int_array is Array[float]) == false)
	assert((int_array is int) == false)
	assert(is_instance_of(int_array, TYPE_INT) == false)

	var const_int_array: Variant = [] as Array[int]
	assert((const_int_array is Variant) == true)
	assert((const_int_array is Array) == true)
	assert(is_instance_of(const_int_array, TYPE_ARRAY) == true)
	assert((const_int_array is Array[int]) == true)
	assert((const_int_array is Array[float]) == false)
	assert((const_int_array is int) == false)
	assert(is_instance_of(const_int_array, TYPE_INT) == false)

	var b_array: Variant = [] as Array[B]
	assert((b_array is Variant) == true)
	assert((b_array is Array) == true)
	assert(is_instance_of(b_array, TYPE_ARRAY) == true)
	assert((b_array is Array[B]) == true)
	assert((b_array is Array[A]) == false)
	assert((b_array is Array[int]) == false)
	assert((b_array is int) == false)
	assert(is_instance_of(b_array, TYPE_INT) == false)

	var const_b_array: Variant = [] as Array[B]
	assert((const_b_array is Variant) == true)
	assert((const_b_array is Array) == true)
	assert(is_instance_of(const_b_array, TYPE_ARRAY) == true)
	assert((const_b_array is Array[B]) == true)
	assert((const_b_array is Array[A]) == false)
	assert((const_b_array is Array[int]) == false)
	assert((const_b_array is int) == false)
	assert(is_instance_of(const_b_array, TYPE_INT) == false)

	var native: Variant = RefCounted.new()
	assert((native is Variant) == true)
	assert((native is Object) == true)
	assert(is_instance_of(native, TYPE_OBJECT) == true)
	assert(is_instance_of(native, Object) == true)
	assert((native is RefCounted) == true)
	assert(is_instance_of(native, RefCounted) == true)
	assert((native is Node) == false)
	assert(is_instance_of(native, Node) == false)
	assert((native is int) == false)
	assert(is_instance_of(native, TYPE_INT) == false)

	var a_script: Variant = A.new()
	assert((a_script is Variant) == true)
	assert((a_script is Object) == true)
	assert(is_instance_of(a_script, TYPE_OBJECT) == true)
	assert(is_instance_of(a_script, Object) == true)
	assert((a_script is RefCounted) == true)
	assert(is_instance_of(a_script, RefCounted) == true)
	assert((a_script is A) == true)
	assert(is_instance_of(a_script, A) == true)
	assert((a_script is B) == false)
	assert(is_instance_of(a_script, B) == false)
	assert((a_script is Node) == false)
	assert(is_instance_of(a_script, Node) == false)
	assert((a_script is int) == false)
	assert(is_instance_of(a_script, TYPE_INT) == false)

	var b_script: Variant = B.new()
	assert((b_script is Variant) == true)
	assert((b_script is Object) == true)
	assert(is_instance_of(b_script, TYPE_OBJECT) == true)
	assert(is_instance_of(b_script, Object) == true)
	assert((b_script is RefCounted) == true)
	assert(is_instance_of(b_script, RefCounted) == true)
	assert((b_script is A) == true)
	assert(is_instance_of(b_script, A) == true)
	assert((b_script is B) == true)
	assert(is_instance_of(b_script, B) == true)
	assert((b_script is Node) == false)
	assert(is_instance_of(b_script, Node) == false)
	assert((b_script is int) == false)
	assert(is_instance_of(b_script, TYPE_INT) == false)

	var var_null: Variant = null
	assert((var_null is Variant) == true)
	assert((var_null is int) == false)
	assert(is_instance_of(var_null, TYPE_INT) == false)
	assert((var_null is Object) == false)
	assert(is_instance_of(var_null, TYPE_OBJECT) == false)
	assert((var_null is RefCounted) == false)
	assert(is_instance_of(var_null, RefCounted) == false)
	assert((var_null is A) == false)
	assert(is_instance_of(var_null, A) == false)

	const const_null: Variant = null
	assert((const_null is Variant) == true)
	assert((const_null is int) == false)
	assert(is_instance_of(const_null, TYPE_INT) == false)
	assert((const_null is Object) == false)
	assert(is_instance_of(const_null, TYPE_OBJECT) == false)
	assert((const_null is RefCounted) == false)
	assert(is_instance_of(const_null, RefCounted) == false)
	assert((const_null is A) == false)
	assert(is_instance_of(const_null, A) == false)

	print('ok')
