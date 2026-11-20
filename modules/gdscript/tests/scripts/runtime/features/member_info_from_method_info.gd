extends Node

func my_func_1(_foo: int, _bar: String) -> void:
	pass

func my_func_2(_foo: bool, _bar: StringName, _baz: NodePath):
	pass

static func my_static_func_1(_foo: Node3D, _bar: bool):
	pass

static func my_static_func_2(_foo, _bar: int, _baz) -> String:
	return ""

@rpc
func my_rpc_func_1(_foo: String, _bar: StringName) -> bool:
	return false

@rpc
func my_rpc_func_2(_foo: int, _bar: String, _baz: bool) -> StringName:
	return &""

func test():
	var builtin_callable_1 : Callable = add_to_group
	var builtin_callable_2 : Callable = find_child

	print("--- built-in methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(builtin_callable_1.get_method_info()))
	print(Utils.get_method_signature(builtin_callable_2.get_method_info()))
	print("--- built-in methods using ClassDB.class_get_method_info ---")
	print(Utils.get_method_signature(ClassDB.class_get_method_info(builtin_callable_1.get_object().get_class(), builtin_callable_1.get_method())))
	print(Utils.get_method_signature(ClassDB.class_get_method_info(builtin_callable_2.get_object().get_class(), builtin_callable_2.get_method())))
	print("--- built-in methods using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(builtin_callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(builtin_callable_2.get_method())))

	var builtin_vararg_callable_1 : Callable = call_thread_safe
	var builtin_vararg_callable_2 : Callable = rpc_id

	print("--- built-in vararg methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(builtin_vararg_callable_1.get_method_info()))
	print(Utils.get_method_signature(builtin_vararg_callable_2.get_method_info()))
	print("--- built-in vararg methods using ClassDB.class_get_method_info ---")
	print(Utils.get_method_signature(ClassDB.class_get_method_info(builtin_vararg_callable_1.get_object().get_class(), builtin_vararg_callable_1.get_method())))
	print(Utils.get_method_signature(ClassDB.class_get_method_info(builtin_vararg_callable_2.get_object().get_class(), builtin_vararg_callable_2.get_method())))
	print("--- built-in vararg methods using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(builtin_vararg_callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(builtin_vararg_callable_2.get_method())))

	var callable_1 : Callable = my_func_1
	var callable_2 : Callable = my_func_2

	print("--- plain methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(callable_1.get_method_info()))
	print(Utils.get_method_signature(callable_2.get_method_info()))
	print("--- plain methods using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(callable_1.get_object().get_class(), callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(callable_2.get_object().get_class(), callable_2.get_method()) == Dictionary()) # Should print true
	print("--- plain methods using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(callable_2.get_method())))

	var static_callable_1 : Callable = my_static_func_1
	var static_callable_2 : Callable = my_static_func_2

	print("--- static methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(static_callable_1.get_method_info()))
	print(Utils.get_method_signature(static_callable_2.get_method_info()))
	print("--- static methods using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(static_callable_1.get_object().get_class(), static_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(static_callable_2.get_object().get_class(), static_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- static methods using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(static_callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(static_callable_2.get_method())))

	var rpc_callable_1 : Callable = my_rpc_func_1
	var rpc_callable_2 : Callable = my_rpc_func_2

	print("--- rpc methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(rpc_callable_1.get_method_info()))
	print(Utils.get_method_signature(rpc_callable_2.get_method_info()))
	print("--- rpc methods using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(rpc_callable_1.get_object().get_class(), rpc_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(rpc_callable_2.get_object().get_class(), rpc_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- rpc methods using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(rpc_callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(rpc_callable_2.get_method())))

	var lambda_callable_1 : Callable = func(_foo, _bar): pass
	var lambda_callable_2 : Callable = func(_foo, _bar, _baz): pass

	print("--- lambdas using Callable.get_method_info ---")
	print(Utils.get_method_signature(lambda_callable_1.get_method_info()))
	print(Utils.get_method_signature(lambda_callable_2.get_method_info()))
	print("--- lambdas using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(lambda_callable_1.get_object().get_class(), lambda_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(lambda_callable_2.get_object().get_class(), lambda_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- lambdas using Object.get_method_info ---")
	print(self.get_method_info(lambda_callable_1.get_method()) == Dictionary()) # Should print true
	print(self.get_method_info(lambda_callable_2.get_method()) == Dictionary()) # Should print true

	var lambda_self_callable_1 : Callable = func(_foo, _bar): return self
	var lambda_self_callable_2 : Callable = func(_foo, _bar, _baz): return self

	print("--- lambdas with self using Callable.get_method_info ---")
	print(Utils.get_method_signature(lambda_self_callable_1.get_method_info()))
	print(Utils.get_method_signature(lambda_self_callable_2.get_method_info()))
	print("--- lambdas with self using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(lambda_self_callable_1.get_object().get_class(), lambda_self_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(lambda_self_callable_2.get_object().get_class(), lambda_self_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- lambdas with self using Object.get_method_info ---")
	print(self.get_method_info(lambda_self_callable_1.get_method()) == Dictionary()) # Should print true
	print(self.get_method_info(lambda_self_callable_2.get_method()) == Dictionary()) # Should print true

	var bind_callable_1 : Callable = my_func_2.bind(1)
	var bind_callable_2 : Callable = my_func_2.bind(1, 2)

	print("--- bind using Callable.get_method_info ---")
	print(Utils.get_method_signature(bind_callable_1.get_method_info()))
	print(Utils.get_method_signature(bind_callable_2.get_method_info()))
	print("--- bind using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(bind_callable_1.get_object().get_class(), bind_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(bind_callable_2.get_object().get_class(), bind_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- bind using Object.get_method_info ---")
	print(Utils.get_method_signature(self.get_method_info(bind_callable_1.get_method())))
	print(Utils.get_method_signature(self.get_method_info(bind_callable_2.get_method())))

	var unbind_callable_1 : Callable = my_func_2.unbind(1)
	var unbind_callable_2 : Callable = my_func_2.unbind(2)

	print("--- unbind using Callable.get_method_info ---")
	print(Utils.get_method_signature(unbind_callable_1.get_method_info()))
	print(Utils.get_method_signature(unbind_callable_2.get_method_info()))
	print("--- unbind using ClassDB.class_get_method_info ---")
	print(ClassDB.class_get_method_info(unbind_callable_1.get_object().get_class(), unbind_callable_1.get_method()) == Dictionary()) # Should print true
	print(ClassDB.class_get_method_info(unbind_callable_2.get_object().get_class(), unbind_callable_2.get_method()) == Dictionary()) # Should print true
	print("--- unbind using Object.get_method_info ---")
	print(Utils.get_method_signature(unbind_callable_1.get_method_info()))
	print(Utils.get_method_signature(unbind_callable_2.get_method_info()))

	var string_tmp := String()
	var variant_callable_1 : Callable = string_tmp.replace
	var variant_callable_2 : Callable = string_tmp.rsplit

	# ClassDB.class_get_method_info can't be used for variant class methods
	# Object.get_method_info can't be used for variant class methods
	print("--- variant callables using Callable.get_method_info ---")
	print(Utils.get_method_signature(variant_callable_1.get_method_info()))
	print(Utils.get_method_signature(variant_callable_2.get_method_info()))

	var callable_tmp := Callable()
	var variant_vararg_callable_1 : Callable = callable_tmp.call
	var variant_vararg_callable_2 : Callable = callable_tmp.rpc_id

	# ClassDB.class_get_method_info can't be used for variant class methods
	# Object.get_method_info can't be used for variant class methods
	print("--- variant vararg using Callable.get_method_info ---")
	print(Utils.get_method_signature(variant_vararg_callable_1.get_method_info()))
	print(Utils.get_method_signature(variant_vararg_callable_2.get_method_info()))

	var global_callable_1 = is_equal_approx
	var global_callable_2 = inverse_lerp

	# ClassDB.class_get_method_info can't be used for Global methods
	print("--- global methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(global_callable_1.get_method_info()))
	print(Utils.get_method_signature(global_callable_2.get_method_info()))
	print("--- global methods using Object.get_method_info ---")
	print(self.get_method_info(global_callable_1.get_method()) == Dictionary()) # Should print true
	print(self.get_method_info(global_callable_2.get_method()) == Dictionary()) # Should print true

	var gdscript_callable_1 = char
	var gdscript_callable_2 = is_instance_of

	# ClassDB.class_get_method_info can't be used for GDScript methods
	print("--- GDScript methods using Callable.get_method_info ---")
	print(Utils.get_method_signature(gdscript_callable_1.get_method_info()))
	print(Utils.get_method_signature(gdscript_callable_2.get_method_info()))
	print("--- GDScript methods using Object.get_method_info ---")
	print(self.get_method_info(gdscript_callable_1.get_method()) == Dictionary()) # Should print true
	print(self.get_method_info(gdscript_callable_2.get_method()) == Dictionary()) # Should print true
