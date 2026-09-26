extends Node

func my_func_1(_foo, _bar):
	pass

func my_func_2(_foo, _bar, _baz):
	pass

static func my_static_func_1(_foo, _bar):
	pass

static func my_static_func_2(_foo, _bar, _baz):
	pass

@rpc
func my_rpc_func_1(_foo, _bar):
	pass

@rpc
func my_rpc_func_2(_foo, _bar, _baz):
	pass

func test():
	print("--- built-in methods ---")
	var builtin_callable_1 : Callable = add_to_group
	print(Utils.get_method_signature(builtin_callable_1.get_method_info()))
	var builtin_callable_2 : Callable = find_child
	print(Utils.get_method_signature(builtin_callable_2.get_method_info()))

	print("--- built-in vararg methods ---")
	var builtin_vararg_callable_1 : Callable = call_thread_safe
	print(Utils.get_method_signature(builtin_vararg_callable_1.get_method_info()))
	var builtin_vararg_callable_2 : Callable = rpc_id
	print(Utils.get_method_signature(builtin_vararg_callable_2.get_method_info()))

	print("--- plain methods ---")
	var callable_1 : Callable = my_func_1
	print(Utils.get_method_signature(callable_1.get_method_info()))
	var callable_2 : Callable = my_func_2
	print(Utils.get_method_signature(callable_2.get_method_info()))

	print("--- static methods ---")
	var static_callable_1 : Callable = my_static_func_1
	print(Utils.get_method_signature(static_callable_1.get_method_info()))
	var static_callable_2 : Callable = my_static_func_2
	print(Utils.get_method_signature(static_callable_2.get_method_info()))

	print("--- rpc methods ---")
	var rpc_callable_1 : Callable = my_rpc_func_1
	print(Utils.get_method_signature(rpc_callable_1.get_method_info()))
	var rpc_callable_2 : Callable = my_rpc_func_2
	print(Utils.get_method_signature(rpc_callable_2.get_method_info()))

	print("--- lambdas ---")
	var lambda_callable_1 : Callable = func(_foo, _bar): pass
	print(Utils.get_method_signature(lambda_callable_1.get_method_info()))
	var lambda_callable_2 : Callable = func(_foo, _bar, _baz): pass
	print(Utils.get_method_signature(lambda_callable_2.get_method_info()))

	print("--- lambdas with self ---")
	var lambda_self_callable_1 : Callable = func(_foo, _bar): return self
	print(Utils.get_method_signature(lambda_self_callable_1.get_method_info()))
	var lambda_self_callable_2 : Callable = func(_foo, _bar, _baz): return self
	print(Utils.get_method_signature(lambda_self_callable_2.get_method_info()))

	print("--- bind ---")
	var bind_callable_1 : Callable = my_func_2.bind(1)
	print(Utils.get_method_signature(bind_callable_1.get_method_info()))
	var bind_callable_2 : Callable = my_func_2.bind(1, 2)
	print(Utils.get_method_signature(bind_callable_2.get_method_info()))

	print("--- unbind ---")
	var unbind_callable_1 : Callable = my_func_2.unbind(1)
	print(Utils.get_method_signature(unbind_callable_1.get_method_info()))
	var unbind_callable_2 : Callable = my_func_2.unbind(2)
	print(Utils.get_method_signature(unbind_callable_2.get_method_info()))

	print("--- variant callables ---")
	var string_tmp := String()
	var variant_callable_1 : Callable = string_tmp.replace
	print(Utils.get_method_signature(variant_callable_1.get_method_info()))
	var variant_callable_2 : Callable = string_tmp.rsplit
	print(Utils.get_method_signature(variant_callable_2.get_method_info()))

	print("--- variant vararg callables ---")
	var callable_tmp := Callable()
	var variant_vararg_callable_1 : Callable = callable_tmp.call
	print(Utils.get_method_signature(variant_vararg_callable_1.get_method_info()))
	var variant_vararg_callable_2 : Callable = callable_tmp.rpc_id
	print(Utils.get_method_signature(variant_vararg_callable_2.get_method_info()))

	print("--- global methods ---")
	var global_callable_1 = is_equal_approx
	print(Utils.get_method_signature(global_callable_1.get_method_info()))
	var global_callable_2 = inverse_lerp
	print(Utils.get_method_signature(global_callable_2.get_method_info()))

	print("--- GDScript methods ---")
	var gdscript_callable_1 = char
	print(Utils.get_method_signature(gdscript_callable_1.get_method_info()))
	var gdscript_callable_2 = is_instance_of
	print(Utils.get_method_signature(gdscript_callable_2.get_method_info()))
