/* godot-cpp integration testing project.
 *
 * This is free and unencumbered software released into the public domain.
 */

#pragma once

// We don't need windows.h in this example plugin but many others do, and it can
// lead to annoying situations due to the ton of macros it defines.
// So we include it and make sure CI warns us if we use something that conflicts
// with a Windows define.
#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/input_event_key.hpp>
#include <godot_cpp/classes/tile_map.hpp>
#include <godot_cpp/classes/tile_set.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/typed_dictionary.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/variant_internal.hpp>

#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

using namespace godot;

class ExampleInternal;

class ExampleRef : public RefCounted {
	GDCLASS(ExampleRef, RefCounted);

private:
	static int instance_count;
	static int last_id;

	int id;
	bool post_initialized = false;

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	ExampleRef();
	~ExampleRef();

	void set_id(int p_id);
	int get_id() const;

	bool was_post_initialized() const { return post_initialized; }
};

class ExampleMin : public Control {
	GDCLASS(ExampleMin, Control);

protected:
	static void _bind_methods() {}
};

class Example : public Control {
	GDCLASS(Example, Control);

protected:
	static void _bind_methods();

	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _property_can_revert(const StringName &p_name) const;
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const;
	void _validate_property(PropertyInfo &p_property) const;

	String _to_string() const;

private:
	Vector2 custom_position;
	Vector3 property_from_list;
	Vector2 dprop[3];
	int last_rpc_arg = 0;

	const bool object_instance_binding_set_by_parent_constructor;
	bool has_object_instance_binding() const;

public:
	// Constants.
	enum Constants {
		FIRST,
		ANSWER_TO_EVERYTHING = 42,
	};

	enum Flags {
		FLAG_ONE = 1,
		FLAG_TWO = 2,
	};

	enum {
		CONSTANT_WITHOUT_ENUM = 314,
	};

	Example();
	~Example();

	// Functions.
	void simple_func();
	void simple_const_func() const;
	int custom_ref_func(Ref<ExampleRef> p_ref);
	int custom_const_ref_func(const Ref<ExampleRef> &p_ref);
	String image_ref_func(Ref<Image> p_image);
	String image_const_ref_func(const Ref<Image> &p_image);
	String return_something(const String &base);
	Viewport *return_something_const() const;
	Ref<ExampleRef> return_ref() const;
	Ref<ExampleRef> return_empty_ref() const;
	ExampleRef *return_extended_ref() const;
	Ref<ExampleRef> extended_ref_checks(Ref<ExampleRef> p_ref) const;
	Variant varargs_func(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int varargs_func_nv(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	void varargs_func_void(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	void emit_custom_signal(const String &name, int value);
	int def_args(int p_a = 100, int p_b = 200);

	bool is_object_binding_set_by_parent_constructor() const;

	Array test_array() const;
	int test_tarray_arg(const TypedArray<int64_t> &p_array);
	TypedArray<Vector2> test_tarray() const;
	Dictionary test_dictionary() const;
	int test_tdictionary_arg(const TypedDictionary<String, int64_t> &p_dictionary);
	TypedDictionary<Vector2, Vector2i> test_tdictionary() const;
	Example *test_node_argument(Example *p_node) const;
	String test_string_ops() const;
	String test_str_utility() const;
	bool test_string_is_forty_two(const String &p_str) const;
	String test_string_resize(String p_original) const;
	TypedArray<PackedInt32Array> test_typed_array_of_packed() const;
	int test_vector_ops() const;
	int test_vector_init_list() const;

	bool test_object_cast_to_node(Object *p_object) const;
	bool test_object_cast_to_control(Object *p_object) const;
	bool test_object_cast_to_example(Object *p_object) const;

	Vector2i test_variant_vector2i_conversion(const Variant &p_variant) const;
	int test_variant_int_conversion(const Variant &p_variant) const;
	float test_variant_float_conversion(const Variant &p_variant) const;
	bool test_object_is_valid(const Variant &p_variant) const;

	void test_add_child(Node *p_node);
	void test_set_tileset(TileMap *p_tilemap, const Ref<TileSet> &p_tileset) const;

	Variant test_variant_call(Variant p_variant);

	Callable test_callable_mp();
	Callable test_callable_mp_ret();
	Callable test_callable_mp_retc() const;
	Callable test_callable_mp_static() const;
	Callable test_callable_mp_static_ret() const;
	Callable test_custom_callable() const;

	void unbound_method1(Object *p_object, String p_string, int p_int);
	String unbound_method2(Object *p_object, String p_string, int p_int);
	String unbound_method3(Object *p_object, String p_string, int p_int) const;
	static void unbound_static_method1(Example *p_object, String p_string, int p_int);
	static String unbound_static_method2(Object *p_object, String p_string, int p_int);

	BitField<Flags> test_bitfield(BitField<Flags> flags);

	Variant test_variant_iterator(const Variant &p_input);

	// RPC
	void test_rpc(int p_value);
	void test_send_rpc(int p_value);
	int return_last_rpc_arg();

	void callable_bind();

	// Property.
	void set_custom_position(const Vector2 &pos);
	Vector2 get_custom_position() const;
	Vector4 get_v4() const;

	bool test_post_initialize() const;

	int64_t test_get_internal(const Variant &p_input) const;

	// Static method.
	static int test_static(int p_a, int p_b);
	static void test_static2();

	// Virtual function override (no need to bind manually).
	virtual bool _has_point(const Vector2 &point) const override;
	virtual void _input(const Ref<InputEvent> &event) override;

	GDVIRTUAL2R(String, _do_something_virtual, String, int);
	String test_virtual_implemented_in_script(const String &p_name, int p_value);
	GDVIRTUAL1(_do_something_virtual_with_control, Control *);

	String test_use_engine_singleton() const;

	static String test_library_path();

	Ref<RefCounted> test_get_internal_class() const;
};

VARIANT_ENUM_CAST(Example::Constants);
VARIANT_BITFIELD_CAST(Example::Flags);

enum EnumWithoutClass {
	OUTSIDE_OF_CLASS = 512
};
VARIANT_ENUM_CAST(EnumWithoutClass);

class ExampleVirtual : public Object {
	GDCLASS(ExampleVirtual, Object);

protected:
	static void _bind_methods() {}
};

class ExampleAbstractBase : public Object {
	GDCLASS(ExampleAbstractBase, Object);

protected:
	static void _bind_methods() {}

	virtual int test_function() = 0;
};

class ExampleConcrete : public ExampleAbstractBase {
	GDCLASS(ExampleConcrete, ExampleAbstractBase);

protected:
	static void _bind_methods() {}

	virtual int test_function() override { return 25; }
};

class ExampleBase : public Node {
	GDCLASS(ExampleBase, Node);

protected:
	int value1 = 0;
	int value2 = 0;

	static void _bind_methods();

	void _notification(int p_what);

public:
	int get_value1() { return value1; }
	int get_value2() { return value2; }
};

class ExampleChild : public ExampleBase {
	GDCLASS(ExampleChild, ExampleBase);

protected:
	static void _bind_methods() {}

	void _notification(int p_what);
};

class ExampleRuntime : public Node {
	GDCLASS(ExampleRuntime, Node);

	int prop_value = 12;

protected:
	static void _bind_methods();

public:
	void set_prop_value(int p_prop_value);
	int get_prop_value() const;

	ExampleRuntime();
	~ExampleRuntime();
};

class ExamplePrzykład : public RefCounted {
	GDCLASS(ExamplePrzykład, RefCounted);

protected:
	static void _bind_methods();

public:
	String get_the_word() const;
};

class ExampleInternal : public RefCounted {
	GDCLASS(ExampleInternal, RefCounted);

protected:
	static void _bind_methods();

public:
	int get_the_answer() const;
};
