#include "api.hpp"

struct MyException : public std::exception {
	using std::exception::exception;
	const char *what() const noexcept override {
		return "This is a test exception";
	}
};

PUBLIC Variant test_exceptions() {
#ifdef ZIG_COMPILER
#warning "Zig does not support exceptions (yet)"
	return "This is a test exception";
#else
	try {
		throw MyException();
	} catch (const std::exception &e) {
		return e.what();
	}
	return "";
#endif
}

// This works: it's being created during initialization
static Dictionary d = Dictionary::Create();

PUBLIC Variant test_static_storage(Variant key, Variant val) {
	d[key] = val;
	return d;
}
PUBLIC Variant test_failing_static_storage(Variant key, Variant val) {
	// This works only once: it's being created after initialization
	static Dictionary fd = Dictionary::Create();
	fd[key] = val;
	return fd;
}
static Dictionary fd = Dictionary::Create();
PUBLIC Variant test_permanent_storage(Variant key, Variant val) {
	fd[key] = val;
	fd = Variant(fd).make_permanent();
	return fd;
}

static String ps = "Hello this is a permanent string";
PUBLIC Variant test_permanent_string(String input) {
	ps = input;
	return ps;
}

static Array pa = Array::Create();
PUBLIC Variant test_permanent_array(Array input) {
	pa = input;
	return pa;
}

static Dictionary pd = Dictionary::Create();
PUBLIC Variant test_permanent_dict(Dictionary input) {
	pd = input;
	return pd;
}

PUBLIC Variant test_check_if_permanent(String test) {
	if (test == "string") {
		printf("Checking if string %d is permanent\n", ps.get_variant_index());
		return ps.is_permanent();
	} else if (test == "array") {
		printf("Checking if array %d is permanent\n", pa.get_variant_index());
		return pa.is_permanent();
	} else if (test == "dict") {
		printf("Checking if dictionary %d is permanent\n", pd.get_variant_index());
		return pd.is_permanent();
	}
	return false;
}

PUBLIC Variant test_infinite_loop() {
	while (true)
		;
}

PUBLIC Variant test_recursive_calls(Node sandbox) {
	sandbox("vmcall", "test_recursive_calls", sandbox);
	return {};
}

PUBLIC Variant public_function() {
	return "Hello from the other side";
}

PUBLIC Variant test_ping_pong(Variant arg) {
	return arg;
}

PUBLIC Variant test_ping_move_pong(Variant arg) {
	Variant v = std::move(arg);
	return v;
}

PUBLIC Variant test_variant_eq(Variant arg1, Variant arg2) {
	return arg1 == arg2;
}

PUBLIC Variant test_variant_neq(Variant arg1, Variant arg2) {
	return (arg1 != arg2) == false;
}

PUBLIC Variant test_variant_lt(Variant arg1, Variant arg2) {
	return arg1 < arg2;
}

PUBLIC Variant test_bool(bool arg) {
	return arg;
}

PUBLIC Variant test_int(long arg) {
	return arg;
}

PUBLIC Variant test_float(double arg) {
	return arg;
}

PUBLIC Variant test_string(String arg) {
	return arg;
}

PUBLIC Variant test_u32string(String arg) {
	std::u32string u32 = arg.utf32();
	return u32;
}

PUBLIC Variant test_nodepath(NodePath arg) {
	return arg;
}

PUBLIC Variant test_vec2(Vector2 arg) {
	Vector2 result = arg;
	return result;
}
PUBLIC Variant test_vec2i(Vector2i arg) {
	Vector2i result = arg;
	return result;
}

PUBLIC Variant test_vec3(Vector3 arg) {
	Vector3 result = arg;
	return result;
}
PUBLIC Variant test_vec3i(Vector3i arg) {
	Vector3i result = arg;
	return result;
}

PUBLIC Variant test_vec4(Vector4 arg) {
	Vector4 result = arg;
	return result;
}
PUBLIC Variant test_vec4i(Vector4i arg) {
	Vector4i result = arg;
	return result;
}

PUBLIC Variant test_color(Color arg) {
	Color result = arg;
	return result;
}

PUBLIC Variant test_plane(Plane arg) {
	Plane result = arg;
	return result;
}

PUBLIC Variant test_array(Array array) {
	array.push_back(2);
	array.push_back("4");
	array.push_back(6.0);
	if (array[0] != 2 || array[1] != "4" || array[2] != 6.0) {
		return "Fail";
	}
	if (!(array[0] == 2 && array[1] == "4" && array[2] == 6.0)) {
		return "Fail";
	}
	array[0] = 1;
	array[1] = "2";
	array[2] = 3.0;
	if (int(array[0]) != 1 || String(array[1]) != "2" || double(array[2]) != 3.0) {
		return "Fail";
	}
	if (int(array[0]) == 1 && String(array[1]) == "2" || double(array[2]) == 3.0) {
		return array;
	}
	return "Fail";
}

PUBLIC Variant test_array_assign(Array arr) {
	arr[0] = 42;
	arr[1] = "Hello";
	arr[2] = PackedArray<double> ({ 3.14, 2.71 });
	if (arr[0] != 42 || arr[1] != "Hello" || arr[2].get().get_type() != Variant::Type::PACKED_FLOAT64_ARRAY) {
		return "Fail";
	}

	Array arr2 = Array::Create();
	arr2.push_back(PackedArray<double> ({ 1.0, 2.0, 3.0 }));
	arr.push_back(arr2);
	arr[3] = arr2;

	PackedArray<double> pa = arr[2];
	if (pa.size() != 2 || pa.is_empty()) {
		return "Fail";
	}
	std::vector<double> vec = pa.fetch();
	if (vec[0] != 3.14 || vec[1] != 2.71 || vec.size() != 2) {
		return "Fail";
	}

	return arr;
}

PUBLIC Variant test_array_assign2(Array arr, const size_t idx) {

	std::vector<size_t> indices = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

	arr.resize(indices.size());
	for (size_t i = 0; i < indices.size(); i++) {
		arr[i] = indices[i];
	}

	return arr;
}

PUBLIC Variant test_dict(Dictionary arg) {
	return arg;
}

PUBLIC Variant test_sub_dictionary(Dictionary dict) {
	return Dictionary(dict["1"].value());
}

PUBLIC Variant test_rid(RID rid) {
	return rid;
}

PUBLIC Variant test_object(Object arg) {
	Object result = arg;
	return result;
}

PUBLIC Variant test_basis(Basis basis) {
	Basis b = basis;
	return b;
}

PUBLIC Variant test_transform2d(Transform2D transform2d) {
	Transform2D t2d = transform2d;
	return t2d;
}

PUBLIC Variant test_transform3d(Transform3D transform3d) {
	Transform3D t3d = transform3d;
	return t3d;
}

PUBLIC Variant test_quaternion(Quaternion quaternion) {
	Quaternion q2 = quaternion;
	return q2;
}

PUBLIC Variant test_callable(Callable callable) {
	return callable.call(1, 2, "3");
}

// clang-format off
PUBLIC Variant test_create_callable() {
	Array array = Array::Create();
	array.push_back(1);
	array.push_back(2);
	array.push_back("3");
	return Callable::Create<Variant(Array, int, int, String)>([](Array array, int a, int b, String c) -> Variant {
		return a + b + std::stoi(c.utf8()) + int(array.at(0)) + int(array.at(1)) + std::stoi(array.at(2).as_string().utf8());
	}, array);
}
// clang-format on

PUBLIC Variant test_pa_u8(PackedByteArray arr) {
	return PackedByteArray (arr.fetch());
}
PUBLIC Variant test_pa_f32(PackedArray<float> arr) {
	return PackedArray<float> (arr.fetch());
}
PUBLIC Variant test_pa_f64(PackedArray<double> arr) {
	return PackedArray<double> (arr.fetch());
}
PUBLIC Variant test_pa_i32(PackedArray<int32_t> arr) {
	return PackedArray<int32_t> (arr.fetch());
}
PUBLIC Variant test_pa_i64(PackedArray<int64_t> arr) {
	return PackedArray<int64_t> (arr.fetch());
}
PUBLIC Variant test_pa_vec2(PackedArray<Vector2> arr) {
	return PackedArray<Vector2> (arr.fetch());
}
PUBLIC Variant test_pa_vec3(PackedArray<Vector3> arr) {
	return PackedArray<Vector3> (arr.fetch());
}
PUBLIC Variant test_pa_vec4(PackedArray<Vector4> arr) {
	return PackedArray<Vector4> (arr.fetch());
}
PUBLIC Variant test_pa_color(PackedArray<Color> arr) {
	return PackedArray<Color> (arr.fetch());
}
PUBLIC Variant test_pa_string(PackedArray<std::string> arr) {
	return PackedArray<std::string> (arr.fetch());
}

PUBLIC Variant test_create_pa_u8() {
	PackedByteArray arr({ 1, 2, 3, 4 });
	return arr;
}
static const uint8_t pa_u8_data[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
PUBLIC Variant test_create_pa_u8_ptr() {
	return PackedByteArray(pa_u8_data, sizeof(pa_u8_data) / sizeof(uint8_t));
}
PUBLIC Variant test_create_pa_f32() {
	PackedArray<float> arr({ 1, 2, 3, 4 });
	return arr;
}
PUBLIC Variant test_create_pa_f64() {
	PackedArray<double> arr({ 1, 2, 3, 4 });
	return arr;
}
PUBLIC Variant test_create_pa_i32() {
	PackedArray<int32_t> arr({ 1, 2, 3, 4 });
	return arr;
}
PUBLIC Variant test_create_pa_i64() {
	PackedArray<int64_t> arr({ 1, 2, 3, 4 });
	return arr;
}
PUBLIC Variant test_create_pa_vec2() {
	PackedArray<Vector2> arr({ { 1, 1 }, { 2, 2 }, { 3, 3 } });
	return arr;
}
PUBLIC Variant test_create_pa_vec3() {
	PackedArray<Vector3> arr({ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
	return arr;
}
PUBLIC Variant test_create_pa_vec4() {
	PackedArray<Vector4> arr({ { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 3, 3, 3, 3 } });
	return arr;
}
PUBLIC Variant test_create_pa_color() {
	PackedArray<Color> arr({ { 0, 0, 0, 0 }, { 1, 1, 1, 1 } });
	return arr;
}
PUBLIC Variant test_create_pa_string() {
	PackedArray<std::string> arr({ "Hello", "from", "the", "other", "side" });
	return arr;
}

PUBLIC Variant test_assign_pa_to_array(PackedArray<int64_t> pa) {
	Array arr = Array::Create();
	arr.push_back(pa);
	arr.push_back(pa);
	return arr;
}

PUBLIC Variant test_assign_pa_to_dict(PackedArray<int64_t> arr) {
	Dictionary d = Dictionary::Create();
	d["a1"] = arr;
	d["a2"] = arr;
	return d;
}

PUBLIC Variant test_construct_pa_from_array_at(Array arr, int idx) {
	PackedArray<int64_t> pa(arr.at(idx));
	return pa;
}

PUBLIC Variant test_exception() {
	asm volatile("unimp");
	return "This should not be reached";
}

static bool timer_got_called = false;
PUBLIC Variant test_timers() {
	long val1 = 11;
	float val2 = 22.0f;
	return CallbackTimer::native_periodic(0.01, [=](Node timer) -> Variant {
		print("Timer with values: ", val1, val2);
		timer.queue_free();
		timer_got_called = true;
		return {};
	});
}
PUBLIC Variant verify_timers() {
	return timer_got_called;
}

PUBLIC Variant call_method(Variant v, Variant vmethod, Variant vargs) {
	std::string method = vmethod.as_std_string();
	Array args_array = vargs.as_array();
	std::vector<Variant> args = args_array.to_vector();
	Variant ret;
	v.callp(method, args.data(), args.size(), ret);
	return ret;
}

PUBLIC Variant voidcall_method(Variant v, Variant vmethod, Variant vargs) {
	std::string method = vmethod.as_std_string();
	Array args_array = vargs.as_array();
	std::vector<Variant> args = args_array.to_vector();
	v.voidcallp(method, args.data(), args.size());
	return Nil;
}

PUBLIC Variant access_a_parent(Node n) {
	Node p = n.get_parent();
	return p;
}

PUBLIC Variant creates_a_node() {
	return Node::Create("test");
}

PUBLIC Variant free_self() {
	get_node()("free");
	return Nil;
}

PUBLIC Variant access_an_invalid_child_node() {
	Node n = Node::Create("test");
	Node c = Node::Create("child");
	n.add_child(c);
	c("free");
	c.set_name("child2");
	return c;
}

PUBLIC Variant access_an_invalid_child_resource(String path) {
	Variant resource = loadv(path.utf8());
	return resource.method_call("instantiate");
}

PUBLIC Variant disable_restrictions() {
	get_node().call("disable_restrictions");
	return Nil;
}

PUBLIC Variant test_property_proxy() {
	Node node = Node::Create("Fail 1");
	node.name() = "Fail 1.5";
	node.set_name("Fail 2");
	if (node.get_name() == "Fail 2") {
		node.set("name", "Fail 3");
		if (node.get("name") == "Fail 3") {
			node.name() = "TestOK";
			if (node.name() != "TestOK") {
				return "Fail 4";
			}
		}
	}
	return node.get_name();
}

// This tests the higher limit for boxed arguments with up to 16 arguments
// We will pass in 10 integers and 6 strings, which we add up and return
PUBLIC Variant test_many_arguments(Variant a1, Variant a2, Variant a3, Variant a4, Variant a5, Variant a6, Variant a7, Variant a8, Variant a9, Variant a10, Variant a11, Variant a12, Variant a13, Variant a14, Variant a15, Variant a16) {
	return int(a1) + int(a2) + int(a3) + int(a4) + int(a5) + int(a6) + int(a7) + int(a8) + int(a9) + int(a10) + a11.as_string().to_int() + a12.as_string().to_int() + a13.as_string().to_int() + a14.as_string().to_int() + a15.as_string().to_int() + a16.as_string().to_int();
}

PUBLIC Variant test_many_arguments2(Variant a1, Variant a2, Variant a3, Variant a4, Variant a5, Variant a6, Variant a7, Variant a8) {
	return int(a1) + int(a2) + int(a3) + int(a4) + int(a5) + int(a6) + int(a7) + a8.as_string().to_int();
}

PUBLIC Variant test_many_unboxed_arguments(int a1, int a2, int a3, int a4, int a5, int a6, int a7, double f1, double f2, double f3, double f4) {
	return int(a1) + int(a2) + int(a3) + int(a4) + int(a5) + int(a6) + int(a7) + int(f1) + int(f2) + int(f3) + int(f4);
}

PUBLIC Variant test_many_unboxed_arguments2(int a1, int a2, int a3, int a4, int a5, int a6, int a7, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4) {
	return int(a1) + int(a2) + int(a3) + int(a4) + int(a5) + int(a6) + int(a7) + int(v1.x) + int(v1.y) + int(v2.x) + int(v2.y) + int(v3.x) + int(v3.y) + int(v4.x) + int(v4.y);
}

PUBLIC Variant get_tree_base_parent() {
	return get_parent();
}
