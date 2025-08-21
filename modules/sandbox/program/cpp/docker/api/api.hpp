/**************************************************************************/
/*  api.hpp                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once
#include <cstddef>
// Variant types
#include "array.hpp"
#include "basis.hpp"
#include "callable.hpp"
#include "dictionary.hpp"
#include "quaternion.hpp"
#include "rid.hpp"
#include "string.hpp"
#include "transform2d.hpp"
#include "transform3d.hpp"
// Objects and nodes
#include "node2d.hpp"
#include "node3d.hpp"
#include "syscalls_fwd.hpp"
#include "timer.hpp"
// Individual packed arrays
#include "packed_byte_array.hpp"

template <typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

/// @brief Print a message to the console.
/// @param ...vars A list of Variants to print.
template <typename... Args>
inline void print(Args &&...vars) {
	std::array<Variant, sizeof...(Args)> vptrs;
	int idx = 0;
	([&] {
		if constexpr (std::is_same_v<Variant, remove_cvref<Args>>)
			vptrs[idx++] = vars;
		else
			vptrs[idx++] = Variant(vars);
	}(),
			...);
	sys_print(vptrs.data(), vptrs.size());
}

/// @brief Get a node by its path. By default, this returns the current node.
/// @param path The path to the node.
/// @return The node at the given path.
template <typename T = Node>
inline T get_node(std::string_view path = ".") {
	return T(path);
}

/// @brief Get the parent of the current node.
/// @return The parent node.
template <typename T = Node>
inline T get_parent() {
	return T("..");
}

#include <unordered_map>
/// @brief A macro to define a static function that returns a custom state object
/// tied to a Node object. For shared sandbox instances, this is the simplest way
/// to store per-node-instance state.
/// @param State The type of the state object.
/// @note There is currently no way to clear the state objects, so be careful
/// with memory usage.
/// @example
/// struct SlimeState {
/// 	int direction = 1;
/// };
/// PER_OBJECT(SlimeState);
/// // Then use it like this:
/// auto& state = GetSlimeState(slime);
#define PER_OBJECT(State)                                 \
	static State &Get##State(const Node &node) {          \
		static std::unordered_map<uint64_t, State> state; \
		return state[node.address()];                     \
	}

/// @brief A property struct that must be instantiated in the global scope.
/// @note This is used to define custom properties for the Sandbox class.
/// On program load, the properties are automatically exposed on the script instance.
/// @example
/// SANDBOXED_PROPERTIES(1, {
/// 	.name = "my_property",
/// 	.type = Variant::Type::INT,
/// 	.getter = []() -> Variant { return 42; },
/// 	.setter = [](Variant value) { print("Set to: ", value); },
/// 	.default_value = Variant{42},
/// });
struct Property {
	using getter_t = Variant (*)();
	using setter_t = Variant (*)(Variant);

	const char *const name = 0;
	const unsigned size = sizeof(Property);
	const Variant::Type type;
	const getter_t getter;
	const setter_t setter;
	const Variant default_value;
};
#define SANDBOXED_PROPERTIES(num, ...) \
	extern "C" const Property properties[num + 1]{ __VA_ARGS__, { 0 } };

/// @brief Add a new property to the Sandbox class.
/// @param name The name of the property.
/// @param type The type of the property.
/// @param default_value The default value of the property.
/// @param getter A function that returns the property value.
/// @param setter A function that sets the property value.
/// @note This function must be called during the initialization phase of the program.
/// @example
/// add_property("meaning_of_life", Variant::Type::INT, 42,
/// 	[]() -> Variant { return 42; },
/// 	[](Variant value) -> Variant { print("Set to: ", value); return Nil; });
static inline void add_property(std::string_view name, Variant::Type type, const Variant &default_value, Variant (*getter)(), Variant (*setter)(Variant)) {
	sys_sandbox_add(0, name.data(), name.size(), type, setter, getter, &default_value);
}

/// @brief Add a new property to the Sandbox class. Simplified version.
/// @param name The name of the property.
/// @param type The type of the property.
/// The ADD_PROPERTY macro is a convenience macro that allows you to define a property
/// without a default value, or getter and setter. It assumes there's a global variable
/// with the same name as the property, and uses that as the getter and setter. The default
/// value is set to the current value of the variable.
/// @example
/// static double player_speed = 60.0;
/// int main() {
/// 	ADD_PROPERTY(player_speed, Variant::FLOAT);
/// }
#define ADD_PROPERTY(name, type) \
	add_property(#name, type, name, []() -> Variant { return name; }, [](Variant value) -> Variant { name = (decltype(name))value; return {}; });

/// @brief Stop execution of the program.
/// @note This function may return if the program is resumed. However, no such
/// functionality is currently implemented.
inline void halt() {
	fast_exit();
}

/// @brief Check if the program is running in the Godot editor.
/// @return True if running in the editor, false otherwise.
inline bool is_editor() {
	static constexpr int ECALL_IS_EDITOR = 512;
	register int a0 asm("a0");
	register int a7 asm("a7") = ECALL_IS_EDITOR;
	asm volatile("ecall" : "=r"(a0) : "r"(a7));
	return a0;
}
inline bool is_editor_hint() {
	return is_editor(); // Alias
}

/// @brief Load a resource (at run-time) from the given path. Can be denied.
/// @param path The path to the resource.
/// @return The loaded resource.
extern Variant loadv(std::string_view path);

/// @brief The class database for instantiating Godot objects.
struct ClassDB {
	/// @brief Instantiate a new object of the given class.
	/// @param class_name The name of the class to instantiate.
	/// @param name The name of the object, if it's a Node. Otherwise, this is ignored.
	/// @return The new object.
	static Object instantiate(std::string_view class_name, std::string_view name = "");

	template <typename T>
	static T instantiate(std::string_view class_name, std::string_view name = "") {
		return T(instantiate(class_name, name).address());
	}
};

/// @brief Math and interpolation operations.
struct Math {
	/// @brief The available 64-bit FP math operations.
	static double sin(double x);
	static double cos(double x);
	static double tan(double x);
	static double asin(double x);
	static double acos(double x);
	static double atan(double x);
	static double atan2(double y, double x);
	static double pow(double x, double y);

	/// @brief The available 32-bit FP math operations.
	static float sinf(float x);
	static float cosf(float x);
	static float tanf(float x);
	static float asinf(float x);
	static float acosf(float x);
	static float atanf(float x);
	static float atan2f(float y, float x);
	static float powf(float x, float y);

	/// @brief Linearly interpolate between two values.
	/// @param a The start value.
	/// @param b The end value.
	/// @param t The interpolation factor (between 0 and 1).
	static double lerp(double a, double b, double t);
	static float lerpf(float a, float b, float t);

	/// @brief Smoothly interpolate between two values.
	/// @param from The start value.
	/// @param to The end value.
	/// @param t The interpolation factor (between 0 and 1).
	static double smoothstep(double from, double to, double t);
	static float smoothstepf(float from, float to, float t);

	/// @brief Clamp a value between two bounds.
	/// @param x The value to clamp.
	/// @param min The minimum value.
	/// @param max The maximum value.
	static double clamp(double x, double min, double max);
	static float clampf(float x, float min, float max);

	/// @brief Spherical linear interpolation between two values.
	/// @param a The start value in radians.
	/// @param b The end value in radians.
	/// @param t The interpolation factor (between 0 and 1).
	static double slerp(double a, double b, double t);
	static float slerpf(float a, float b, float t);
};

/* Embed binary data into executable. This data has no guaranteed alignment. */
#define EMBED_BINARY(name, filename)                 \
	asm(".pushsection .rodata\n"                     \
		"	.global " #name "\n" #name ":\n"         \
		"	.incbin " #filename "\n" #name "_end:\n" \
		"	.int  0\n"                                 \
		"	.global " #name "_size\n"                \
		"	.type   " #name "_size, @object\n"       \
		"	.align 4\n" #name "_size:\n"             \
		"	.int  " #name "_end - " #name "\n"       \
		".popsection");                              \
	extern char name[];                              \
	extern unsigned name##_size;

#include "api_inline.hpp"

#if __has_include(<generated_api.hpp>)
#include <generated_api.hpp>

template <typename T = Resource>
inline T load(std::string_view path) {
	return Object(loadv(path)).address();
}

/// @brief Get the current scene tree.
/// @return The root node of the scene tree.
inline SceneTree get_tree() {
	return Object("SceneTree").address();
}

/// @brief Check if the given Node is a part of the current scene tree. Not an instance of another scene.
/// @param node The Node to check.
/// @return True if the Node is a part of the current scene tree, false otherwise.
inline bool is_part_of_tree(Node node) {
	return get_tree().get_edited_scene_root() == node.get_owner();
}
#endif

/// @brief Add a new public API function to the program during initialization.
/// @param name  The name of the function. Eg. "my_function".
/// @param address  The address of the function. Eg. my_function.
/// @param return_type  The return type of the function. Eg. void, int, String, Dictionary, etc.
/// @param args  The comma-separated arguments of the function. Eg. "int a, double b, String c"
/// @param description  The description of the function. Can be empty.
/// @example add_sandbox_api_function(
///    "add_numbers", (void *)add_numbers, "int", "int a, int b", "Adds two numbers together.");
template <typename F>
static inline void add_sandbox_api_function(std::string_view name, F *address, std::string_view return_type, std::string_view args = "", std::string_view description = "") {
	struct GuestFunctionExtra {
		const char *desc;
		size_t desc_len;
		const char *ret;
		size_t ret_len;
		const char *args;
		size_t args_len;
	};
	GuestFunctionExtra extra = {
		.desc = description.data(),
		.desc_len = description.size(),
		.ret = return_type.data(),
		.ret_len = return_type.size(),
		.args = args.data(),
		.args_len = args.size(),
	};
	sys_sandbox_add(1, name.data(), name.size(), address, &extra);
}
#define ADD_API_FUNCTION(func, return_type, ...) \
	add_sandbox_api_function(#func, func, return_type, ##__VA_ARGS__)
