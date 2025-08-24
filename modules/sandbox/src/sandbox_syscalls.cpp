/**************************************************************************/
/*  sandbox_syscalls.cpp                                                  */
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

#include "elf/script_elf.h"
#include "guest_datatypes.h"
#include "sandbox.h"
#include "syscalls.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/variant/callable.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/main/timer.h"
#include "scene/resources/packed_scene.h"
#include <mutex>
#include <string>
#include <unordered_map>
//#define ENABLE_SYSCALL_TRACE 1
#include "syscalls_helpers.hpp"
#include <libriscv/rv32i_instr.hpp>

#define PENALIZE(x)             \
	if (!emu.get_profiling()) { \
		machine.penalize(x);    \
	}

namespace riscv {
extern std::unordered_map<std::string, std::function<uint64_t()>> global_singleton_list;

::Object *get_object_from_address(const Sandbox &emu, uint64_t addr) {
	SYS_TRACE("get_object_from_address", addr);
	::Object *obj = (::Object *)uintptr_t(addr);
	if (UNLIKELY(obj == nullptr)) {
		ERR_PRINT("Object is Null");
		throw std::runtime_error("Object is Null");
	} else if (UNLIKELY(!emu.is_scoped_object(obj))) {
		char buffer[256];
		const uintptr_t obj_uint = reinterpret_cast<uintptr_t>(obj);
		if (obj_uint < 0x1000) {
			snprintf(buffer, sizeof(buffer), "Object is not found, but likely a Variant with index: %lu", long(obj_uint));
			ERR_PRINT(buffer);
			throw std::runtime_error(buffer);
		}
		snprintf(buffer, sizeof(buffer), "Object is not scoped: %p", obj);
		ERR_PRINT(buffer);
		throw std::runtime_error(buffer);
	}
	return obj;
}
inline ::Node *get_node_from_address(const Sandbox &emu, uint64_t addr) {
	SYS_TRACE("get_node_from_address", addr);
	::Object *obj = get_object_from_address(emu, addr);
	::Node *node = ::Object::cast_to<::Node>(obj);
	if (UNLIKELY(node == nullptr)) {
		ERR_PRINT("Object is not a Node: " + obj->get_class());
		throw std::runtime_error("Object was not a Node");
	}
	return node;
}

static inline Variant object_call(Sandbox &emu, ::Object *obj, const Variant &method, const GuestVariant *args, int argc) {
	SYS_TRACE("object_call", method, argc);
	std::array<Variant, 8> vstorage;
	std::array<const Variant *, 8> vargs;
	for (int i = 0; i < argc; i++) {
		if (args[i].is_scoped_variant()) {
			vargs[i] = args[i].toVariantPtr(emu);
		} else {
			vstorage[i] = args[i].toVariant(emu);
			vargs[i] = &vstorage[i];
		}
	}

	Callable::CallError error;
	Variant ret = obj->callp(method, vargs.data(), argc, error);
	return ret;
}

APICALL(api_print) {
	auto [array, len] = machine.sysargs<gaddr_t, unsigned>();
	Sandbox &emu = riscv::emu(machine);

	if (len >= 64) {
		ERR_PRINT("print(): Too many Variants to print");
		throw std::runtime_error("print(): Too many Variants to print");
	}
	const GuestVariant *array_ptr = machine.memory.memarray<GuestVariant>(array, len);

	// We really want print_internal to be a public function.
	for (unsigned i = 0; i < len; i++) {
		const GuestVariant &var = array_ptr[i];
		if (var.is_scoped_variant())
			emu.print(*var.toVariantPtr(emu));
		else
			emu.print(var.toVariant(emu));
	}
}

APICALL(api_vcall) {
	auto [vp, method, mlen, args_ptr, args_size, vret_addr] = machine.sysargs<GuestVariant *, gaddr_t, unsigned, gaddr_t, gaddr_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	SYS_TRACE("vcall", method, mlen, args_ptr, args_size, vret_addr);

	if (args_size > 8) {
		ERR_PRINT("Variant::call(): Too many arguments");
		throw std::runtime_error("Variant::call(): Too many arguments");
	}

	const GuestVariant *args = machine.memory.memarray<GuestVariant>(args_ptr, args_size);
	StringName method_sn;
	std::string_view method_sv = machine.memory.memview(method, mlen + 1); // Include null terminator.
	if (method_sv.back() == '\0') {
		method_sn = StringName(method_sv.data());
	} else {
		method_sn = String::utf8(method_sv.data(), mlen);
	}

	Variant ret;

	if (vp->type == Variant::OBJECT) {
		::Object *obj = get_object_from_address(emu, vp->v.i);

		// Check if the method is allowed.
		if (!emu.is_allowed_method(obj, method_sn)) {
			ERR_PRINT("Variant::call(): Method not allowed: " + method_sn);
			throw std::runtime_error("Variant::call(): Method not allowed: " + std::string(method_sv));
		}

		ret = object_call(emu, obj, method_sn, args, args_size);
	} else {
		std::array<Variant, 8> vargs;
		std::array<const Variant *, 8> argptrs;
		for (size_t i = 0; i < args_size; i++) {
			if (args[i].is_scoped_variant()) {
				argptrs[i] = args[i].toVariantPtr(emu);
			} else {
				vargs[i] = args[i].toVariant(emu);
				argptrs[i] = &vargs[i];
			}
		}

		Callable::CallError error;
		if (vp->is_scoped_variant()) {
			Variant *vcall = const_cast<Variant *>(vp->toVariantPtr(emu));
			vcall->callp(method_sn, argptrs.data(), args_size, ret, error);
		} else {
			Variant vcall = vp->toVariant(emu);
			vcall.callp(method_sn, argptrs.data(), args_size, ret, error);
		}
	}
	// Create a new Variant with the result, if any.
	if (vret_addr != 0) {
		GuestVariant *vret = machine.memory.memarray<GuestVariant>(vret_addr, 1);
		vret->create(emu, std::move(ret));
	}
}

APICALL(api_veval) {
	auto [op, ap, bp, retp] = machine.sysargs<int, GuestVariant *, GuestVariant *, GuestVariant *>();
	auto &emu = riscv::emu(machine);
	SYS_TRACE("veval", op, ap, bp, retp);

	// Special case for comparing objects.
	if (ap->type == Variant::OBJECT && bp->type == Variant::OBJECT) {
		// Special case for equality, allowing invalid objects to be compared.
		if (op == static_cast<int>(Variant::Operator::OP_EQUAL)) {
			machine.set_result(false);
			retp->set(emu, Variant(ap->v.i == bp->v.i));
			return;
		}
		::Object *a = get_object_from_address(emu, ap->v.i);
		::Object *b = get_object_from_address(emu, bp->v.i);
		bool valid = false;
		Variant ret;
		Variant::evaluate(static_cast<Variant::Operator>(op), a, b, ret, valid);

		machine.set_result(valid);
		retp->set(emu, ret, false);
		return;
	}

	bool valid = false;
	Variant ret;
	Variant::evaluate(static_cast<Variant::Operator>(op), ap->toVariant(emu), bp->toVariant(emu), ret, valid);

	machine.set_result(valid);
	retp->create(emu, std::move(ret));
}

APICALL(api_vcreate) {
	auto [vp, type, method, gdata] = machine.sysargs<GuestVariant *, Variant::Type, int, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("vcreate", vp, type, method, gdata);

	switch (type) {
		case Variant::STRING:
		case Variant::STRING_NAME:
		case Variant::NODE_PATH: { // From std::string
			String godot_str;
			if (method == 0) {
				const CppString *str = machine.memory.memarray<const CppString>(gdata, 1);
				godot_str = to_godot_string(str, machine);
			} else if (method == 1) { // const char*, size_t
				const struct LocalBuffer {
					gaddr_t data;
					gaddr_t size;
				} *buffer = machine.memory.memarray<const LocalBuffer>(gdata, 1);
				// View the string from guest memory.
				std::string_view view = machine.memory.memview(buffer->data, buffer->size);
				godot_str = String::utf8(view.data(), view.size());
			} else if (method == 2) { // From std::u32string
				const GuestStdU32String *str = machine.memory.memarray<const GuestStdU32String>(gdata, 1);
				godot_str = str->to_godot_string(machine);
			} else {
				ERR_PRINT("vcreate: Unsupported method for Variant::STRING");
				throw std::runtime_error("vcreate: Unsupported method for Variant::STRING: " + std::to_string(method));
			}
			// Create a new Variant with the string, modify vp.
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(godot_str)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::ARRAY: {
			// Create a new empty? array, assign to vp.
			Array a;
			if (gdata != 0x0) {
				if (method == -1) {
					// Copy std::vector<Variant> from guest memory.
					const CppVector<GuestVariant> *vec = machine.memory.memarray<const CppVector<GuestVariant>>(gdata, 1);
					for (size_t i = 0; i < vec->size(); i++) {
						const GuestVariant &v = vec->at(machine, i);
						a.push_back(std::move(v.toVariant(emu)));
					}
				} else if (method >= 0) {
					// Get elements from method argument.
					const unsigned size = method;
					// Copy array of Variants from guest memory.
					const GuestVariant *gvec = machine.memory.memarray<const GuestVariant>(gdata, size);
					for (unsigned i = 0; i < size; i++) {
						a.push_back(gvec[i].toVariant(emu));
					}
				} else {
					// In order to support future methods, we shouldn't throw an error here.
					WARN_PRINT("vcreate: Unsupported method for Variant::ARRAY: " + itos(method));
				}
			}
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::DICTIONARY: {
			// Create a new empty? dictionary, assign to vp.
			unsigned new_idx = emu.create_scoped_variant(Variant(Dictionary()));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::PACKED_BYTE_ARRAY: {
			PackedByteArray a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<uint8_t> from guest memory.
					const CppVector<uint8_t> *gvec = machine.memory.memarray<const CppVector<uint8_t>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const uint8_t *ptr = machine.memory.memarray<const uint8_t>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method);
				}
			}
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			PackedFloat32Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<float> from guest memory.
					const CppVector<float> *gvec = machine.memory.memarray<const CppVector<float>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const float *ptr = machine.memory.memarray<const float>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(float));
				}
			}
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			PackedFloat64Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<double> from guest memory.
					const CppVector<double> *gvec = machine.memory.memarray<const CppVector<double>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const double *ptr = machine.memory.memarray<const double>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(double));
				}
			}
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::PACKED_INT32_ARRAY: {
			PackedInt32Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<int32_t> from guest memory.
					const CppVector<int32_t> *gvec = machine.memory.memarray<const CppVector<int32_t>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const int32_t *ptr = machine.memory.memarray<const int32_t>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(int32_t));
				}
			}
			unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = new_idx;
		} break;
		case Variant::PACKED_INT64_ARRAY: {
			PackedInt64Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<int64_t> from guest memory.
					const CppVector<int64_t> *gvec = machine.memory.memarray<const CppVector<int64_t>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const int64_t *ptr = machine.memory.memarray<const int64_t>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(int64_t));
				}
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		case Variant::PACKED_VECTOR2_ARRAY: {
			PackedVector2Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<Vector2> from guest memory.
					const CppVector<Vector2> *gvec = machine.memory.memarray<const CppVector<Vector2>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const Vector2 *ptr = machine.memory.memarray<const Vector2>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(Vector2));
				}
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
			PackedVector3Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<Vector3> from guest memory.
					const CppVector<Vector3> *gvec = machine.memory.memarray<const CppVector<Vector3>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const Vector3 *ptr = machine.memory.memarray<const Vector3>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(Vector3));
				}
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		case Variant::PACKED_VECTOR4_ARRAY: {
			PackedVector4Array a;
			if (gdata != 0x0) {
				if (method < 0) {
					// Copy std::vector<Vector4> from guest memory.
					const CppVector<Vector4> *gvec = machine.memory.memarray<const CppVector<Vector4>>(gdata, 1);
					a.resize(gvec->size());
					std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
				} else {
					// Method is the buffer length.
					a.resize(method);
					// Copy the buffer from guest memory.
					const Vector4 *ptr = machine.memory.memarray<const Vector4>(gdata, method);
					std::memcpy(a.ptrw(), ptr, method * sizeof(Vector4));
				}
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		case Variant::PACKED_COLOR_ARRAY: {
			PackedColorArray a;
			if (gdata != 0x0) {
				// Copy std::vector<Color> from guest memory.
				const CppVector<Color> *gvec = machine.memory.memarray<const CppVector<Color>>(gdata, 1);
				a.resize(gvec->size());
				std::memcpy(a.ptrw(), gvec->as_array(machine), gvec->size_bytes());
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		case Variant::PACKED_STRING_ARRAY: {
			PackedStringArray a;
			if (gdata != 0x0) {
				// Copy std::vector<String> from guest memory.
				if (method == -1) {
					const CppVector<CppString> *gvec = machine.memory.memarray<const CppVector<CppString>>(gdata, 1);
					const CppString *str_array = gvec->as_array(machine);
					for (size_t i = 0; i < gvec->size(); i++) {
						a.push_back(to_godot_string(&str_array[i], machine));
					}
				} else if (method == -2) {
					// libc++ std::string implementation.
					struct StringBuffer {
						gaddr_t ptr;
						gaddr_t size;
					};
					const CppVector<StringBuffer> *gvec = machine.memory.memarray<const CppVector<StringBuffer>>(gdata, 1);
					const StringBuffer *buffers = gvec->as_array(machine);
					for (size_t i = 0; i < gvec->size(); i++) {
						const StringBuffer &buf = buffers[i];
						std::string_view view = machine.memory.memview(buf.ptr, buf.size);
						a.push_back(String::utf8(view.data(), view.size()));
					}
				} else {
					ERR_PRINT("vcreate: Unsupported method for Variant::PACKED_STRING_ARRAY");
					throw std::runtime_error("vcreate: Unsupported method for Variant::PACKED_STRING_ARRAY: " + std::to_string(method));
				}
			}
			unsigned idx = emu.create_scoped_variant(Variant(std::move(a)));
			vp->type = type;
			vp->v.i = idx;
		} break;
		default:
			ERR_PRINT("Unsupported Variant type for Variant::create()");
			throw std::runtime_error("Unsupported Variant type for Variant::create(): " + std::string(GuestVariant::type_name(type)));
	}
}

APICALL(api_vfetch) {
	auto [index, gdata, method] = machine.sysargs<unsigned, gaddr_t, int>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("vfetch", index, gdata, method);

	// Find scoped Variant and copy data into gdata.
	std::optional<const Variant *> opt = emu.get_scoped_variant(index);
	if (opt.has_value()) {
		const ::Variant &var = *opt.value();
		switch (var.get_type()) {
			case Variant::STRING:
			case Variant::STRING_NAME:
			case Variant::NODE_PATH: {
				if (method == 0) { // std::string
					auto u8str = var.operator String().utf8();
					CppString *gstr = machine.memory.memarray<CppString>(gdata, 1);
					gstr->set_string(machine, gdata, u8str.ptr(), u8str.length());
				} else if (method == 1) { // const char*, size_t struct
					auto u8str = var.operator String().utf8();
					struct LocalBuffer {
						gaddr_t ptr;
						gaddr_t size;
					} *gstr = machine.memory.memarray<LocalBuffer>(gdata, 1);
					gstr->ptr = machine.arena().malloc(u8str.length());
					gstr->size = u8str.length();
					machine.memory.memcpy(gstr->ptr, u8str.ptr(), u8str.length());
				} else if (method == 2) { // std::u32string
					auto u32str = var.operator String();
					auto *gstr = machine.memory.memarray<GuestStdU32String>(gdata, 1);
					gstr->set_string(machine, gdata, u32str.ptr(), u32str.length());
				} else {
					ERR_PRINT("vfetch: Unsupported method for Variant::STRING");
					throw std::runtime_error("vfetch: Unsupported method for Variant::STRING");
				}
				break;
			}
			case Variant::PACKED_BYTE_ARRAY: {
				CppVector<uint8_t> *gvec = machine.memory.memarray<CppVector<uint8_t>>(gdata, 1);
				auto arr = var.operator PackedByteArray();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_FLOAT32_ARRAY: {
				CppVector<float> *gvec = machine.memory.memarray<CppVector<float>>(gdata, 1);
				auto arr = var.operator PackedFloat32Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_FLOAT64_ARRAY: {
				CppVector<double> *gvec = machine.memory.memarray<CppVector<double>>(gdata, 1);
				auto arr = var.operator PackedFloat64Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_INT32_ARRAY: {
				CppVector<int32_t> *gvec = machine.memory.memarray<CppVector<int32_t>>(gdata, 1);
				auto arr = var.operator PackedInt32Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_INT64_ARRAY: {
				CppVector<int64_t> *gvec = machine.memory.memarray<CppVector<int64_t>>(gdata, 1);
				auto arr = var.operator PackedInt64Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_VECTOR2_ARRAY: {
				CppVector<Vector2> *gvec = machine.memory.memarray<CppVector<Vector2>>(gdata, 1);
				auto arr = var.operator PackedVector2Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_VECTOR3_ARRAY: {
				CppVector<Vector3> *gvec = machine.memory.memarray<CppVector<Vector3>>(gdata, 1);
				auto arr = var.operator PackedVector3Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_VECTOR4_ARRAY: {
				CppVector<Vector4> *gvec = machine.memory.memarray<CppVector<Vector4>>(gdata, 1);
				auto arr = var.operator PackedVector4Array();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_COLOR_ARRAY: {
				CppVector<Color> *gvec = machine.memory.memarray<CppVector<Color>>(gdata, 1);
				auto arr = var.operator PackedColorArray();
				gvec->assign(machine, arr.ptr(), arr.size());
				break;
			}
			case Variant::PACKED_STRING_ARRAY: {
				auto arr = var.operator PackedStringArray();
				if (method == 0) {
					CppVector<CppString> *gvec = machine.memory.memarray<CppVector<CppString>>(gdata, 1);
					gvec->resize(machine, arr.size());
					for (unsigned i = 0; i < arr.size(); i++) {
						auto u8str = arr[i].utf8();
						const gaddr_t self = gvec->address_at(i);
						gvec->at(machine, i).set_string(machine, self, std::string_view(u8str.ptr(), u8str.length()));
					}
				} else if (method == 1) {
					// libc++ std::string implementation.
					struct StringBuffer {
						gaddr_t ptr;
						gaddr_t size;
					};
					CppVector<StringBuffer> *gvec = machine.memory.memarray<CppVector<StringBuffer>>(gdata, 1);
					gvec->reserve(machine, arr.size());
					for (unsigned i = 0; i < arr.size(); i++) {
						auto u8str = arr[i].utf8();
						StringBuffer gb;
						gb.ptr = machine.arena().malloc(u8str.length());
						gb.size = u8str.length();
						machine.memory.memcpy(gb.ptr, u8str.ptr(), u8str.length());
						gvec->push_back(machine, gb);
					}
				} else {
					ERR_PRINT("vfetch: Unsupported method for Variant::PACKED_STRING_ARRAY");
					throw std::runtime_error("vfetch: Unsupported method for Variant::PACKED_STRING_ARRAY");
				}
				break;
			}
			default:
				ERR_PRINT("vfetch: Cannot fetch value into guest for Variant type");
				throw std::runtime_error("vfetch: Cannot fetch value into guest for Variant type");
		}
	} else {
		ERR_PRINT("vfetch: Variant is not scoped");
		throw std::runtime_error("vfetch: Variant is not scoped");
	}
}

APICALL(api_vclone) {
	auto [vp, vret_addr] = machine.sysargs<GuestVariant *, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("vclone", vp, vret);

	if (vret_addr != 0) {
		// Find scoped Variant and clone it.
		std::optional<const Variant *> var = emu.get_scoped_variant(vp->v.i);
		if (var.has_value()) {
			const unsigned index = emu.create_scoped_variant(var.value()->duplicate());
			// Duplicate the Variant and store the index in the guest memory.
			GuestVariant *vret = machine.memory.memarray<GuestVariant>(vret_addr, 1);
			vret->type = var.value()->get_type();
			vret->v.i = index;
		} else {
			ERR_PRINT("vclone: Variant is not scoped");
			throw std::runtime_error("vclone: Variant is not scoped");
		}
	} else {
		// Duplicate or move the Variant into permanent storage (m_level[0]).
		const unsigned idx = vp->v.i;
		unsigned new_idx = emu.create_permanent_variant(idx);
		// Update the Variant with the new index.
		vp->v.i = new_idx;
	}
}

APICALL(api_vstore) {
	auto [vidx, type, gdata, gsize] = machine.sysargs<unsigned *, Variant::Type, gaddr_t, gaddr_t>();
	auto &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("vstore", vidx, type, gdata, gsize);
	if (gsize > 16'777'216) {
		ERR_PRINT("vstore: Array size is too large: " + itos(gsize));
		throw std::runtime_error("vstore: Array size is too large: " + std::to_string(gsize));
	}

	// Find scoped Variant and store data from guest memory.
	switch (type) {
		case Variant::PACKED_BYTE_ARRAY: {
			PackedByteArray arr;
			// Copy the array from guest memory into the Variant.
			uint8_t *data = machine.memory.memarray<uint8_t>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize);
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_FLOAT32_ARRAY: {
			PackedFloat32Array arr;
			// Copy the array from guest memory into the Variant.
			float *data = machine.memory.memarray<float>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(float));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_FLOAT64_ARRAY: {
			PackedFloat64Array arr;
			// Copy the array from guest memory into the Variant.
			double *data = machine.memory.memarray<double>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(double));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_INT32_ARRAY: {
			PackedInt32Array arr;
			// Copy the array from guest memory into the Variant.
			int32_t *data = machine.memory.memarray<int32_t>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(int32_t));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_INT64_ARRAY: {
			PackedInt64Array arr;
			// Copy the array from guest memory into the Variant.
			int64_t *data = machine.memory.memarray<int64_t>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(int64_t));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_VECTOR2_ARRAY: {
			PackedVector2Array arr;
			// Copy the array from guest memory into the Variant.
			auto *data = machine.memory.memarray<Vector2>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(Vector2));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_VECTOR3_ARRAY: {
			PackedVector3Array arr;
			// Copy the array from guest memory into the Variant.
			auto *data = machine.memory.memarray<Vector3>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(Vector3));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_VECTOR4_ARRAY: {
			PackedVector4Array arr;
			// Copy the array from guest memory into the Variant.
			auto *data = machine.memory.memarray<Vector4>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(Vector4));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_COLOR_ARRAY: {
			PackedColorArray arr;
			// Copy the array from guest memory into the Variant.
			auto *data = machine.memory.memarray<Color>(gdata, gsize);
			arr.resize(gsize);
			std::memcpy(arr.ptrw(), data, gsize * sizeof(Color));
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		case Variant::PACKED_STRING_ARRAY: {
			PackedStringArray arr;
			if (gsize & 0x80000000) {
				// Work-around for libc++ std::string implementation.
				struct StringBuffer {
					gaddr_t ptr;
					gaddr_t size;
				};
				gsize &= 0x7FFFFFFF;
				auto *buffers = machine.memory.memarray<StringBuffer>(gdata, gsize);
				arr.resize(gsize);
				for (unsigned i = 0; i < gsize; i++) {
					std::string_view view = machine.memory.memview(buffers[i].ptr, buffers[i].size);
					arr.set(i, String::utf8(view.data(), view.size()));
				}
			} else {
				// Copy the array from guest memory into the Variant.
				const CppString *data = machine.memory.memarray<CppString>(gdata, gsize);
				arr.resize(gsize);
				for (unsigned i = 0; i < gsize; i++) {
					arr.set(i, to_godot_string(&data[i], machine));
				}
			}
			*vidx = emu.create_scoped_variant(Variant(std::move(arr)));
			break;
		}
		default:
			ERR_PRINT("vstore: Cannot store value for Variant type");
			throw std::runtime_error("vstore: Cannot store value for Variant type " + std::to_string(type));
	}
}

APICALL(api_vassign) {
	auto [a_idx, b_idx] = machine.sysargs<unsigned, unsigned>();
	auto &emu = riscv::emu(machine);
	PENALIZE(150'000);
	SYS_TRACE("vassign", a_idx, b_idx);

	if (int32_t(a_idx) == INT32_MIN) {
		machine.set_result(b_idx); // Assign b to a directly when a is "empty".
		return;
	}

	// Find scoped Variants and assign the value of b to a.
	std::optional<const Variant *> a_opt = emu.get_scoped_variant(a_idx);
	std::optional<const Variant *> b_opt = emu.get_scoped_variant(b_idx);
	if (a_opt.has_value() && b_opt.has_value()) {
		const Variant *va = *a_opt;
		const Variant *vb = *b_opt;
		// XXX: This might be too strict. Assigning arbitrarily between different types is allowed in GDScript.
		if (va->get_type() != Variant::NIL && va->get_type() != vb->get_type()) {
			ERR_PRINT("vassign: Variant types do not match");
			throw std::runtime_error("vassign: Variant types do not match: " + std::to_string(va->get_type()) + " != " + std::to_string(vb->get_type()));
		}

		// Try assigning the value of b to a.
		unsigned res_idx = emu.try_reuse_assign_variant(b_idx, *va, a_idx, *vb);
		machine.set_result(res_idx);
	} else {
		ERR_PRINT("vassign: Variants were not scoped");
		throw std::runtime_error("vassign: Variants were not scoped");
	}
}

APICALL(api_get_obj) {
	auto [name] = machine.sysargs<std::string>();
	auto &emu = riscv::emu(machine);
	PENALIZE(150'000);
	SYS_TRACE("get_obj", String::utf8(name.c_str(), name.size()));

	// Objects retrieved by name are named globals, eg. "Engine", "Input", "Time",
	// which are also their class names. As such, we can restrict access using
	// the allowed_classes list in the Sandbox.
	if (!emu.is_allowed_class(String::utf8(name.c_str(), name.size()))) {
		ERR_PRINT("Class is not allowed");
		machine.set_result(0);
		return;
	}

	// Find allowed object by name and get its address from a lambda.
	auto it = global_singleton_list.find(name);
	if (it != global_singleton_list.end()) {
		auto obj = it->second();
		emu.add_scoped_object(reinterpret_cast<::Object *>(obj));
		machine.set_result(obj);
		return;
	}
	// Special case for SceneTree.
	if (name == "SceneTree") {
		// Get the current SceneTree.
		auto *owner_node = emu.get_tree_base();
		if (owner_node == nullptr) {
			ERR_PRINT("Sandbox has no parent Node");
			machine.set_result(0);
			return;
		}
		SceneTree *tree = owner_node->get_tree();
		emu.add_scoped_object(tree);
		machine.set_result(uint64_t(uintptr_t(tree)));
	} else {
		ERR_PRINT(("Unknown or inaccessible object: " + name).c_str());
		machine.set_result(0);
	}
}

APICALL(api_obj) {
	auto [op, addr, gvar] = machine.sysargs<int, uint64_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(250'000); // Costly Object operations.
	SYS_TRACE("obj_op", op, addr, gvar);

	::Object *obj = get_object_from_address(emu, addr);

	switch (Object_Op(op)) {
		case Object_Op::GET_METHOD_LIST: {
			CppVector<CppString> *vec = machine.memory.memarray<CppVector<CppString>>(gvar, 1);
			// XXX: vec->free(machine);
			List<MethodInfo> methods;
			obj->get_method_list(&methods);
			vec->resize(machine, methods.size());
			int i = 0;
			for (const MethodInfo &method : methods) {
				auto name = String(method.name).utf8();
				const gaddr_t self = vec->address_at(i);
				vec->at(machine, i).set_string(machine, self, name.ptr(), name.length());
				i++;
			}
		} break;
		case Object_Op::GET: { // Get a property of the object.
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 2);
			String name = var[0].toVariant(emu).operator String();
			if (UNLIKELY(!emu.is_allowed_property(obj, name, false))) {
				ERR_PRINT("Banned property accessed: " + name);
				throw std::runtime_error("Banned property accessed");
			}
			var[1].create(emu, obj->get(name));
		} break;
		case Object_Op::SET: { // Set a property of the object.
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 2);
			String name = var[0].toVariant(emu).operator String();
			if (UNLIKELY(!emu.is_allowed_property(obj, name, true))) {
				ERR_PRINT("Banned property set: " + name);
				throw std::runtime_error("Banned property set");
			}
			obj->set(name, var[1].toVariant(emu));
		} break;
		case Object_Op::GET_PROPERTY_LIST: {
			CppVector<CppString> *vec = machine.memory.memarray<CppVector<CppString>>(gvar, 1);
			// XXX: vec->free(machine);
			List<PropertyInfo> properties;
			obj->get_property_list(&properties);
			vec->resize(machine, properties.size());
			int i = 0;
			for (const PropertyInfo &property : properties) {
				auto name = String(property.name).utf8();
				const gaddr_t self = vec->address_at(i);
				vec->at(machine, i).set_string(machine, self, name.ptr(), name.length());
				i++;
			}
		} break;
		case Object_Op::CONNECT: {
			GuestVariant *vars = machine.memory.memarray<GuestVariant>(gvar, 3);
			::Object *target = get_object_from_address(emu, vars[0].v.i);
			Callable callable = Callable(target, vars[2].toVariant(emu).operator String());
			obj->connect(vars[1].toVariant(emu).operator String(), callable);
		} break;
		case Object_Op::DISCONNECT: {
			GuestVariant *vars = machine.memory.memarray<GuestVariant>(gvar, 3);
			::Object *target = get_object_from_address(emu, vars[0].v.i);
			auto callable = Callable(target, vars[2].toVariant(emu).operator String());
			obj->disconnect(vars[1].toVariant(emu).operator String(), callable);
		} break;
		case Object_Op::GET_SIGNAL_LIST: {
			CppVector<CppString> *vec = machine.memory.memarray<CppVector<CppString>>(gvar, 1);
			List<MethodInfo> signals;
			obj->get_signal_list(&signals);
			vec->resize(machine, signals.size());
			int i = 0;
			for (const MethodInfo &signal : signals) {
				auto name = String(signal.name).utf8();
				const gaddr_t self = vec->address_at(i);
				vec->at(machine, i).set_string(machine, self, std::string_view(name.ptr(), name.length()));
				i++;
			}
		} break;
		default:
			throw std::runtime_error("Invalid Object operation: " + std::to_string(op));
	}
}

APICALL(api_obj_property_get) {
	auto [addr, method, vret] = machine.sysargs<uint64_t, std::string_view, GuestVariant *>();
	auto &emu = riscv::emu(machine);
	PENALIZE(150'000);
	SYS_TRACE("obj_property_get", addr, method, vret);

	::Object *obj = get_object_from_address(emu, addr);
	String prop_name = String::utf8(method.data(), method.size());

	if (UNLIKELY(!emu.is_allowed_property(obj, prop_name, false))) {
		ERR_PRINT("Banned property accessed: " + prop_name);
		throw std::runtime_error("Banned property accessed: " + std::string(prop_name.utf8().ptr()));
	}

	vret->create(emu, obj->get(prop_name));
}

APICALL(api_obj_property_set) {
	auto [addr, method, g_value] = machine.sysargs<uint64_t, std::string_view, const GuestVariant *>();
	auto &emu = riscv::emu(machine);
	PENALIZE(150'000);
	SYS_TRACE("obj_property_set", addr, method, value);

	::Object *obj = get_object_from_address(emu, addr);
	String prop_name = String::utf8(method.data(), method.size());

	if (UNLIKELY(!emu.is_allowed_property(obj, prop_name, true))) {
		ERR_PRINT("Banned property set: " + prop_name);
		throw std::runtime_error("Banned property set: " + std::string(prop_name.utf8().ptr()));
	}

	obj->set(prop_name, g_value->toVariant(emu));
}

APICALL(api_obj_callp) {
	auto [addr, g_method, g_method_len, deferred, vret_ptr, args_addr, args_size] = machine.sysargs<uint64_t, gaddr_t, unsigned, bool, gaddr_t, gaddr_t, unsigned>();
	auto &emu = riscv::emu(machine);
	PENALIZE(250'000); // Costly Object call operation.
	SYS_TRACE("obj_callp", addr, g_method, g_method_len, deferred, vret_ptr, args_addr, args_size);

	auto *obj = get_object_from_address(emu, addr);
	if (UNLIKELY(args_size > 8)) {
		ERR_PRINT("Too many arguments to obj_callp");
		throw std::runtime_error("Too many arguments to obj_callp");
	}
	const GuestVariant *g_args = machine.memory.memarray<GuestVariant>(args_addr, args_size);

	Variant method;
	std::string_view method_view = machine.memory.memview(g_method, g_method_len + 1);

	// Check if the method is null-terminated.
	if (method_view.back() == '\0') {
		method = StringName(method_view.data(), false);
	} else {
		method = String::utf8(method_view.data(), method_view.size() - 1);
	}

	// Check for banned methods.
	if (UNLIKELY(!emu.is_allowed_method(obj, method))) {
		ERR_PRINT("Banned method called: " + method.operator String());
		throw std::runtime_error("Banned method called: " + std::string(method_view));
	}

	if (!deferred) {
		Variant ret = object_call(emu, obj, method, g_args, args_size);
		if (vret_ptr != 0) {
			GuestVariant *vret = machine.memory.memarray<GuestVariant>(vret_ptr, 1);
			vret->create(emu, std::move(ret));
		}
	} else {
		// Call deferred unfortunately takes a parameter pack, so we have to manually
		// check the number of arguments, and call the correct function.
		if (args_size == 0) {
			obj->call_deferred(method);
		} else if (args_size == 1) {
			obj->call_deferred(method, g_args[0].toVariant(emu));
		} else if (args_size == 2) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu));
		} else if (args_size == 3) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu));
		} else if (args_size == 4) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu), g_args[3].toVariant(emu));
		} else if (args_size == 5) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu), g_args[3].toVariant(emu), g_args[4].toVariant(emu));
		} else if (args_size == 6) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu), g_args[3].toVariant(emu), g_args[4].toVariant(emu), g_args[5].toVariant(emu));
		} else if (args_size == 7) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu), g_args[3].toVariant(emu), g_args[4].toVariant(emu), g_args[5].toVariant(emu), g_args[6].toVariant(emu));
		} else if (args_size == 8) {
			obj->call_deferred(method, g_args[0].toVariant(emu), g_args[1].toVariant(emu), g_args[2].toVariant(emu), g_args[3].toVariant(emu), g_args[4].toVariant(emu), g_args[5].toVariant(emu), g_args[6].toVariant(emu), g_args[7].toVariant(emu));
		}
	}
}

APICALL(api_get_node) {
	auto [addr, name] = machine.sysargs<uint64_t, std::string_view>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(150'000);
	SYS_TRACE("get_node", addr, String::utf8(name.data(), name.size()));

	Node *node = nullptr;
	const std::string c_name(name);

	if (addr == 0) {
		auto *owner_node = emu.get_tree_base();
		if (owner_node == nullptr) {
			ERR_PRINT("Sandbox has no parent Node");
			machine.set_result(0);
			return;
		}
		node = owner_node->get_node(NodePath(c_name.c_str()));
	} else {
		Node *base_node = get_node_from_address(emu, addr);
		node = base_node->get_node(NodePath(c_name.c_str()));
	}
	if (node == nullptr) {
		ERR_PRINT(("Node not found: " + c_name).c_str());
		machine.set_result(0);
		return;
	}

	emu.add_scoped_object(node);
	machine.set_result(reinterpret_cast<uint64_t>(node));
}

APICALL(api_node_create) {
	auto [type, g_class_name, g_class_len, name] = machine.sysargs<Node_Create_Shortlist, gaddr_t, unsigned, std::string_view>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(150'000);
	Node *node = nullptr;

	switch (type) {
		case Node_Create_Shortlist::CREATE_CLASSDB: {
			// Get the class name from guest memory, including null terminator.
			std::string_view class_name = machine.memory.memview(g_class_name, g_class_len + 1);
			// Verify the class name is null-terminated.
			if (class_name[g_class_len] != '\0') {
				ERR_PRINT("Class name is not null-terminated");
				throw std::runtime_error("Class name is not null-terminated");
			}
			StringName class_name_sn(class_name.data());
			if (!emu.is_allowed_class(class_name_sn)) {
				ERR_PRINT("Class name is not allowed");
				throw std::runtime_error("Class name is not allowed");
			}
			// Now that it's null-terminated, we can use it for StringName.
			Object *obj = ClassDB::instantiate(class_name_sn);
			if (obj == nullptr) {
				ERR_PRINT("Failed to create object from class name");
				throw std::runtime_error("Failed to create object from class name");
			}

			node = Object::cast_to<Node>(obj);
			// If it's not a Node, just return the Object.
			if (node == nullptr) {
				emu.add_scoped_object(obj);
				machine.set_result(uint64_t(uintptr_t(obj)));
				return;
			}
			// It's a Node, so continue to set the name.
			break;
		}
		case Node_Create_Shortlist::CREATE_NODE: // Node
			if (!emu.is_allowed_class("Node")) {
				ERR_PRINT("Class name is not allowed");
				throw std::runtime_error("Class name is not allowed");
			}
			node = memnew(Node);
			break;
		case Node_Create_Shortlist::CREATE_NODE2D: // Node2D
			if (!emu.is_allowed_class("Node2D")) {
				ERR_PRINT("Class name is not allowed");
				throw std::runtime_error("Class name is not allowed");
			}
			node = memnew(Node2D);
			break;
		case Node_Create_Shortlist::CREATE_NODE3D: // Node3D
			if (!emu.is_allowed_class("Node3D")) {
				ERR_PRINT("Class name is not allowed");
				throw std::runtime_error("Class name is not allowed");
			}
			node = memnew(Node3D);
			break;
		default:
			ERR_PRINT("Unknown Node type");
			throw std::runtime_error("Unknown Node type");
	}

	if (node == nullptr) {
		ERR_PRINT("Failed to create Node");
		throw std::runtime_error("Failed to create Node");
	}
	if (!name.empty()) {
		node->set_name(String::utf8(name.data(), name.size()));
	}
	emu.add_scoped_object(node);
	machine.set_result(uint64_t(uintptr_t(node)));
}

APICALL(api_node) {
	auto [op, addr, gvar] = machine.sysargs<int, uint64_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(250'000); // Costly Node operations.
	SYS_TRACE("node_op", op, addr, gvar);

	// Get the Node object by its address.
	::Node *node = get_node_from_address(emu, addr);

	switch (Node_Op(op)) {
		case Node_Op::GET_NAME: {
			// Check if getting the name is allowed.
			if (UNLIKELY(!emu.is_allowed_property(node, "name", false))) {
				ERR_PRINT("Banned property accessed: name");
				throw std::runtime_error("Banned property accessed: name");
			}
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
			var->create(emu, node->get_name());
		} break;
		case Node_Op::SET_NAME: {
			// Check if setting the name is allowed.
			if (UNLIKELY(!emu.is_allowed_property(node, "name", true))) {
				ERR_PRINT("Banned property set: name");
				throw std::runtime_error("Banned property set: name");
			}
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
			node->set_name(var->toVariant(emu));
		} break;
		case Node_Op::GET_PATH: {
			// Check if getting the path is allowed.
			if (UNLIKELY(!emu.is_allowed_method(node, "path"))) {
				ERR_PRINT("Banned method accessed: path");
				throw std::runtime_error("Banned method accessed: path");
			}
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
			var->create(emu, node->get_path());
		} break;
		case Node_Op::GET_PARENT: {
			// Check if getting the parent is allowed.
			if (UNLIKELY(!emu.is_allowed_method(node, "get_parent"))) {
				ERR_PRINT("Banned method accessed: get_parent");
				throw std::runtime_error("Banned method accessed: get_parent");
			}
			uint64_t *result = machine.memory.memarray<uint64_t>(gvar, 1);
			::Object *parent = node->get_parent();
			if (UNLIKELY(parent == nullptr)) {
				*result = 0;
			} else {
				// TODO: Parent nodes allow access higher up the tree, which could be a security issue.
				if (!emu.is_allowed_object(parent))
					throw std::runtime_error("Node::get_parent(): Parent is not allowed");
				emu.add_scoped_object(parent);
				*result = uint64_t(uintptr_t(parent));
			}
		} break;
		case Node_Op::QUEUE_FREE:
			if (UNLIKELY(node == &emu)) {
				ERR_PRINT("Cannot queue free the sandbox");
				throw std::runtime_error("Cannot queue free the sandbox");
			}
			// Check if queue_free is an allowed method.
			if (UNLIKELY(!emu.is_allowed_method(node, "queue_free"))) {
				ERR_PRINT("Banned method called: queue_free");
				throw std::runtime_error("Banned method called: queue_free");
			}
			//emu.rem_scoped_object(node);
			node->queue_free();
			break;
		case Node_Op::DUPLICATE: {
			// Check if creating a new node of this type is allowed.
			if (UNLIKELY(!emu.is_allowed_class(node->get_class()))) {
				throw std::runtime_error("Node::duplicate(): Creating a new node of this type is not allowed");
			}
			// Check if duplicate is an allowed method.
			if (UNLIKELY(!emu.is_allowed_method(node, "duplicate"))) {
				ERR_PRINT("Banned method called: duplicate");
				throw std::runtime_error("Banned method called: duplicate");
			}
			uint64_t *result = machine.memory.memarray<uint64_t>(gvar, 1);
			int flags = machine.cpu.reg(13); // Flags are passed in reg 13.
			auto *new_node = node->duplicate(flags);
			emu.add_scoped_object(new_node);
			*result = uint64_t(uintptr_t(new_node));
		} break;
		case Node_Op::GET_CHILD_COUNT: {
			int64_t *result = machine.memory.memarray<int64_t>(gvar, 1);
			*result = node->get_child_count();
		} break;
		case Node_Op::GET_CHILD: {
			GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
			Node *child_node = node->get_child(var[0].v.i);
			if (UNLIKELY(child_node == nullptr)) {
				var[0].set(emu, Variant());
			} else {
				emu.add_scoped_object(child_node);
				var[0].set(emu, int64_t(uintptr_t(child_node)));
			}
		} break;
		case Node_Op::ADD_CHILD_DEFERRED:
		case Node_Op::ADD_CHILD: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "add_child"))) {
				ERR_PRINT("Banned method called: add_child");
				throw std::runtime_error("Banned method called: add_child");
			}
			GuestVariant *child = machine.memory.memarray<GuestVariant>(gvar, 1);
			::Node *child_node = get_node_from_address(emu, child->v.i);
			if (Node_Op(op) == Node_Op::ADD_CHILD_DEFERRED)
				node->call_deferred("add_child", child_node);
			else
				node->add_child(child_node);
		} break;
		case Node_Op::ADD_SIBLING_DEFERRED:
		case Node_Op::ADD_SIBLING: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "add_sibling"))) {
				ERR_PRINT("Banned method called: add_sibling");
				throw std::runtime_error("Banned method called: add_sibling");
			}
			GuestVariant *sibling = machine.memory.memarray<GuestVariant>(gvar, 1);
			::Node *sibling_node = get_node_from_address(emu, sibling->v.i);
			if (Node_Op(op) == Node_Op::ADD_SIBLING_DEFERRED)
				node->call_deferred("add_sibling", sibling_node);
			else
				node->add_sibling(sibling_node);
		} break;
		case Node_Op::MOVE_CHILD: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "move_child"))) {
				ERR_PRINT("Banned method called: move_child");
				throw std::runtime_error("Banned method called: move_child");
			}
			GuestVariant *vars = machine.memory.memarray<GuestVariant>(gvar, 2);
			::Node *child_node = get_node_from_address(emu, vars[0].v.i);
			// TODO: Check if the child is actually a child of the node? Verify index?
			node->move_child(child_node, vars[1].v.i);
		} break;
		case Node_Op::REMOVE_CHILD_DEFERRED:
		case Node_Op::REMOVE_CHILD: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "remove_child"))) {
				ERR_PRINT("Banned method called: remove_child");
				throw std::runtime_error("Banned method called: remove_child");
			}
			GuestVariant *child = machine.memory.memarray<GuestVariant>(gvar, 1);
			::Node *child_node = get_node_from_address(emu, child->v.i);
			if (Node_Op(op) == Node_Op::REMOVE_CHILD_DEFERRED)
				node->call_deferred("remove_child", child_node);
			else
				node->remove_child(child_node);
		} break;
		case Node_Op::GET_CHILDREN: {
			// Check if getting the children is allowed.
			if (UNLIKELY(!emu.is_allowed_method(node, "get_children"))) {
				ERR_PRINT("Banned method accessed: get_children");
				throw std::runtime_error("Banned method accessed: get_children");
			}
			// Get a GuestStdVector from guest to store the children.
			CppVector<uint64_t> *vec = machine.memory.memarray<CppVector<uint64_t>>(gvar, 1);
			// Get the children of the node.
			TypedArray<Node> children = node->get_children();
			// Allocate memory for the children in the guest vector.
			vec->reserve(machine, children.size());
			// Copy the children to the guest vector, and add them to the scoped objects.
			for (int i = 0; i < children.size(); i++) {
				::Node *child = ::Object::cast_to<::Node>(children[i]);
				if (child) {
					emu.add_scoped_object(child);
					vec->push_back(machine, uint64_t(uintptr_t(child)));
				} else {
					vec->push_back(machine, 0);
				}
			}
			// No return value is needed.
		} break;
		case Node_Op::ADD_TO_GROUP: {
			// Reg 12: Group string pointer, Reg 13: Group string length.
			std::string_view group = machine.memory.memview(gvar, machine.cpu.reg(13));
			node->add_to_group(String::utf8(group.data(), group.size()));
		} break;
		case Node_Op::REMOVE_FROM_GROUP: {
			// Reg 12: Group string pointer, Reg 13: Group string length.
			std::string_view group = machine.memory.memview(gvar, machine.cpu.reg(13));
			node->remove_from_group(String::utf8(group.data(), group.size()));
		} break;
		case Node_Op::IS_IN_GROUP: {
			// Reg 12: Group string pointer, Reg 13: Group string length, Reg 14: Result bool pointer.
			std::string_view group = machine.memory.memview(gvar, machine.cpu.reg(13));
			bool *result = machine.memory.memarray<bool>(machine.cpu.reg(14), 1);
			*result = node->is_in_group(String::utf8(group.data(), group.size()));
		} break;
		case Node_Op::REPLACE_BY: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "replace_by"))) {
				ERR_PRINT("Banned method called: replace_by");
				throw std::runtime_error("Banned method called: replace_by");
			}
			// Reg 12: Node address to replace with, Reg 13: Keep groups bool.
			::Node *replace_node = get_node_from_address(emu, gvar);
			bool keep_groups = machine.cpu.reg(13);
			node->replace_by(replace_node, keep_groups);
		} break;
		case Node_Op::REPARENT: {
			// Check for banned methods.
			if (UNLIKELY(!emu.is_allowed_method(node, "reparent"))) {
				ERR_PRINT("Banned method called: reparent");
				throw std::runtime_error("Banned method called: reparent");
			}
			// Reg 12: New parent node address, Reg 13: Keep transform bool.
			::Node *new_parent = get_node_from_address(emu, gvar);
			bool keep_transform = machine.cpu.reg(13);
			node->reparent(new_parent, keep_transform);
		} break;
		case Node_Op::IS_INSIDE_TREE: {
			// Reg 12: Result bool pointer.
			bool *result = machine.memory.memarray<bool>(gvar, 1);
			*result = node->is_inside_tree();
		} break;
		default:
			throw std::runtime_error("Invalid Node operation");
	}
}

APICALL(api_node2d) {
	// Node2D operation, Node2D address, and the variant to get/set the value.
	auto [op, addr, gvar] = machine.sysargs<int, uint64_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(100'000); // Costly Node2D operations.
	SYS_TRACE("node2d_op", op, addr, gvar);

	// Get the Node2D object by its address.
	::Node *node = get_node_from_address(emu, addr);

	// Cast the Node2D object to a Node2D object.
	::Node2D *node2d = ::Object::cast_to<::Node2D>(node);
	if (node2d == nullptr) {
		ERR_PRINT("Node2D object is not a Node2D");
		throw std::runtime_error("Node2D object is not a Node2D");
	}

	// View the variant from the guest memory.
	GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
	switch (Node2D_Op(op)) {
		case Node2D_Op::GET_POSITION:
			var->set(emu, node2d->get_position());
			break;
		case Node2D_Op::SET_POSITION:
			node2d->set_deferred("position", var->toVariant(emu));
			break;
		case Node2D_Op::GET_ROTATION:
			var->set(emu, node2d->get_rotation());
			break;
		case Node2D_Op::SET_ROTATION:
			node2d->set_rotation(var->toVariant(emu));
			break;
		case Node2D_Op::GET_SCALE:
			var->set(emu, node2d->get_scale());
			break;
		case Node2D_Op::SET_SCALE:
			node2d->set_scale(var->toVariant(emu));
			break;
		case Node2D_Op::GET_SKEW:
			var->set(emu, node2d->get_skew());
			break;
		case Node2D_Op::SET_SKEW:
			node2d->set_skew(var->toVariant(emu));
			break;
		case Node2D_Op::GET_TRANSFORM:
			var->create(emu, node2d->get_transform());
			break;
		case Node2D_Op::SET_TRANSFORM:
			node2d->set_transform(*var->toVariantPtr(emu));
			break;
		default:
			ERR_PRINT("Invalid Node2D operation");
			throw std::runtime_error("Invalid Node2D operation");
	}
}

APICALL(api_node3d) {
	// Node3D operation, Node3D address, and the variant to get/set the value.
	auto [op, addr, gvar] = machine.sysargs<int, uint64_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(100'000); // Costly Node3D operations.
	SYS_TRACE("node3d_op", op, addr, gvar);

	// Get the Node3D object by its address
	::Node *node = get_node_from_address(emu, addr);

	// Cast the Node3D object to a Node3D object.
	::Node3D *node3d = ::Object::cast_to<::Node3D>(node);
	if (node3d == nullptr) {
		ERR_PRINT("Node3D object is not a Node3D");
		throw std::runtime_error("Node3D object is not a Node3D");
	}

	// View the variant from the guest memory.
	GuestVariant *var = machine.memory.memarray<GuestVariant>(gvar, 1);
	switch (Node3D_Op(op)) {
		case Node3D_Op::GET_POSITION:
			var->set(emu, node3d->get_position());
			break;
		case Node3D_Op::SET_POSITION:
			node3d->set_position(var->toVariant(emu));
			break;
		case Node3D_Op::GET_ROTATION:
			var->set(emu, node3d->get_rotation());
			break;
		case Node3D_Op::SET_ROTATION:
			node3d->set_rotation(var->toVariant(emu));
			break;
		case Node3D_Op::GET_SCALE:
			var->set(emu, node3d->get_scale());
			break;
		case Node3D_Op::SET_SCALE:
			node3d->set_scale(var->toVariant(emu));
			break;
		case Node3D_Op::GET_TRANSFORM:
			var->create(emu, node3d->get_transform());
			break;
		case Node3D_Op::SET_TRANSFORM:
			node3d->set_transform(*var->toVariantPtr(emu));
			break;
		case Node3D_Op::GET_QUATERNION:
			var->set(emu, node3d->get_quaternion());
			break;
		case Node3D_Op::SET_QUATERNION:
			node3d->set_quaternion(var->toVariant(emu));
			break;
		default:
			ERR_PRINT("Invalid Node3D operation");
			throw std::runtime_error("Invalid Node3D operation");
	}
}

APICALL(api_throw) {
	auto [type, msg, vaddr, vfunc] = machine.sysargs<std::string_view, std::string_view, gaddr_t, gaddr_t>();
	SYS_TRACE("throw", String::utf8(type.data(), type.size()), String::utf8(msg.data(), msg.size()), vaddr);

	if (vaddr != 0) {
		GuestVariant *var = machine.memory.memarray<GuestVariant>(vaddr, 1);
		String error_string = "Sandbox exception of type " + String::utf8(type.data(), type.size()) + ": " + String::utf8(msg.data(), msg.size()) + " for Variant of type " + itos(var->type);
		if (var->type >= 0 && var->type < Variant::VARIANT_MAX) {
			const char *type_hint = GuestVariant::type_name(var->type);
			error_string += " (" + String::utf8(type_hint) + ")";
		}
		if (vfunc != 0x0) {
			error_string += " in function " + String::utf8(machine.memory.memstring(vfunc).c_str());
		}
		ERR_PRINT(error_string);
		const std::string cpp_error = error_string.utf8().ptr();
		throw std::runtime_error(cpp_error);
	} else {
		String error_string = "Sandbox exception in " + String::utf8(type.data(), type.size()) + ": " + String::utf8(msg.data(), msg.size());
		if (vfunc != 0x0) {
			error_string += " in function " + String::utf8(machine.memory.memstring(vfunc).c_str());
		}
		ERR_PRINT(error_string);
		throw std::runtime_error(error_string.utf8().ptr());
	}
}

APICALL(api_array_ops) {
	auto [op, arr_idx, idx, vaddr] = machine.sysargs<Array_Op, unsigned, int, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(50'000); // Costly Array operations.
	SYS_TRACE("array_ops", int(op), arr_idx, idx, vaddr);

	if (op == Array_Op::CREATE) {
		// There is no scoped array, so we need to create one.
		Array a;
		a.resize(arr_idx); // Resize the array to the given size.
		const unsigned new_idx = emu.create_scoped_variant(Variant(std::move(a)));
		GuestVariant *vp = machine.memory.memarray<GuestVariant>(vaddr, 1);
		vp->type = Variant::ARRAY;
		vp->v.i = new_idx;
		return;
	}

	std::optional<const Variant *> opt_array = emu.get_scoped_variant(arr_idx);
	if (!opt_array.has_value() || opt_array.value()->get_type() != Variant::ARRAY) {
		ERR_PRINT("Invalid Array object");
		throw std::runtime_error("Invalid Array object, idx = " + std::to_string(arr_idx));
	}
	::Array array = opt_array.value()->operator Array();

	switch (op) {
		case Array_Op::PUSH_BACK:
			array.push_back(machine.memory.memarray<GuestVariant>(vaddr, 1)->toVariant(emu));
			break;
		case Array_Op::PUSH_FRONT:
			array.push_front(machine.memory.memarray<GuestVariant>(vaddr, 1)->toVariant(emu));
			break;
		case Array_Op::POP_AT:
			array.pop_at(idx);
			break;
		case Array_Op::POP_BACK:
			array.pop_back();
			break;
		case Array_Op::POP_FRONT:
			array.pop_front();
			break;
		case Array_Op::INSERT:
			array.insert(idx, machine.memory.memarray<GuestVariant>(vaddr, 1)->toVariant(emu));
			break;
		case Array_Op::ERASE:
			// TODO: Check if we can use pointer to Variant to avoid copying.
			array.erase(machine.memory.memarray<GuestVariant>(vaddr, 1)->toVariant(emu));
			break;
		case Array_Op::RESIZE:
			array.resize(idx);
			break;
		case Array_Op::CLEAR:
			array.clear();
			break;
		case Array_Op::SORT:
			array.sort();
			break;
		case Array_Op::FETCH_TO_VECTOR: {
			CppVector<GuestVariant> *vec = machine.memory.memarray<CppVector<GuestVariant>>(vaddr, 1);
			vec->resize(machine, array.size());
			for (int i = 0; i < array.size(); i++) {
				vec->at(machine, i).create(emu, array[i].duplicate(false));
			}
			break;
		}
		case Array_Op::HAS: {
			auto *vp = machine.memory.memarray<GuestVariant>(vaddr, 1);
			const bool result = array.has(vp->toVariant(emu));
			vp->set(emu, result);
			break;
		}
		default:
			ERR_PRINT("Invalid Array operation");
			throw std::runtime_error("Invalid Array operation");
	}
}

APICALL(api_array_at) {
	auto [arr_idx, idx, vret] = machine.sysargs<unsigned, int, GuestVariant *>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000); // Costly Array operations.
	SYS_TRACE("array_at", arr_idx, idx, vret);

	std::optional<const Variant *> opt_array = emu.get_scoped_variant(arr_idx);
	if (!opt_array.has_value() || opt_array.value()->get_type() != Variant::ARRAY) {
		ERR_PRINT("Invalid Array object");
		throw std::runtime_error("Invalid Array object, idx = " + std::to_string(arr_idx));
	}

	::Array array = opt_array.value()->operator Array();
	const bool set_mode = idx < 0;
	if (set_mode) {
		idx = -idx - 1;
	}
	if (idx >= array.size()) {
		ERR_PRINT("Array index out of bounds: " + itos(idx));
		throw std::runtime_error("Array index out of bounds: " + std::to_string(idx));
	}

	if (set_mode) {
		array[idx] = vret->toVariant(emu);
	} else {
		Variant ref = array[idx];
		vret->create(emu, std::move(ref));
	}
}

APICALL(api_array_size) {
	auto [arr_idx] = machine.sysargs<unsigned>();
	Sandbox &emu = riscv::emu(machine);

	std::optional<const Variant *> opt_array = emu.get_scoped_variant(arr_idx);
	if (!opt_array.has_value() || opt_array.value()->get_type() != Variant::ARRAY) {
		ERR_PRINT("Invalid Array object");
		throw std::runtime_error("Invalid Array object");
	}

	::Array array = opt_array.value()->operator Array();
	machine.set_result(array.size());
}

APICALL(api_dict_ops) {
	auto [op, dict_idx, vkey, vaddr] = machine.sysargs<Dictionary_Op, unsigned, gaddr_t, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(50'000); // Costly Dictionary operations.
	SYS_TRACE("dict_ops", int(op), dict_idx, vkey, vaddr);

	std::optional<const Variant *> opt_dict = emu.get_scoped_variant(dict_idx);
	if (!opt_dict.has_value() || opt_dict.value()->get_type() != Variant::DICTIONARY) {
		ERR_PRINT("Invalid Dictionary object");
		throw std::runtime_error("Invalid Dictionary object");
	}
	::Dictionary dict = opt_dict.value()->operator Dictionary();

	switch (op) {
		case Dictionary_Op::GET: {
			GuestVariant *key = machine.memory.memarray<GuestVariant>(vkey, 1);
			GuestVariant *vp = machine.memory.memarray<GuestVariant>(vaddr, 1);
			// TODO: Check if the value is already scoped?
			Variant v = dict[key->toVariant(emu)];
			vp->create(emu, std::move(v));
			break;
		}
		case Dictionary_Op::SET: {
			GuestVariant *key = machine.memory.memarray<GuestVariant>(vkey, 1);
			GuestVariant *value = machine.memory.memarray<GuestVariant>(vaddr, 1);
			dict[key->toVariant(emu)] = value->toVariant(emu);
			break;
		}
		case Dictionary_Op::ERASE: {
			GuestVariant *key = machine.memory.memarray<GuestVariant>(vkey, 1);
			dict.erase(key->toVariant(emu));
			break;
		}
		case Dictionary_Op::HAS: {
			GuestVariant *key = machine.memory.memarray<GuestVariant>(vkey, 1);
			machine.set_result(dict.has(key->toVariant(emu)));
			break;
		}
		case Dictionary_Op::GET_SIZE:
			machine.set_result(dict.size());
			break;
		case Dictionary_Op::CLEAR:
			dict.clear();
			break;
		case Dictionary_Op::MERGE: {
			GuestVariant *other_dict = machine.memory.memarray<GuestVariant>(vkey, 1);
			dict.merge(other_dict->toVariant(emu).operator Dictionary());
			break;
		}
		case Dictionary_Op::GET_OR_ADD: {
			GuestVariant *key = machine.memory.memarray<GuestVariant>(vkey, 1);
			GuestVariant *vp = machine.memory.memarray<GuestVariant>(vaddr, 1);
			Variant &v = dict[key->toVariant(emu)];
			if (v.get_type() == Variant::NIL) {
				const gaddr_t vdefaddr = machine.cpu.reg(14); // A4
				const GuestVariant *vdef = machine.memory.memarray<GuestVariant>(vdefaddr, 1);
				v = vdef->toVariant(emu);
			}
			vp->set(emu, v, true); // Implicit trust, as we are returning our own object.
			break;
		}
		default:
			ERR_PRINT("Invalid Dictionary operation");
			throw std::runtime_error("Invalid Dictionary operation");
	}
}

APICALL(api_string_create) {
	auto [strview] = machine.sysargs<std::string_view>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("string_create", String::utf8(strview.data(), strview.size()));

	String str = String::utf8(strview.data(), strview.size());
	const unsigned idx = emu.create_scoped_variant(Variant(std::move(str)));
	machine.set_result(idx);
}

APICALL(api_string_ops) {
	auto [op, str_idx, index, vaddr] = machine.sysargs<String_Op, unsigned, int, gaddr_t>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000); // Costly String operations.
	SYS_TRACE("string_ops", int(op), str_idx, index, vaddr);

	std::optional<const Variant *> opt_str = emu.get_scoped_variant(str_idx);
	if (!opt_str.has_value()) {
		ERR_PRINT("Invalid String object idx: " + itos(str_idx));
		throw std::runtime_error("Invalid String object: " + std::to_string(str_idx));
	}
	const Variant::Type type = opt_str.value()->get_type();
	if (type != Variant::STRING && type != Variant::STRING_NAME && type != Variant::NODE_PATH) {
		ERR_PRINT("Invalid String object type: " + itos(type));
		throw std::runtime_error("Invalid String object type: " + std::to_string(type));
	}
	::String str = opt_str.value()->operator String();

	switch (op) {
		case String_Op::APPEND: {
			GuestVariant *gvar = machine.memory.memarray<GuestVariant>(vaddr, 1);
			str += gvar->toVariant(emu).operator String();
			break;
		}
		case String_Op::GET_LENGTH:
			machine.set_result(str.length());
			break;
		case String_Op::TO_STD_STRING: {
			if (index == 0) { // Get the string as a std::string.
				CharString utf8 = str.utf8();
				CppString *gstr = machine.memory.memarray<CppString>(vaddr, 1);
				gstr->set_string(machine, vaddr, utf8.ptr(), utf8.length());
			} else if (index == 1) { // Get the string as a const char*, size_t struct.
				struct LocalBuffer {
					gaddr_t ptr;
					gaddr_t size;
				} *buffer = machine.memory.memarray<LocalBuffer>(vaddr, 1);
				CharString utf8 = str.utf8();
				const size_t size = utf8.length();
				// Allocate memory for the string in the guest memory.
				if (buffer->size < size) {
					buffer->size = size;
					if (buffer->ptr)
						machine.arena().free(buffer->ptr);
					buffer->ptr = machine.arena().malloc(buffer->size);
				}
				// Copy the string to the guest memory.
				machine.memory.memcpy(buffer->ptr, utf8.ptr(), buffer->size);
			} else if (index == 2) { // Get the string as a std::u32string.
				GuestStdU32String *gstr = machine.memory.memarray<GuestStdU32String>(vaddr, 1);
				gstr->set_string(machine, vaddr, str.ptr(), str.length());
			} else {
				ERR_PRINT("Invalid String conversion");
				throw std::runtime_error("Invalid String conversion");
			}
			break;
		}
		case String_Op::COMPARE: {
			unsigned *vother = machine.memory.memarray<unsigned>(vaddr, 1);
			const Variant *other = emu.get_scoped_variant(*vother).value();
			machine.set_result(str == other->operator String());
			break;
		}
		case String_Op::COMPARE_CSTR: {
			const std::string vother = machine.memory.memstring(vaddr);
			machine.set_result(str == String::utf8(vother.c_str(), vother.size()));
			break;
		}
		default:
			ERR_PRINT("Invalid String operation");
			throw std::runtime_error("Invalid String operation");
	}
}

APICALL(api_string_at) {
	auto [str_idx, index] = machine.sysargs<unsigned, int>();
	Sandbox &emu = riscv::emu(machine);
	SYS_TRACE("string_at", str_idx, index);

	std::optional<const Variant *> opt_str = emu.get_scoped_variant(str_idx);
	if (!opt_str.has_value() || opt_str.value()->get_type() != Variant::STRING) {
		ERR_PRINT("Invalid String object");
		throw std::runtime_error("Invalid String object");
	}
	::String str = opt_str.value()->operator String();

	if (index < 0 || index >= str.length()) {
		ERR_PRINT("String index out of bounds");
		throw std::runtime_error("String index out of bounds");
	}

	char32_t new_string = str[index];
	unsigned int new_varidx = emu.create_scoped_variant(Variant(std::move(new_string)));
	machine.set_result(new_varidx);
}

APICALL(api_string_size) {
	auto [str_idx] = machine.sysargs<unsigned>();
	Sandbox &emu = riscv::emu(machine);
	SYS_TRACE("string_size", str_idx);

	std::optional<const Variant *> opt_str = emu.get_scoped_variant(str_idx);
	if (!opt_str.has_value() || opt_str.value()->get_type() != Variant::STRING) {
		ERR_PRINT("Invalid String object");
		throw std::runtime_error("Invalid String object");
	}
	::String str = opt_str.value()->operator String();
	machine.set_result(str.length());
}

APICALL(api_string_append) {
	auto [str_idx, strview] = machine.sysargs<unsigned, std::string_view>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(10'000);
	SYS_TRACE("string_append", str_idx, String::utf8(strview.data(), strview.size()));

	Variant &var = emu.get_mutable_scoped_variant(str_idx);

	::String str = var.operator String();
	str += String::utf8(strview.data(), strview.size());
	var = Variant(std::move(str));
}

APICALL(api_timer_periodic) {
	auto [interval, oneshot, callback, capture, vret] = machine.sysargs<double, bool, gaddr_t, std::array<uint8_t, 32> *, GuestVariant *>();
	Sandbox &emu = riscv::emu(machine);
	PENALIZE(100'000); // Costly Timer node creation.
	SYS_TRACE("timer_periodic", interval, oneshot, callback, capture, vret);

	Timer *timer = memnew(Timer);
	timer->set_wait_time(interval);
	timer->set_one_shot(oneshot);
	Node *topnode = emu.get_tree_base();
	// Add the timer to the top node, as long as the Sandbox is in a tree.
	if (topnode != nullptr) {
		topnode->add_child(timer);
		timer->set_owner(topnode);
		timer->start();
	} else {
		timer->set_autostart(true);
	}
	// Copy the callback capture storage to the timer timeout callback.
	PackedByteArray capture_data;
	capture_data.resize(capture->size());
	memcpy(capture_data.ptrw(), capture->data(), capture->size());
	// Connect the timer to the guest callback function.
	Array args;
	args.push_back(Variant(timer));
	args.push_back(Variant(std::move(capture_data)));
	timer->connect("timeout", emu.vmcallable_address(callback, std::move(args)));
	// Return the timer object to the guest.
	vret->set_object(emu, timer);
}

APICALL(api_timer_stop) {
	throw std::runtime_error("timer_stop: Not implemented");
}

APICALL(api_callable_create) {
	auto [address, vargs] = machine.sysargs<gaddr_t, GuestVariant *>();
	Sandbox &emu = riscv::emu(machine);
	SYS_TRACE("callable_create", address, vargs);

	// Create a new callable object, using emu.vmcallable_address() to get the callable function.
	Array arguments;
	if (vargs->type != Variant::NIL) {
		// The argument idx is a Variant of another type.
		arguments.push_back(vargs->toVariant(emu));
	}
	Callable callable = emu.vmcallable_address(address, std::move(arguments));

	// Return the callable object to the guest.
	auto idx = emu.create_scoped_variant(Variant(std::move(callable)));
	machine.set_result(idx);
}

APICALL(api_load) {
	auto [path, g_result] = machine.sysargs<std::string_view, GuestVariant *>();
	Sandbox &emu = riscv::emu(machine);
	const String godot_path = String::utf8(path.data(), path.size());
	SYS_TRACE("load", godot_path, g_result);

	// Check if the path is allowed.
	if (!emu.is_allowed_resource(godot_path)) {
		ERR_PRINT("Resource path is not allowed: " + godot_path);
		throw std::runtime_error("Resource path is not allowed: " + std::string(path));
	}

	// Preload the resource from the given path.
	Ref<Resource> resource = ResourceLoader::load(godot_path);
	if (resource.is_null()) {
		ERR_PRINT("Failed to preload resource");
		// TODO: Return a null object instead?
		throw std::runtime_error("Failed to preload resource");
	}

	Variant result(std::move(resource));
	::Object *obj = result.operator Object *();

	// Return the result to the guest.
	emu.create_scoped_variant(std::move(result));
	g_result->set_object(emu, obj);
}

APICALL(api_sandbox_add) {
	// Add a new sandboxed property or public API method to the sandbox.
	Sandbox &emu = riscv::emu(machine);
	if (!emu.is_initializing()) {
		ERR_PRINT("Sandbox add called outside of initialization");
		throw std::runtime_error("Sandbox add called outside of initialization");
	}
	PENALIZE(100'000); // Costly Sandbox operations.
	// Check which operation it is.
	int method = machine.cpu.reg(10); // A0
	switch (method) {
		case 0: {
			// Add a new sandboxed property.
			auto [op_method, name, type, setter, getter, defval] = machine.sysargs<int, std::string_view, Variant::Type, gaddr_t, gaddr_t, GuestVariant *>();
			String utf8_name = String::utf8(name.data(), name.size());
			SYS_TRACE("sandbox_add", "property", utf8_name, int(type), setter, getter, defval->toVariant(emu));
			emu.add_property(utf8_name, type, setter, getter, defval->toVariant(emu));
		} break;
		case 1: {
			// Add a new sandboxed public API method. Name, address, description, return type and arguments.
			struct GuestFunctionExtra {
				gaddr_t desc;
				gaddr_t desc_len;
				gaddr_t ret;
				gaddr_t ret_len;
				gaddr_t args;
				gaddr_t args_len;
			};
			auto [op_method, name, address, g_extra] = machine.sysargs<int, std::string_view, gaddr_t, GuestFunctionExtra *>();
			SYS_TRACE("sandbox_add", "method", String::utf8(name.data(), name.size()));
			// Get the description, return type and arguments. We have a limited amount of registers,
			// so we will use zero-terminated strings for the description and return type.
			std::string_view description = machine.memory.memview(g_extra->desc, g_extra->desc_len);
			std::string_view return_type = machine.memory.memview(g_extra->ret, g_extra->ret_len);
			std::string_view arguments = machine.memory.memview(g_extra->args, g_extra->args_len);
			// Add the function to the ELFScript method list.
			if (Ref<ELFScript> program = emu.get_program(); program.is_valid()) {
				Dictionary func = Sandbox::create_public_api_function(name, address, description, return_type, arguments);
				if (func.size() > 0) {
					if (static_cast<unsigned>(program->functions.size()) >= Sandbox::MAX_PUBLIC_FUNCTIONS) {
						ERR_PRINT("Too many public functions in the Sandbox program");
						throw std::runtime_error("Too many public functions in the Sandbox program");
					}
					if (program->function_names.has(func["name"])) {
						// Remove the old function with the same name.
						for (int i = 0; i < program->functions.size(); i++) {
							Dictionary old_func = program->functions[i];
							if (old_func["name"].operator String() == func["name"].operator String()) {
								program->functions.remove_at(i);
								break;
							}
						}
					}
					program->functions.push_back(func);
				}
			}
			// Cache the function name hash with the address for faster lookup.
			emu.add_cached_address(String::utf8(name.data(), name.size()), address);
		} break;
		case 2: { // Set new exit address.
			SYS_TRACE("sandbox_add", "exit", machine.cpu.reg(11));
			const gaddr_t exit_address = machine.cpu.reg(11); // A1
			if (exit_address == 0 || (exit_address & 0x1) != 0) {
				ERR_PRINT("Invalid program exit address");
				throw std::runtime_error("Invalid program exit address");
			}
			const auto &exec = emu.machine().memory.exec_segment_for(exit_address);
			if (!exec->is_within(exit_address)) {
				ERR_PRINT("Invalid program exit address");
				throw std::runtime_error("Invalid program exit address");
			}
			emu.machine().memory.set_exit_address(exit_address);
			break;
		}
		default:
			WARN_PRINT("Unhandled sandbox add method: " + itos(method));
	}
}

} //namespace riscv

void Sandbox::initialize_syscalls() {
	using namespace riscv;

	// Initialize common Linux system calls
	machine().setup_linux_syscalls(false, false);
	// Initialize POSIX threads
	machine().setup_posix_threads();

	machine().on_unhandled_syscall = [](machine_t &machine, size_t syscall) {
#if defined(__linux__) // We only want to print these kinds of warnings on Linux.
		WARN_PRINT(("Unhandled system call: " + std::to_string(syscall)).c_str());
		auto &emu = riscv::emu(machine);
		PENALIZE(100'000); // Add to the instruction counter due to I/O.
#endif
		machine.set_result(-ENOSYS);
	};

	// Thread-safe global syscall handler installation (one-time only)
	static std::once_flag global_syscalls_initialized;
	std::call_once(global_syscalls_initialized, []() {
		// Add the Godot system calls (global registration).
		machine_t::install_syscall_handlers({
				{ ECALL_PRINT, api_print },
				{ ECALL_VCALL, api_vcall },
				{ ECALL_VEVAL, api_veval },
				{ ECALL_VASSIGN, api_vassign },
				{ ECALL_GET_OBJ, api_get_obj },
				{ ECALL_OBJ, api_obj },
				{ ECALL_OBJ_CALLP, api_obj_callp },
				{ ECALL_GET_NODE, api_get_node },
				{ ECALL_NODE, api_node },
				{ ECALL_NODE2D, api_node2d },
				{ ECALL_NODE3D, api_node3d },
				{ ECALL_THROW, api_throw },
				{ ECALL_IS_EDITOR, [](machine_t &machine) {
					 machine.set_result(::Engine::get_singleton()->is_editor_hint());
				 } },

				{ ECALL_VCREATE, api_vcreate },
				{ ECALL_VFETCH, api_vfetch },
				{ ECALL_VCLONE, api_vclone },
				{ ECALL_VSTORE, api_vstore },

				{ ECALL_ARRAY_OPS, api_array_ops },
				{ ECALL_ARRAY_AT, api_array_at },
				{ ECALL_ARRAY_SIZE, api_array_size },

				{ ECALL_DICTIONARY_OPS, api_dict_ops },

				{ ECALL_STRING_CREATE, api_string_create },
				{ ECALL_STRING_OPS, api_string_ops },
				{ ECALL_STRING_AT, api_string_at },
				{ ECALL_STRING_SIZE, api_string_size },
				{ ECALL_STRING_APPEND, api_string_append },

				{ ECALL_TIMER_PERIODIC, api_timer_periodic },
				{ ECALL_TIMER_STOP, api_timer_stop },

				{ ECALL_NODE_CREATE, api_node_create },

				{ ECALL_CALLABLE_CREATE, api_callable_create },

				{ ECALL_LOAD, api_load },

				{ ECALL_OBJ_PROP_GET, api_obj_property_get },
				{ ECALL_OBJ_PROP_SET, api_obj_property_set },

				{ ECALL_SANDBOX_ADD, api_sandbox_add },
		});

		// Set up custom instruction handling (global, one-time setup)
		using namespace riscv;
		static const Instruction<RISCV_ARCH> validated_syscall_instruction{
			[](CPU<RISCV_ARCH> &cpu, rv32i_instruction instr) {
				Machine<RISCV_ARCH>::syscall_handlers[instr.Itype.imm](cpu.machine());
			},
			[](char *buffer, size_t len, const CPU<RISCV_ARCH> &, rv32i_instruction instr) -> int {
				return snprintf(buffer, len,
						"DYNCALL: 4-byte idx=0x%X (inline, 0x%X)",
						uint32_t(instr.Itype.imm),
						instr.whole);
			}
		};
		// Override the machines unimplemented instruction handling,
		// in order to use the custom instruction instead.
		CPU<RISCV_ARCH>::on_unimplemented_instruction = [](rv32i_instruction instr) -> const Instruction<RISCV_ARCH> & {
			if (instr.opcode() == 0b1011011 && instr.Itype.rs1 == 0 && instr.Itype.rd == 0) {
				if (instr.Itype.imm < Machine<RISCV_ARCH>::syscall_handlers.size()) {
					return validated_syscall_instruction;
				}
			}
			return CPU<RISCV_ARCH>::get_unimplemented_instruction();
		};
	});

	// Add system calls from other modules (per-instance, after global setup).
	this->initialize_syscalls_2d();
	this->initialize_syscalls_3d();
}
