/**************************************************************************/
/*  sandbox_syscalls_2d.cpp                                               */
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

#include "guest_datatypes.h"
#include "sandbox.h"
#include "syscalls.h"

#include "core/math/math_funcs.h"
#include "core/math/transform_2d.h"
#include "core/math/vector2.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
//#define ENABLE_SYSCALL_TRACE 1
#include "syscalls_helpers.hpp"

namespace riscv {

APICALL(api_vector2_length) {
	auto [dx, dy] = machine.sysargs<float, float>();
	const float length = std::sqrt(dx * dx + dy * dy);
	machine.set_result(length);
}

APICALL(api_vector2_normalize) {
	auto [dx, dy] = machine.sysargs<float, float>();
	const float length = std::sqrt(dx * dx + dy * dy);
	if (length > 0.0001f) // FLT_EPSILON?
	{
		dx /= length;
		dy /= length;
	}
	machine.set_result(dx, dy);
}

APICALL(api_vector2_rotated) {
	auto [dx, dy, angle] = machine.sysargs<float, float, float>();
	const float x = cos(angle) * dx - sin(angle) * dy;
	const float y = sin(angle) * dx + cos(angle) * dy;
	machine.set_result(x, y);
}

APICALL(api_vec2_ops) {
	auto [op, vec2] = machine.sysargs<Vec2_Op, Vector2 *>();

	// Integer arguments start from A2, and float arguments start from FA0.
	switch (op) {
		case Vec2_Op::NORMALIZE:
			vec2->normalize();
			break;
		case Vec2_Op::LENGTH: {
			double *result = machine.memory.memarray<double>(machine.cpu.reg(12), 1); // A2
			*result = vec2->length();
			break;
		}
		case Vec2_Op::LENGTH_SQ: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			double *result = machine.memory.memarray<double>(vaddr, 1);
			*result = vec2->length_squared();
			break;
		}
		case Vec2_Op::ANGLE: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			double *result = machine.memory.memarray<double>(vaddr, 1);
			*result = vec2->angle();
			break;
		}
		case Vec2_Op::ANGLE_TO: {
			const double angle = machine.cpu.registers().getfl(10).get<double>(); // FA0
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			double *result = machine.memory.memarray<double>(vaddr, 1);
			*result = vec2->angle_to(Vector2(cos(angle), sin(angle)));
			break;
		}
		case Vec2_Op::ANGLE_TO_POINT: {
			const double x = machine.cpu.registers().getfl(10).get<double>(); // FA0
			const double y = machine.cpu.registers().getfl(11).get<double>(); // FA1
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			double *result = machine.memory.memarray<double>(vaddr, 1);
			*result = vec2->angle_to(Vector2(x, y));
			break;
		}
		case Vec2_Op::PROJECT: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->project(*vec);
			break;
		}
		case Vec2_Op::DIRECTION_TO: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->direction_to(*vec);
			break;
		}
		case Vec2_Op::SLIDE: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->slide(*vec);
			break;
		}
		case Vec2_Op::BOUNCE: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->bounce(*vec);
			break;
		}
		case Vec2_Op::REFLECT: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->reflect(*vec);
			break;
		}
		case Vec2_Op::LIMIT_LENGTH: {
			const double length = machine.cpu.registers().getfl(10).get<double>(); // FA0
			*vec2 = vec2->limit_length(length);
			break;
		}
		case Vec2_Op::LERP: {
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // FA0
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			*vec2 = vec2->lerp(*vec, weight);
			break;
		}
		case Vec2_Op::CUBIC_INTERPOLATE: {
			Vector2 *v_b = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			Vector2 *vpre_a = machine.memory.memarray<Vector2>(machine.cpu.reg(13), 1); // A3
			Vector2 *vpost_b = machine.memory.memarray<Vector2>(machine.cpu.reg(14), 1); // A4
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // FA0
			*vec2 = vec2->cubic_interpolate(*v_b, *vpre_a, *vpost_b, weight);
			break;
		}
		case Vec2_Op::MOVE_TOWARD: {
			Vector2 *vec = machine.memory.memarray<Vector2>(machine.cpu.reg(12), 1); // A2
			const double delta = machine.cpu.registers().getfl(10).get<double>(); // FA0
			*vec2 = vec2->move_toward(*vec, delta);
			break;
		}
		default:
			ERR_PRINT("Invalid Vector2 operation");
			throw std::runtime_error("Invalid Vector2 operation");
	}
}

APICALL(api_transform2d_ops) {
	auto [idx, op] = machine.sysargs<unsigned, Transform2D_Op>();
	SYS_TRACE("transform2d_ops", idx, int(op), vres);
	Sandbox &emu = riscv::emu(machine);

	if (op == Transform2D_Op::IDENTITY) {
		const gaddr_t vaddr = machine.cpu.reg(12); // A2
		GuestVariant *vres = machine.memory.memarray<GuestVariant>(vaddr, 1);
		vres->create(emu, Transform2D());
		return;
	}

	std::optional<const Variant *> opt_t = emu.get_scoped_variant(idx);
	if (!opt_t.has_value() || opt_t.value()->get_type() != Variant::TRANSFORM2D) {
		ERR_PRINT("Invalid Transform2D object");
		throw std::runtime_error("Invalid Transform2D object");
	}
	const Variant *t_variant = *opt_t;
	::Transform2D t = t_variant->operator Transform2D();

	// Additional integers start at A2 (12), and floats start at FA0 (10).
	switch (op) {
		case Transform2D_Op::GET_COLUMN: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			Vector2 *v = machine.memory.memarray<Vector2>(vaddr, 1);
			const int column = machine.cpu.reg(13); // A3
			if (column < 0 || column >= 3) {
				ERR_PRINT("Invalid Transform2D column");
				throw std::runtime_error("Invalid Transform2D column");
			}

			*v = t.columns[column];
			break;
		}
		case Transform2D_Op::SET_COLUMN: {
			unsigned *vres = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const int column = machine.cpu.reg(13); // A3
			const gaddr_t vaddr = machine.cpu.reg(14); // A4
			const Vector2 *v = machine.memory.memarray<Vector2>(vaddr, 1);
			if (column < 0 || column >= 3) {
				ERR_PRINT("Invalid Transform2D column");
				throw std::runtime_error("Invalid Transform2D column");
			}

			t.columns[column] = *v;
			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t));
			break;
		}
		case Transform2D_Op::ROTATED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);
			const double angle = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Rotate the transform by the given angle, return a new transform.
			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.rotated(angle)));
			break;
		}
		case Transform2D_Op::SCALED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);
			const gaddr_t v2addr = machine.cpu.reg(13); // A3
			const Vector2 *scale = machine.memory.memarray<Vector2>(v2addr, 1);

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.scaled(*scale)));
			break;
		}
		case Transform2D_Op::TRANSLATED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);
			const gaddr_t v2addr = machine.cpu.reg(13); // A3
			const Vector2 *offset = machine.memory.memarray<Vector2>(v2addr, 1);

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.translated(*offset)));
			break;
		}
		case Transform2D_Op::INVERTED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.inverse()));
			break;
		}
		case Transform2D_Op::AFFINE_INVERTED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.affine_inverse()));
			break;
		}
		case Transform2D_Op::ORTHONORMALIZED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vres = machine.memory.memarray<unsigned>(vaddr, 1);

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.orthonormalized()));
			break;
		}
		case Transform2D_Op::INTERPOLATE_WITH: {
			unsigned *vres = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned t2_idx = machine.cpu.reg(13); // A3
			const Transform2D to = emu.get_scoped_variant(t2_idx).value()->operator Transform2D();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			*vres = emu.try_reuse_assign_variant(idx, *t_variant, *vres, Variant(t.interpolate_with(to, weight)));
			break;
		}
		default:
			ERR_PRINT("Invalid Transform2D operation");
			throw std::runtime_error("Invalid Transform2D operation");
	}
}

template <typename Float>
static void api_math_op(machine_t &machine) {
	auto [op, arg1] = machine.sysargs<Math_Op, Float>();
	SYS_TRACE("math_op", int(op), arg1);

	switch (op) {
		case Math_Op::SIN:
			machine.set_result(Float(sin(arg1)));
			break;
		case Math_Op::COS:
			machine.set_result(Float(cos(arg1)));
			break;
		case Math_Op::TAN:
			machine.set_result(Float(tan(arg1)));
			break;
		case Math_Op::ASIN:
			machine.set_result(Float(asin(arg1)));
			break;
		case Math_Op::ACOS:
			machine.set_result(Float(acos(arg1)));
			break;
		case Math_Op::ATAN:
			machine.set_result(Float(atan(arg1)));
			break;
		case Math_Op::ATAN2: {
			Float arg2 = machine.cpu.registers().getfl(11).get<Float>(); // fa1
			machine.set_result(Float(atan2(arg1, arg2)));
			break;
		}
		case Math_Op::POW: {
			Float arg2 = machine.cpu.registers().getfl(11).get<Float>(); // fa1
			machine.set_result(Float(pow(arg1, arg2)));
			break;
		}
		default:
			ERR_PRINT("Invalid Math operation");
			throw std::runtime_error("Invalid Math operation");
	}
}

template <typename Float>
static void api_lerp_op(machine_t &machine) {
	auto [op, arg1, arg2, arg3] = machine.sysargs<Lerp_Op, Float, Float, Float>();
	SYS_TRACE("lerp_op", int(op), arg1, arg2, arg3);

	switch (op) {
		case Lerp_Op::LERP: {
			const Float t = arg3; // t is the interpolation factor.
			machine.set_result(arg1 * (Float(1.0) - t) + arg2 * t);
			break;
		}
		case Lerp_Op::SMOOTHSTEP: {
			const Float a = arg1; // a is the start value.
			const Float b = arg2; // b is the end value.
			const Float t = CLAMP<Float>((arg3 - a) / (b - a), Float(0.0), Float(1.0));
			machine.set_result(t * t * (Float(3.0) - Float(2.0) * t));
			break;
		}
		case Lerp_Op::CLAMP:
			machine.set_result(CLAMP<Float>(arg1, arg2, arg3));
			break;
		case Lerp_Op::SLERP: { // Spherical linear interpolation
			const Float a = arg1; // a is the start value.
			const Float b = arg2; // b is the end value.
			const Float t = arg3; // t is the interpolation factor.
			const Float dot = a * b + Float(1.0);
			if (dot > Float(0.9995)) {
				machine.set_result(a);
			} else {
				const Float theta = acos(CLAMP<Float>(dot, Float(-1.0), Float(1.0)));
				const Float sin_theta = sin(theta);
				machine.set_result((a * sin((Float(1.0) - t) * theta) + b * sin(t * theta)) / sin_theta);
			}
			break;
		}
		default:
			ERR_PRINT("Invalid Lerp operation");
			throw std::runtime_error("Invalid Lerp operation");
	}
} // api_lerp_op

} // namespace riscv

void Sandbox::initialize_syscalls_2d() {
	using namespace riscv;

	// Add the Godot system calls.
	machine_t::install_syscall_handlers({
			{ ECALL_SINCOS, [](machine_t &machine) {
				 float angle = machine.cpu.registers().getfl(10).get<float>(); // fa0
				 machine.set_result(sin(angle), cos(angle));
			 } },
			{ ECALL_VEC2_LENGTH, api_vector2_length },
			{ ECALL_VEC2_NORMALIZED, api_vector2_normalize },
			{ ECALL_VEC2_ROTATED, api_vector2_rotated },
			{ ECALL_VEC2_OPS, api_vec2_ops },
			{ ECALL_TRANSFORM_2D_OPS, api_transform2d_ops },
			{ ECALL_MATH_OP32, api_math_op<float> },
			{ ECALL_MATH_OP64, api_math_op<double> },
			{ ECALL_LERP_OP32, api_lerp_op<float> },
			{ ECALL_LERP_OP64, api_lerp_op<double> },
	});
}
