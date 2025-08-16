#include "guest_datatypes.h"
#include "syscalls.h"

#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/variant.hpp>
//#define ENABLE_SYSCALL_TRACE 1
#include "syscalls_helpers.hpp"

namespace riscv {

APICALL(api_vec3_ops) {
	auto [v, v2addr, op] = machine.sysargs<Vector3 *, gaddr_t, Vec3_Op>();
	SYS_TRACE("vec3_ops", v, v2addr, int(op));

	switch (op) {
		case Vec3_Op::HASH: {
			gaddr_t seed = 0;
			hash_combine(seed, std::hash<float>{}(v->x));
			hash_combine(seed, std::hash<float>{}(v->y));
			hash_combine(seed, std::hash<float>{}(v->z));
			machine.set_result(seed);
			break;
		}
		case Vec3_Op::LENGTH: {
			machine.set_result(sqrt(v->x * v->x + v->y * v->y + v->z * v->z));
			break;
		}
		case Vec3_Op::NORMALIZE: {
			const float length = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
			if (length > 0.0001f) // FLT_EPSILON?
			{
				v->x /= length;
				v->y /= length;
				v->z /= length;
			}
			break;
		}
		case Vec3_Op::CROSS: {
			Vector3 *v2 = machine.memory.memarray<Vector3>(v2addr, 1);
			const gaddr_t resaddr = machine.cpu.reg(13); // a3
			Vector3 *res = machine.memory.memarray<Vector3>(resaddr, 1);
			res->x = v->y * v2->z - v->z * v2->y;
			res->y = v->z * v2->x - v->x * v2->z;
			res->z = v->x * v2->y - v->y * v2->x;
			break;
		}
		case Vec3_Op::DOT: {
			Vector3 *v2 = machine.memory.memarray<Vector3>(v2addr, 1);
			machine.set_result(v->x * v2->x + v->y * v2->y + v->z * v2->z);
			break;
		}
		case Vec3_Op::ANGLE_TO: {
			Vector3 *v2 = machine.memory.memarray<Vector3>(v2addr, 1);
			machine.set_result(float(v->angle_to(*v2)));
			break;
		}
		case Vec3_Op::DISTANCE_TO: {
			Vector3 *v2 = machine.memory.memarray<Vector3>(v2addr, 1);
			const float dx = v->x - v2->x;
			const float dy = v->y - v2->y;
			const float dz = v->z - v2->z;
			machine.set_result(sqrt(dx * dx + dy * dy + dz * dz));
			break;
		}
		case Vec3_Op::DISTANCE_SQ_TO: {
			Vector3 *v2 = machine.memory.memarray<Vector3>(v2addr, 1);
			const float dx = v->x - v2->x;
			const float dy = v->y - v2->y;
			const float dz = v->z - v2->z;
			machine.set_result(float(dx * dx + dy * dy + dz * dz));
			break;
		}
		case Vec3_Op::FLOOR: {
			machine.set_result(floorf(v->x), floorf(v->y), floorf(v->z));
			break;
		}
		default:
			ERR_PRINT("Invalid Vec3 operation");
			throw std::runtime_error("Invalid Vec3 operation");
	}
}

APICALL(api_transform3d_ops) {
	auto [idx, op] = machine.sysargs<unsigned, Transform3D_Op>();
	SYS_TRACE("transform3d_ops", idx, int(op));
	Sandbox &emu = riscv::emu(machine);

	if (op == Transform3D_Op::CREATE) {
		const gaddr_t vaddr = machine.cpu.reg(12); // A2 (Result index)
		unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
		const Vector3 *v3 = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3
		unsigned b_idx = machine.cpu.reg(14); // A4 (Basis index)

		// Get the basis from the given index.
		const Basis basis = emu.get_scoped_variant(b_idx).value()->operator Basis();

		// Create a new scoped Variant with the transform.
		*vidx = emu.create_scoped_variant(Variant(Transform3D(basis, *v3)));
		return;
	} else if (op == Transform3D_Op::IDENTITY) {
		const gaddr_t vaddr = machine.cpu.reg(12); // A2
		unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

		// Create a new scoped Variant with the identity transform.
		*vidx = emu.create_scoped_variant(Variant(Transform3D()));
		return;
	}

	std::optional<const Variant *> opt_t = emu.get_scoped_variant(idx);
	if (!opt_t.has_value() || (*opt_t)->get_type() != Variant::TRANSFORM3D) {
		ERR_PRINT("Invalid Transform3D object");
		throw std::runtime_error("Invalid Transform3D object: " + std::to_string(int32_t(idx)));
	}
	const Variant *t_variant = *opt_t;
	godot::Transform3D t = t_variant->operator Transform3D();

	// Additional integers start at A2 (12), and floats start at FA0 (10).
	switch (op) {
		case Transform3D_Op::ASSIGN: {
			unsigned *new_idx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned t2_idx = machine.cpu.reg(13); // A3
			const Variant *t2 = emu.get_scoped_variant(t2_idx).value();

			// Smart-assign the given transform to the current Variant.
			*new_idx = emu.try_reuse_assign_variant(idx, *t_variant, *new_idx, *t2);
			break;
		}
		case Transform3D_Op::GET_BASIS: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
			const Basis &basis = t.basis;

			// Create a new scoped Variant with the basis.
			*vidx = emu.create_scoped_variant(Variant(basis));
			break;
		}
		case Transform3D_Op::SET_BASIS: {
			unsigned *new_idx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned b_idx = machine.cpu.reg(13); // A3
			const Variant *vbasis = emu.get_scoped_variant(b_idx).value();

			// Set the basis of the current transform.
			t.basis = vbasis->operator Basis();
			*new_idx = emu.try_reuse_assign_variant(idx, *t_variant, *new_idx, Variant(t));
			break;
		}
		case Transform3D_Op::GET_ORIGIN: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			Vector3 *vres = machine.memory.memarray<Vector3>(vaddr, 1);

			*vres = t.origin;
			break;
		}
		case Transform3D_Op::SET_ORIGIN: {
			unsigned *new_idx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *origin = machine.memory.memarray<Vector3>(v3addr, 1);

			// Set the origin of the current transform.
			t.origin = *origin;
			*new_idx = emu.try_reuse_assign_variant(idx, *t_variant, *new_idx, Variant(t));
			break;
		}
		case Transform3D_Op::ROTATED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *axis = machine.memory.memarray<Vector3>(v3addr, 1);
			const double angle = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Rotate the transform by the given axis and angle, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.rotated(*axis, angle)));
			break;
		}
		case Transform3D_Op::ROTATED_LOCAL: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *axis = machine.memory.memarray<Vector3>(v3addr, 1);
			const double angle = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Rotate the transform by the given axis and angle, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.rotated_local(*axis, angle)));
			break;
		}
		case Transform3D_Op::SCALED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *scale = machine.memory.memarray<Vector3>(v3addr, 1);

			// Scale the transform by the given scale, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.scaled(*scale)));
			break;
		}
		case Transform3D_Op::SCALED_LOCAL: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *scale = machine.memory.memarray<Vector3>(v3addr, 1);

			// Scale the transform by the given scale in local space, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.scaled_local(*scale)));
			break;
		}
		case Transform3D_Op::TRANSLATED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *offset = machine.memory.memarray<Vector3>(v3addr, 1);

			// Translate the transform by the given offset, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.translated(*offset)));
			break;
		}
		case Transform3D_Op::TRANSLATED_LOCAL: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *offset = machine.memory.memarray<Vector3>(v3addr, 1);

			// Translate the transform by the given offset in local space, return a new transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.translated_local(*offset)));
			break;
		}
		case Transform3D_Op::INVERTED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Return the inverse of the current transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.inverse()));
			break;
		}
		case Transform3D_Op::ORTHONORMALIZED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Return the orthonormalized version of the current transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.orthonormalized()));
			break;
		}
		case Transform3D_Op::LOOKING_AT: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
			const Vector3 *target = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3
			const Vector3 *up = machine.memory.memarray<Vector3>(machine.cpu.reg(14), 1); // A4

			// Return the transform looking at the target with the given up vector.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t.looking_at(*target, *up)));
			break;
		}
		case Transform3D_Op::INTERPOLATE_WITH: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const unsigned t2_idx = machine.cpu.reg(13); // A3
			const Transform3D to = emu.get_scoped_variant(t2_idx).value()->operator Transform3D();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			t = t.interpolate_with(to, weight);
			// Return the interpolated transform between the current and the given transform.
			*vidx = emu.try_reuse_assign_variant(idx, *t_variant, *vidx, Variant(t));
			break;
		}
		default:
			ERR_PRINT("Invalid Transform3D operation");
			throw std::runtime_error("Invalid Transform3D operation");
	}
}

APICALL(api_basis_ops) {
	auto [idx, op] = machine.sysargs<unsigned, Basis_Op>();
	SYS_TRACE("basis_ops", idx, int(op));
	Sandbox &emu = riscv::emu(machine);

	if (op == Basis_Op::IDENTITY) {
		const gaddr_t vaddr = machine.cpu.reg(12); // A2
		unsigned *tres = machine.memory.memarray<unsigned>(vaddr, 1);

		*tres = emu.create_scoped_variant(Variant(Basis()));
		return;
	} else if (op == Basis_Op::CREATE) {
		const gaddr_t vaddr = machine.cpu.reg(12); // A2
		unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
		const Vector3 *v1 = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3
		const Vector3 *v2 = machine.memory.memarray<Vector3>(machine.cpu.reg(14), 1); // A4
		const Vector3 *v3 = machine.memory.memarray<Vector3>(machine.cpu.reg(15), 1); // A5

		// Create a new scoped Variant with the given vectors.
		*vidx = emu.create_scoped_variant(Variant(Basis(*v1, *v2, *v3)));
		return;
	}

	std::optional<const Variant *> opt_b = emu.get_scoped_variant(idx);
	if (!opt_b.has_value() || opt_b.value()->get_type() != Variant::BASIS) {
		ERR_PRINT("Invalid Basis object");
		throw std::runtime_error("Invalid Basis object");
	}
	const Variant *b_variant = *opt_b;
	godot::Basis b = b_variant->operator Basis();

	// Additional integers start at A2 (12), and floats start at FA0 (10).
	switch (op) {
		case Basis_Op::ASSIGN: {
			unsigned *new_idx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned b_idx = machine.cpu.reg(13); // A3
			const Variant *new_value = emu.get_scoped_variant(b_idx).value();

			// Smart-assign the given basis to the current Variant.
			*new_idx = emu.try_reuse_assign_variant(idx, *b_variant, *new_idx, *new_value);
			break;
		}
		case Basis_Op::GET_ROW: {
			const unsigned row = machine.cpu.reg(12); // A2
			if (row < 0 || row >= 3) {
				ERR_PRINT("Invalid Basis row");
				throw std::runtime_error("Invalid Basis row " + std::to_string(row));
			}
			Vector3 *vres = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3

			*vres = b[row];
			break;
		}
		case Basis_Op::SET_ROW: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned row = machine.cpu.reg(13); // A3
			if (row < 0 || row >= 3) {
				ERR_PRINT("Invalid Basis row");
				throw std::runtime_error("Invalid Basis row " + std::to_string(row));
			}

			const gaddr_t v3addr = machine.cpu.reg(14); // A4
			const Vector3 *v = machine.memory.memarray<Vector3>(v3addr, 1);

			// Set the row of the current basis.
			b[row] = *v;
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b));
			break;
		}
		case Basis_Op::GET_COLUMN: {
			const unsigned column = machine.cpu.reg(12); // A2
			if (column < 0 || column >= 3) {
				ERR_PRINT("Invalid Basis column");
				throw std::runtime_error("Invalid Basis column " + std::to_string(column));
			}
			Vector3 *vres = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3

			*vres = b.get_column(column);
			break;
		}
		case Basis_Op::SET_COLUMN: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned column = machine.cpu.reg(13); // A3
			if (column < 0 || column >= 3) {
				ERR_PRINT("Invalid Basis column");
				throw std::runtime_error("Invalid Basis column " + std::to_string(column));
			}
			const Vector3 *v = machine.memory.memarray<Vector3>(machine.cpu.reg(14), 1); // A4

			// Set the column of the current basis.
			b.set_column(column, *v);
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b));
			break;
		}
		case Basis_Op::TRANSPOSED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Return the transposed version of the current basis.
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b.transposed()));
			break;
		}
		case Basis_Op::INVERTED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Return the inverse of the current basis.
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b.inverse()));
			break;
		}
		case Basis_Op::DETERMINANT: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			double *res = machine.memory.memarray<double>(vaddr, 1);

			// Return the determinant of the current basis.
			*res = b.determinant();
			break;
		}
		case Basis_Op::ROTATED: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			const gaddr_t v3addr = machine.cpu.reg(13); // A3
			const Vector3 *axis = machine.memory.memarray<Vector3>(v3addr, 1);
			const double angle = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Rotate the basis by the given axis and angle, return a new basis.
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b.rotated(*axis, angle)));
			break;
		}
		case Basis_Op::LERP: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Get the second basis (from scoped Variant index) to interpolate with.
			const unsigned b_idx = machine.cpu.reg(13); // A3
			const Basis b2 = emu.get_scoped_variant(b_idx).value()->operator Basis();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Linearly interpolate between the two bases, return a new basis.
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b.lerp(b2, weight)));
			break;
		}
		case Basis_Op::SLERP: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Get the second basis (from scoped Variant index) to interpolate with.
			const unsigned b_idx = machine.cpu.reg(13); // A3
			const Basis b2 = emu.get_scoped_variant(b_idx).value()->operator Basis();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Spherically interpolate between the two bases, return a new basis.
			*vidx = emu.try_reuse_assign_variant(idx, *b_variant, *vidx, Variant(b.slerp(b2, weight)));
			break;
		}
		default:
			ERR_PRINT("Invalid Basis operation");
			throw std::runtime_error("Invalid Basis operation: " + std::to_string(int(op)));
	}
}

APICALL(api_quat_ops) {
	auto [idx, op] = machine.sysargs<unsigned, Quaternion_Op>();
	SYS_TRACE("quat_ops", idx, int(op));
	Sandbox &emu = riscv::emu(machine);
	SYS_TRACE("quat_ops", idx, int(op));
	printf("Quaternion operation %d %d\n", idx, int(op));

	if (op == Quaternion_Op::CREATE) {
		switch (idx) {
			case 0: { // IDENTITY
				const gaddr_t vaddr = machine.cpu.reg(12); // A2
				GuestVariant *vres = machine.memory.memarray<GuestVariant>(vaddr, 1);

				vres->create(emu, Quaternion());
				return;
			}
			case 1: { // FROM_AXIS_ANGLE
				const gaddr_t vaddr = machine.cpu.reg(12); // A2
				unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
				const Vector3 *axis = machine.memory.memarray<Vector3>(machine.cpu.reg(13), 1); // A3
				const double angle = machine.cpu.registers().getfl(10).get<double>(); // fa0

				// Create a new scoped Variant with the given axis and angle.
				*vidx = emu.create_scoped_variant(Variant(Quaternion(*axis, angle)));
				return;
			}
			default:
				ERR_PRINT("Invalid Quaternion constructor");
				throw std::runtime_error("Invalid Quaternion constructor: " + std::to_string(idx));
		}
		__builtin_unreachable();
	}

	// Outside of CREATE (constructor) operations, idx is the scoped Variant index for the Quaternion.
	std::optional<const Variant *> opt_q = emu.get_scoped_variant(idx);
	if (!opt_q.has_value() || opt_q.value()->get_type() != Variant::QUATERNION) {
		ERR_PRINT("Invalid Quaternion object");
		throw std::runtime_error("Invalid Quaternion object");
	}
	const Variant *q_variant = *opt_q;
	godot::Quaternion q = q_variant->operator Quaternion();

	// Additional integers start at A2 (12), and floats start at FA0 (10).
	switch (op) {
		case Quaternion_Op::ASSIGN: {
			unsigned *new_idx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2
			const unsigned q_idx = machine.cpu.reg(13); // A3
			const Variant *new_value = emu.get_scoped_variant(q_idx).value();
			printf("Quaternion assign %d %d\n", idx, q_idx);

			// Smart-assign the given quaternion to the current Variant.
			*new_idx = emu.try_reuse_assign_variant(idx, *q_variant, *new_idx, *new_value);
			return;
		}
		case Quaternion_Op::DOT: {
			const unsigned q2_idx = machine.cpu.reg(12); // A2
			const Quaternion q2 = emu.get_scoped_variant(q2_idx).value()->operator Quaternion();
			double *res = machine.memory.memarray<double>(machine.cpu.reg(13), 1); // A3

			// Return the dot product of the two quaternions.
			*res = q.dot(q2);
			return;
		}
		case Quaternion_Op::LENGTH_SQUARED: {
			double *res = machine.memory.memarray<double>(machine.cpu.reg(12), 1); // A2

			// Return the squared length of the quaternion.
			*res = q.length_squared();
			return;
		}
		case Quaternion_Op::LENGTH: {
			double *res = machine.memory.memarray<double>(machine.cpu.reg(12), 1); // A2

			// Return the length of the quaternion.
			*res = q.length();
			return;
		}
		case Quaternion_Op::NORMALIZE: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2

			// Create a new normalized quaternion and store it in the destination Variant.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.normalized()));
			return;
		}
		case Quaternion_Op::INVERSE: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2

			// Create a new inverse quaternion and store it in the destination Variant.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.inverse()));
			return;
		}
		case Quaternion_Op::LOG: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2

			// Create a new logarithmic quaternion and store it in the destination Variant.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.log()));
			return;
		}
		case Quaternion_Op::EXP: {
			unsigned *vidx = machine.memory.memarray<unsigned>(machine.cpu.reg(12), 1); // A2

			// Create a new exponential quaternion and store it in the destination Variant.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.exp()));
			return;
		}
		case Quaternion_Op::AT: {
			const unsigned idx = machine.cpu.reg(12); // A2
			if (idx < 0 || idx >= 4) {
				ERR_PRINT("Invalid Quaternion index");
				throw std::runtime_error("Invalid Quaternion index: " + std::to_string(idx));
			}
			double *res = machine.memory.memarray<double>(machine.cpu.reg(13), 1); // A3

			*res = q[idx];
			return;
		}
		case Quaternion_Op::GET_AXIS: {
			Vector3 *vres = machine.memory.memarray<Vector3>(machine.cpu.reg(12), 1); // A2

			*vres = q.get_axis();
			return;
		}
		case Quaternion_Op::GET_ANGLE: {
			double *res = machine.memory.memarray<double>(machine.cpu.reg(12), 1); // A2

			// Return the angle of the quaternion.
			*res = q.get_angle();
			return;
		}
		case Quaternion_Op::MUL: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);
			const unsigned q2_idx = machine.cpu.reg(13); // A3
			const Quaternion q2 = emu.get_scoped_variant(q2_idx).value()->operator Quaternion();

			// Multiply the two quaternions, return a new quaternion.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q * q2));
			return;
		}
		case Quaternion_Op::SLERP: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Get the second quaternion (from scoped Variant index) to interpolate with.
			const unsigned q2_idx = machine.cpu.reg(13); // A3
			const Quaternion q2 = emu.get_scoped_variant(q2_idx).value()->operator Quaternion();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Spherically interpolate between the two quaternions, return a new quaternion.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.slerp(q2, weight)));
			return;
		}
		case Quaternion_Op::SLERPNI: {
			const gaddr_t vaddr = machine.cpu.reg(12); // A2
			unsigned *vidx = machine.memory.memarray<unsigned>(vaddr, 1);

			// Get the second quaternion (from scoped Variant index) to interpolate with.
			const unsigned q2_idx = machine.cpu.reg(13); // A3
			const Quaternion q2 = emu.get_scoped_variant(q2_idx).value()->operator Quaternion();
			const double weight = machine.cpu.registers().getfl(10).get<double>(); // fa0

			// Spherically interpolate between the two quaternions, return a new quaternion.
			*vidx = emu.try_reuse_assign_variant(idx, *q_variant, *vidx, Variant(q.slerpni(q2, weight)));
			return;
		}
		default:
			ERR_PRINT("Invalid Quaternion operation");
			throw std::runtime_error("Invalid Quaternion operation: " + std::to_string(int(op)));
	}
}

} // namespace riscv

void Sandbox::initialize_syscalls_3d() {
	using namespace riscv;

	// Add the Godot system calls.
	machine_t::install_syscall_handlers({
			{ ECALL_VEC3_OPS, api_vec3_ops },
			{ ECALL_TRANSFORM_3D_OPS, api_transform3d_ops },
			{ ECALL_BASIS_OPS, api_basis_ops },
			{ ECALL_QUAT_OPS, api_quat_ops },
	});
}
