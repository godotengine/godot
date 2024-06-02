#include "jolt_shape_impl_3d.hpp"

#include "objects/jolt_shaped_object_impl_3d.hpp"
#include "shapes/jolt_custom_user_data_shape.hpp"

namespace {

constexpr float DEFAULT_SOLVER_BIAS = 0.0;

} // namespace

JoltShapeImpl3D::~JoltShapeImpl3D() = default;

void JoltShapeImpl3D::add_owner(JoltShapedObjectImpl3D* p_owner) {
	ref_counts_by_owner[p_owner]++;
}

void JoltShapeImpl3D::remove_owner(JoltShapedObjectImpl3D* p_owner) {
	if (--ref_counts_by_owner[p_owner] <= 0) {
		ref_counts_by_owner.erase(p_owner);
	}
}

void JoltShapeImpl3D::remove_self() {
	// `remove_owner` will be called when we `remove_shape`, so we need to copy the map since the
	// iterator would be invalidated from underneath us
	const auto ref_counts_by_owner_copy = ref_counts_by_owner;

	for (const auto& [owner, ref_count] : ref_counts_by_owner_copy) {
		owner->remove_shape(this);
	}
}

float JoltShapeImpl3D::get_solver_bias() const {
	return DEFAULT_SOLVER_BIAS;
}

void JoltShapeImpl3D::set_solver_bias(float p_bias) {
	if (!Math::is_equal_approx(p_bias, DEFAULT_SOLVER_BIAS)) {
		WARN_PRINT(vformat(
			"Custom solver bias for shapes is not supported by Godot Jolt. "
			"Any such value will be ignored. "
			"This shape belongs to %s.",
			_owners_to_string()
		));
	}
}

JPH::ShapeRefC JoltShapeImpl3D::try_build() {
	if (jolt_ref == nullptr) {
		jolt_ref = _build();
	}

	return jolt_ref;
}

void JoltShapeImpl3D::destroy() {
	jolt_ref = nullptr;

	for (const auto& [owner, ref_count] : ref_counts_by_owner) {
		owner->_shapes_changed();
	}
}

JPH::ShapeRefC JoltShapeImpl3D::with_scale(const JPH::Shape* p_shape, const Vector3& p_scale) {
	ERR_FAIL_NULL_D(p_shape);

	const JPH::ScaledShapeSettings shape_settings(p_shape, to_jolt(p_scale));
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Failed to scale shape with {scale=%v}. "
			"It returned the following error: '%s'.",
			p_scale,
			to_godot(shape_result.GetError())
		)
	);

	return shape_result.Get();
}

JPH::ShapeRefC JoltShapeImpl3D::with_basis_origin(
	const JPH::Shape* p_shape,
	const Basis& p_basis,
	const Vector3& p_origin
) {
	ERR_FAIL_NULL_D(p_shape);

	const JPH::RotatedTranslatedShapeSettings shape_settings(
		to_jolt(p_origin),
		to_jolt(p_basis),
		p_shape
	);

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Failed to offset shape with {basis=%s origin=%v}. "
			"It returned the following error: '%s'.",
			p_basis,
			p_origin,
			to_godot(shape_result.GetError())
		)
	);

	return shape_result.Get();
}

JPH::ShapeRefC JoltShapeImpl3D::with_center_of_mass_offset(
	const JPH::Shape* p_shape,
	const Vector3& p_offset
) {
	ERR_FAIL_NULL_D(p_shape);

	const JPH::OffsetCenterOfMassShapeSettings shape_settings(to_jolt(p_offset), p_shape);
	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Failed to offset center of mass with {offset=%v}. "
			"It returned the following error: '%s'.",
			p_offset,
			to_godot(shape_result.GetError())
		)
	);

	return shape_result.Get();
}

JPH::ShapeRefC JoltShapeImpl3D::with_center_of_mass(
	const JPH::Shape* p_shape,
	const Vector3& p_center_of_mass
) {
	ERR_FAIL_NULL_D(p_shape);

	const Vector3 center_of_mass_inner = to_godot(p_shape->GetCenterOfMass());
	const Vector3 center_of_mass_offset = p_center_of_mass - center_of_mass_inner;

	if (center_of_mass_offset == Vector3()) {
		return p_shape;
	}

	return with_center_of_mass_offset(p_shape, center_of_mass_offset);
}

JPH::ShapeRefC JoltShapeImpl3D::with_user_data(const JPH::Shape* p_shape, uint64_t p_user_data) {
	JoltCustomUserDataShapeSettings shape_settings(p_shape);
	shape_settings.mUserData = (JPH::uint64)p_user_data;

	const JPH::ShapeSettings::ShapeResult shape_result = shape_settings.Create();

	ERR_FAIL_COND_D_MSG(
		shape_result.HasError(),
		vformat(
			"Failed to override user data. "
			"It returned the following error: '%s'.",
			to_godot(shape_result.GetError())
		)
	);

	return shape_result.Get();
}

JPH::ShapeRefC JoltShapeImpl3D::without_custom_shapes(const JPH::Shape* p_shape) {
	switch (p_shape->GetSubType()) {
		case JoltCustomShapeSubType::EMPTY:
		case JoltCustomShapeSubType::RAY:
		case JoltCustomShapeSubType::MOTION: {
			// Replace unsupported shapes with a small sphere
			return new JPH::SphereShape(0.1f);
		}

		case JoltCustomShapeSubType::OVERRIDE_USER_DATA:
		case JoltCustomShapeSubType::DOUBLE_SIDED: {
			const auto* shape = static_cast<const JPH::DecoratedShape*>(p_shape);

			// Replace unsupported decorator shapes with the inner shape
			return without_custom_shapes(shape->GetInnerShape());
		}

		case JPH::EShapeSubType::StaticCompound: {
			const auto* shape = static_cast<const JPH::StaticCompoundShape*>(p_shape);

			JPH::StaticCompoundShapeSettings settings;

			for (const JPH::CompoundShape::SubShape& sub_shape : shape->GetSubShapes()) {
				settings.AddShape(
					shape->GetCenterOfMass() + sub_shape.GetPositionCOM() -
						sub_shape.GetRotation() * sub_shape.mShape->GetCenterOfMass(),
					sub_shape.GetRotation(),
					without_custom_shapes(sub_shape.mShape)
				);
			}

			const JPH::ShapeSettings::ShapeResult shape_result = settings.Create();

			ERR_FAIL_COND_D_MSG(
				shape_result.HasError(),
				vformat(
					"Failed to recreate static compound shape during filtering of custom shapes. "
					"It returned the following error: '%s'.",
					to_godot(shape_result.GetError())
				)
			);

			return shape_result.Get();
		}

		case JPH::EShapeSubType::MutableCompound: {
			const auto* shape = static_cast<const JPH::MutableCompoundShape*>(p_shape);

			JPH::MutableCompoundShapeSettings settings;

			for (const JPH::MutableCompoundShape::SubShape& sub_shape : shape->GetSubShapes()) {
				settings.AddShape(
					shape->GetCenterOfMass() + sub_shape.GetPositionCOM() -
						sub_shape.GetRotation() * sub_shape.mShape->GetCenterOfMass(),
					sub_shape.GetRotation(),
					without_custom_shapes(sub_shape.mShape)
				);
			}

			const JPH::ShapeSettings::ShapeResult shape_result = settings.Create();

			ERR_FAIL_COND_D_MSG(
				shape_result.HasError(),
				vformat(
					"Failed to recreate mutable compound shape during filtering of custom shapes. "
					"It returned the following error: '%s'.",
					to_godot(shape_result.GetError())
				)
			);

			return shape_result.Get();
		}

		case JPH::EShapeSubType::RotatedTranslated: {
			const auto* shape = static_cast<const JPH::RotatedTranslatedShape*>(p_shape);

			const JPH::Shape* inner_shape = shape->GetInnerShape();
			const JPH::ShapeRefC new_inner_shape = without_custom_shapes(inner_shape);

			if (inner_shape == new_inner_shape) {
				return p_shape;
			}

			return new JPH::RotatedTranslatedShape(
				shape->GetPosition(),
				shape->GetRotation(),
				new_inner_shape
			);
		}

		case JPH::EShapeSubType::Scaled: {
			const auto* shape = static_cast<const JPH::ScaledShape*>(p_shape);

			const JPH::Shape* inner_shape = shape->GetInnerShape();
			const JPH::ShapeRefC new_inner_shape = without_custom_shapes(inner_shape);

			if (inner_shape == new_inner_shape) {
				return p_shape;
			}

			return new JPH::ScaledShape(new_inner_shape, shape->GetScale());
		}

		case JPH::EShapeSubType::OffsetCenterOfMass: {
			const auto* shape = static_cast<const JPH::OffsetCenterOfMassShape*>(p_shape);

			const JPH::Shape* inner_shape = shape->GetInnerShape();
			const JPH::ShapeRefC new_inner_shape = without_custom_shapes(inner_shape);

			if (inner_shape == new_inner_shape) {
				return p_shape;
			}

			return new JPH::OffsetCenterOfMassShape(new_inner_shape, shape->GetOffset());
		}

		default: {
			return p_shape;
		}
	}
}

String JoltShapeImpl3D::_owners_to_string() const {
	const int32_t owner_count = ref_counts_by_owner.size();

	if (owner_count == 0) {
		return "'<unknown>' and 0 other object(s)";
	}

	const JoltShapedObjectImpl3D& random_owner = *ref_counts_by_owner.begin()->key;

	return vformat("'%s' and %d other object(s)", random_owner.to_string(), owner_count - 1);
}
