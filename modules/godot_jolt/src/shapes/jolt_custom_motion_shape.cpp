#include "jolt_custom_motion_shape.hpp"

namespace {

class JoltMotionConvexSupport final : public JPH::ConvexShape::Support {
public:
	JoltMotionConvexSupport(JPH::Vec3Arg p_motion, const JPH::ConvexShape::Support* p_inner_support)
		: motion(p_motion)
		, inner_support(p_inner_support) { }

	JPH::Vec3 GetSupport(JPH::Vec3Arg p_direction) const override {
		JPH::Vec3 support = inner_support->GetSupport(p_direction);

		if (p_direction.Dot(motion) > 0) {
			support += motion;
		}

		return support;
	}

	float GetConvexRadius() const override { return inner_support->GetConvexRadius(); }

private:
	JPH::Vec3 motion = JPH::Vec3::sZero();

	const JPH::ConvexShape::Support* inner_support = nullptr;
};

} // namespace

JPH::AABox JoltCustomMotionShape::GetLocalBounds() const {
	JPH::AABox aabb = inner_shape.GetLocalBounds();
	JPH::AABox aabb_translated = aabb;
	aabb_translated.Translate(motion);
	aabb.Encapsulate(aabb_translated);

	return aabb;
}

void JoltCustomMotionShape::GetSupportingFace(
	[[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id,
	[[maybe_unused]] JPH::Vec3Arg p_direction,
	[[maybe_unused]] JPH::Vec3Arg p_scale,
	[[maybe_unused]] JPH::Mat44Arg p_center_of_mass_transform,
	[[maybe_unused]] JPH::Shape::SupportingFace& p_vertices
) const {
	// HACK(mihe): This is technically called when using the enhanced internal edge removal, but
	// `JPH::InternalEdgeRemovingCollector` will only ever use the faces of the second shape in the
	// collision pair, and this shape will always be the first in the pair, so we can safely skip
	// this.
}

const JPH::ConvexShape::Support* JoltCustomMotionShape::GetSupportFunction(
	JPH::ConvexShape::ESupportMode p_mode,
	JPH::ConvexShape::SupportBuffer& p_buffer,
	JPH::Vec3Arg p_scale
) const {
	return new (&p_buffer) JoltMotionConvexSupport(
		motion,
		inner_shape.GetSupportFunction(p_mode, inner_support_buffer, p_scale)
	);
}
