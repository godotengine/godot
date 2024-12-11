#pragma once
#include "../common.h"
class JoltObjectImpl3D;

class JoltGroupFilter final : public JPH::GroupFilter {
public:
	inline static JoltGroupFilter* instance = nullptr;

	static void encode_object(
		const JoltObjectImpl3D* p_object,
		JPH::CollisionGroup::GroupID& p_group_id,
		JPH::CollisionGroup::SubGroupID& p_sub_group_id
	);

	static const JoltObjectImpl3D* decode_object(
		JPH::CollisionGroup::GroupID p_group_id,
		JPH::CollisionGroup::SubGroupID p_sub_group_id
	);

private:
	bool CanCollide(const JPH::CollisionGroup& p_group1, const JPH::CollisionGroup& p_group2)
		const override;
};
