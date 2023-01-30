#ifndef COMBAT_SERVER_MT_H
#define COMBAT_SERVER_MT_H

#include "combat_server.h"
#include "combat_server_mt_common.gen.h"

class SentrienceWrapMT : public Sentrience {
	_PRIVATE_MEMBERS_
// ---------------------------------
public:
	SentrienceWrapMT();
	~SentrienceWrapMT();

	FUNCRID_0A(recording);
	FUNC_1A_R(bool, recording_assert, RID_TYPE);
	FUNC_2A(recording_add_simulation, RID_TYPE, RID_TYPE);
	FUNC_2A(recording_remove_simulation, RID_TYPE, RID_TYPE);
	FUNC_2A_R(Dictionary, recording_compile_by_simulation, RID_TYPE, RID_TYPE);
	FUNC_1A_R(bool, recording_start, RID_TYPE);
	FUNC_1A_R(bool, recording_end, RID_TYPE);
	FUNC_1A_R(bool, recording_running, RID_TYPE);
	FUNC_1A_S(recording_purge, RID_TYPE);

	FUNCRID_0A(simulation);
	FUNC_1A_R(bool, simulation_assert, RID_TYPE);
	FUNC_2A(simulation_set_active, RID_TYPE, bool);
	FUNC_1A_R(bool, simulation_is_active, RID_TYPE);
	FUNC_1A_R(Array, simulation_get_all_engagements, RID_TYPE);
	FUNC_1A_R(Array, simulation_get_all_active_engagements, RID_TYPE);
	FUNC_2A_S(simulation_set_profile, RID_TYPE, const Ref<RCSSimulationProfile>&);
	FUNC_1A_R(Ref<RCSSimulationProfile>, simulation_get_profile, RID_TYPE);
	FUNC_2A(simulation_bind_recording, RID_TYPE, RID_TYPE);
	FUNC_1A(simulation_unbind_recording, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_combatant, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_squad, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_team, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_radar, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_engagement, RID_TYPE);
	FUNC_1A_R(uint32_t, simulation_count_all_instances, RID_TYPE);

	FUNCRID_0A(combatant);
	FUNC_1A_R(bool, combatant_assert, RID_TYPE);
	FUNC_2A_S(combatant_set_simulation, RID_TYPE, RID_TYPE);
	FUNC_1A_R(RID_TYPE, combatant_get_simulation, RID_TYPE);
	FUNC_2A_R(bool, combatant_is_squad, RID_TYPE, RID_TYPE);
	FUNC_2A_R(bool, combatant_is_team, RID_TYPE, RID_TYPE);
	FUNC_1A_R(Array, combatant_get_involving_engagements, RID_TYPE);
	FUNC_2A(combatant_set_local_transform, RID_TYPE, Transform);
	FUNC_2A(combatant_set_space_transform, RID_TYPE, Transform);
	FUNC_1A_R(Transform, combatant_get_space_transform, RID_TYPE);
	FUNC_1A_R(Transform, combatant_get_local_transform, RID_TYPE);
	FUNC_1A_R(Transform, combatant_get_combined_transform, RID_TYPE);
	FUNC_1A_R(uint32_t, combatant_get_status, RID_TYPE);
	FUNC_2A(combatant_set_iid, RID_TYPE, const uint64_t&);
	FUNC_1A_R(uint32_t, combatant_get_status, RID_TYPE);
	FUNC_2A(combatant_set_space_transform, RID_TYPE, Transform);
	FUNC_1A_R(uint32_t, combatant_get_status, RID_TYPE);
	FUNC_2A_R(uint32_t, combatant_get_status, RID_TYPE);
	FUNC_2A_S(combatant_set_space_transform, RID_TYPE, Transform);
	FUNC_1A_S(combatant_set_space_transform, RID_TYPE);
	FUNC_2A_S(combatant_set_space_transform, RID_TYPE, Transform);
	FUNC_1A_R(uint32_t, combatant_get_status, RID_TYPE);
	FUNC_2A_S(combatant_set_space_transform, RID_TYPE, Transform);
	FUNC_1A_R(uint32_t, combatant_get_status, RID_TYPE);
};

#endif