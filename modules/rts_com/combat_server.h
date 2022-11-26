#ifndef COMBAT_SERVER_H
#define COMBAT_SERVER_H

#include <string>

#include "core/object.h"
#include "core/rid.h"
#include "core/vector.h"
#include "rcs_maincomp.h"

class Sentrience : public Object {
	GDCLASS(Sentrience, Object);

	bool active;

	VECTOR<RCSSimulation*> active_simulations;
	VECTOR<RCSRecording*> active_rec;
	// VECTOR<Ref<RCSUnilateralTeamsBind>> active_team_links;
	VECTOR<RID> all_rids;

	mutable RID_Owner<RCSRecording> recording_owner;
	mutable RID_Owner<RCSSimulation> simulation_owner;
	mutable RID_Owner<RCSCombatant> combatant_owner;
	mutable RID_Owner<RCSSquad> squad_owner;
	mutable RID_Owner<RCSTeam> team_owner;
	mutable RID_Owner<RCSRadar> radar_owner;
protected:
	static void _bind_methods();

	static void log(const String& msg);
	static void log(const std::string& msg);
	static void log(const char *msg);
	static Sentrience* singleton;

	String rid_sort(const RID& target);
public:
	Sentrience();
	~Sentrience();

	// _FORCE_INLINE_ RID_Owner<RCSCombatant> *get_combatant_owner() { return &combatant_owner; }

	static Sentrience* get_singleton() { return singleton; }
	virtual void poll(const float& delta);

	/* Core */
	_FORCE_INLINE_ void set_active(const bool& is_active) { active = is_active;}
	_FORCE_INLINE_ bool get_state() const { return active; }
	virtual void free_rid(const RID& target);
	void free_all_instances();
	void flush_instances_pool();

	/* Recording API */
	virtual RID recording_create();
	virtual bool recording_assert(const RID& r_rec);
	virtual bool recording_start(const RID& r_rec);
	virtual bool recording_end(const RID& r_rec);
	virtual bool recording_running(const RID& r_rec);
	virtual void recording_purge(const RID& r_rec);

	/* Simulation API */
	virtual RID simulation_create();
	virtual bool simulation_assert(const RID& r_simul);
	virtual void simulation_set_active(const RID& r_simul, const bool& p_active);
	virtual bool simulation_is_active(const RID& r_simul);
	virtual void simulation_bind_recording(const RID& r_simul, const RID& r_rec);
	virtual void simulation_unbind_recording(const RID& r_simul);
	virtual uint32_t simulation_count_combatant(const RID& r_simul);
	virtual uint32_t simulation_count_squad(const RID& r_simul);
	virtual uint32_t simulation_count_team(const RID& r_simul);
	virtual uint32_t simulation_count_radar(const RID& r_simul);
	virtual uint32_t simulation_count_all_instances(const RID& r_simul);

	/* Combatant API */
	virtual RID combatant_create();
	virtual bool combatant_assert(const RID& r_com);
	virtual void combatant_set_simulation(const RID& r_com, const RID& r_simul);
	virtual RID combatant_get_simulation(const RID& r_com);
	virtual bool combatant_is_squad(const RID& r_com, const RID& r_squad);
	virtual bool combatant_is_team(const RID& r_com, const RID& r_team);
	virtual void combatant_set_local_transform(const RID& r_com, const Transform& trans);
	virtual Transform combatant_get_space_transform(const RID& r_com);
	virtual Transform combatant_get_local_transform(const RID& r_com);
	virtual Transform combatant_get_combined_transform(const RID& r_com);
	virtual void combatant_set_space_transform(const RID& r_com, const Transform& trans);
	virtual void combatant_set_stand(const RID& r_com, const uint32_t& stand);
	virtual uint32_t combatant_get_stand(const RID& r_com);
	virtual uint32_t combatant_get_status(const RID& r_com);
	virtual void combatant_set_iid(const RID& r_com, const uint64_t& iid);
	virtual uint64_t combatant_get_iid(const RID& r_com);
	virtual void combatant_set_detection_meter(const RID& r_com, const double& dmeter);
	virtual double combatant_get_detection_meter(const RID& r_com);
	virtual bool combatant_engagable(const RID& from, const RID& to);
	virtual void combatant_bind_chip(const RID& r_com, const Ref<RCSChip>& chip, const bool& auto_unbind);
	virtual void combatant_unbind_chip(const RID& r_com);
	virtual void combatant_set_profile(const RID& r_com, const Ref<RCSCombatantProfile>& profile);
	virtual Ref<RCSCombatantProfile> combatant_get_profile(const RID& r_com);

	/* Squad API */
	virtual RID squad_create();
	virtual bool squad_assert(const RID& r_squad);
	virtual void squad_set_simulation(const RID& r_squad, const RID& r_simul);
	virtual RID squad_get_simulation(const RID& r_squad);
	virtual bool squad_is_team(const RID& r_squad, const RID& r_team);
	virtual void squad_add_combatant(const RID& r_squad, const RID& r_com);
	virtual void squad_remove_combatant(const RID& r_squad, const RID& r_com);
	virtual bool squad_has_combatant(const RID& r_squad, const RID& r_com);
	virtual bool squad_engagable(const RID& from, const RID& to);
	virtual uint32_t squad_count_combatant(const RID& r_squad);
	virtual void squad_bind_chip(const RID& r_com, const Ref<RCSChip>& chip, const bool& auto_unbind);
	virtual void squad_unbind_chip(const RID& r_com);

	/* Team API */
	virtual RID team_create();
	virtual bool team_assert(const RID& r_team);
	virtual void team_set_simulation(const RID& r_team, const RID& r_simul);
	virtual RID team_get_simulation(const RID& r_team);
	virtual void team_add_squad(const RID& r_team, const RID& r_squad);
	virtual void team_remove_squad(const RID& r_team, const RID& r_squad);
	virtual bool team_has_squad(const RID& r_team, const RID& r_squad);
	virtual bool team_engagable(const RID& from, const RID& to);
	virtual Ref<RCSUnilateralTeamsBind> team_create_link(const RID& from, const RID& to);
	virtual void team_create_link_bilateral(const RID& from, const RID& to);
	virtual Ref<RCSUnilateralTeamsBind> team_get_link(const RID& from, const RID& to);
	virtual bool team_has_link(const RID& from, const RID& to);
	virtual bool team_unlink(const RID& from, const RID& to);
	virtual bool team_unlink_bilateral(const RID& from, const RID& to);
	virtual void team_purge_links_multilateral(const RID& from);
	virtual uint32_t team_count_squad(const RID& r_team);
	virtual void team_bind_chip(const RID& r_team, const Ref<RCSChip>& chip, const bool& auto_unbind);
	virtual void team_unbind_chip(const RID& r_team);

	/* Radar API */
	virtual RID radar_create();
	virtual bool radar_assert(const RID& r_rad);
	virtual void radar_set_simulation(const RID& r_rad, const RID& r_simul);
	virtual RID radar_get_simulation(const RID& r_rad);
	virtual void radar_set_profile(const RID& r_rad, const Ref<RCSRadarProfile>& profile);
	virtual Ref<RCSRadarProfile> radar_get_profile(const RID& r_rad);
	virtual void radar_request_recheck_on(const RID& r_rad, const RID& r_com);
	virtual Array radar_get_detected(const RID& r_rad);
	virtual Array radar_get_locked(const RID& r_rad);
};

#endif
