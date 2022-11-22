#ifndef RCS_MAINCOMP_H
#define RCS_MAINCOMP_H

#include "modules/advanced_curve/advanced_curve.h"
#include "core/string_name.h"
#include "core/math/transform.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/reference.h"
#include "rtscom_rid.h"
#include "rtscom_meta.h"
// #include "rcs_simulation.h"

#define PI 3.1415926535897932384626433833
#define TAU 6.2831853071795864769252867666

#define MIN_RADAR_SPREAD 0.008726646
#define MAX_RADAR_SPREAD PI

#define MIN_RADAR_FREQ 0.001
#define MAX_RADAR_FREQ 5.0

class RCSCombatantProfile;
class RCSCombatant;
class RCSSquad;
class RCSTeam;
class RCSRecording;
class RCSSimulation;
class RCSRadarProfile;
class RCSRadar;

class RCSRecording : public RID_RCS {
private:
	Vector<RCSSimulation*> simulations;
	Vector<RID> rids;
protected:
	void recieve_event(RCSSimulation* handler, RID_RCS* from, void* event = nullptr);
public:
	RCSRecording();
	~RCSRecording();

	friend class RCSSimulation;

	void add_simulation(RCSSimulation* simul);
	void remove_simulation(RCSSimulation* simul);
	bool has_simulation(RCSSimulation* simul) const;
	Vector<RID> get_rids() const;

	void poll(const float& delta) override;

};

struct RadarRecheckTicket{
	RCSRadar* request_sender = nullptr;
	RCSCombatant* bogey = nullptr;
	RadarRecheckTicket(RCSRadar* sender, RCSCombatant* reciever) { request_sender = sender; bogey = reciever; }
};

class RCSSimulation : public RID_RCS{
private:
	Vector<RCSCombatant*> combatants;
	Vector<RCSSquad*> squads;
	Vector<RCSTeam*> teams;
	Vector<RCSRadar*> radars;
	Vector<RadarRecheckTicket*> rrecheck_requests;

	RCSRecording* recorder;
public:
	RCSSimulation();
	~RCSSimulation();

	void add_combatant(RCSCombatant* com);
	void add_squad(RCSSquad* squad);
	void add_team(RCSTeam* team);
	void add_radar(RCSRadar* rad);

	void remove_combatant(RCSCombatant* com);
	void remove_squad(RCSSquad* squad);
	void remove_team(RCSTeam* team);
	void remove_radar(RCSRadar* rad);

	Vector<RCSCombatant*> *get_combatants() { return &combatants; }
	Vector<RCSSquad*> *get_squads() { return &squads; }
	Vector<RCSTeam*> *get_teams() { return &teams; }
	Vector<RCSRadar*> *get_radars() { return &radars; }

	void poll(const float& delta) override;
	void set_recorder(RCSRecording* rec);
	void radar_request_recheck(RadarRecheckTicket* ticket);
};

class RCSCombatantProfile : public Resource{
	GDCLASS(RCSCombatantProfile, Resource);
private:
	double detection_threshold = 1000.0;
	double detection_subvention = 0.5;
	StringName combatant_name;
	// uint32_t combatant_id = 0;
	uint32_t stand = 0;
protected:
	static void _bind_methods();
public:
	RCSCombatantProfile();
	~RCSCombatantProfile() = default;

	// _FORCE_INLINE_ void set_combatant_id(const uint32_t& new_id) { combatant_id = new_id; emit_changed(); }
	// _FORCE_INLINE_ uint32_t get_id() const { return combatant_id; }

	_FORCE_INLINE_ void set_stand(const uint32_t& new_stand) { stand = new_stand; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_stand() const { return stand; }

	_FORCE_INLINE_ void set_pname(const StringName& name) { combatant_name = name; emit_changed(); }
	_FORCE_INLINE_ StringName get_pname() const { return combatant_name; }

	_FORCE_INLINE_ void set_detection_threshold(const double& new_threshold) { detection_threshold = new_threshold; emit_changed(); }
	_FORCE_INLINE_ double get_detection_threshold() const { return detection_threshold; }

	_FORCE_INLINE_ void set_subvention(const double& new_sub) { detection_subvention = CLAMP(new_sub, 0.001, 1.0); emit_changed(); }
	_FORCE_INLINE_ double get_subvention() const { return detection_subvention; }
};

class RCSCombatant : public RID_RCS {
private:
	// RID squad_ref;
	RCSSimulation* simulation;
	RCSSquad* my_squad;
	Ref<RCSCombatantProfile> combatant_profile;
private:
	Transform space_transform;
	Transform local_transform;
	double detection_meter = 0.0;
	uint32_t combatant_id = 0;

	uint32_t status;
public:
	RCSCombatant();
	~RCSCombatant() override;

	friend class Sentrience;

	void set_squad(RCSSquad* new_squad);
	RCSSquad* get_squad() { return my_squad; }

	bool is_same_team(RCSCombatant* com);
	bool is_hostile(RCSCombatant* com);
	bool is_ally(RCSCombatant* com);

	void poll(const float& delta) override;

	_FORCE_INLINE_ Transform get_combined_transform() const { return space_transform * local_transform;  }

	void set_profile(const Ref<RCSCombatantProfile>& prof) { combatant_profile = prof; }
	Ref<RCSCombatantProfile> get_profile() const { return combatant_profile; }

	virtual Ref<RawRecord> serialize() const;
	virtual bool serialize(const Ref<RawRecord>& from);

	virtual void set_simulation(RCSSimulation* sim);

	_FORCE_INLINE_ void set_combatant_id(const uint32_t& new_id) { combatant_id = new_id; }
	_FORCE_INLINE_ uint32_t get_combatant_id() const { return combatant_id; }

	_FORCE_INLINE_ void _set_detection_meter(const double& dm) { detection_meter = dm; }
	_FORCE_INLINE_ double _get_detection_meter() const { return detection_meter; }

	virtual void set_st(const Transform& trans) { space_transform = trans;}
	virtual Transform get_st() const { return space_transform; }

	virtual void set_lt(const Transform& trans) { local_transform = trans;}
	virtual Transform get_lt() const { return local_transform; }

	virtual void set_cname(const StringName& cname);
	virtual StringName get_cname() const;

	virtual void set_stand(const uint32_t& new_stand);
	virtual uint32_t get_stand() const;

	virtual void set_status(const uint32_t& new_status) { status = new_status;  }
	virtual uint32_t get_status() const { return status; }

};

class RCSSquad : public RID_RCS {
private:
	StringName squad_name;
	uint32_t squad_id = 0;
	uint32_t combatant_id_allocator = 0;

	RCSSimulation* simulation;
	RCSTeam *my_team;

	Vector<RCSCombatant*> combatants;
public:
	RCSSquad();
	~RCSSquad();

	friend class Sentrience;

	void set_team(RCSTeam* new_team);
	RCSTeam* get_team() { return my_team; }

	_FORCE_INLINE_ void set_squad_name(const StringName& new_name) { squad_name = new_name; }
	_FORCE_INLINE_ StringName get_squad_name() const { return squad_name; }

	_FORCE_INLINE_ void set_squad_id(const uint32_t& new_id) { squad_id = new_id;}
	_FORCE_INLINE_ uint32_t get_squad_id() const { return squad_id; }

	virtual void set_simulation(RCSSimulation* sim);
	_FORCE_INLINE_ bool has_combatant(RCSCombatant* com) const { return (combatants.find(com) != -1); }
	_FORCE_INLINE_ void add_combatant(RCSCombatant* com) { combatants.push_back(com); combatant_id_allocator += 1; com->set_combatant_id(combatant_id_allocator); }
	_FORCE_INLINE_ void remove_combatant(RCSCombatant* com) { return combatants.erase(com);  }
};

class RCSTeam : public RID_RCS {
private:
	StringName team_name;
	uint32_t team_id = 0;
	uint32_t squad_id_allocator = 0;

	RCSSimulation* simulation;

	Vector<RCSSquad*> squads;
public:
	RCSTeam();
	~RCSTeam();

	friend class Sentrience;

	_FORCE_INLINE_ void set_team_name(const StringName& new_name) { team_name = new_name; }
	_FORCE_INLINE_ StringName get_team_name() const { return team_name; }

	_FORCE_INLINE_ void set_team_id(const uint32_t& new_id) { team_id = new_id;}
	_FORCE_INLINE_ uint32_t get_team_id() const { return team_id; }

	virtual void set_simulation(RCSSimulation* sim);
	_FORCE_INLINE_ bool has_squad(RCSSquad* squad) const { return (squads.find(squad) != -1); }
	_FORCE_INLINE_ void add_squad(RCSSquad* squad) { squads.push_back(squad); squad_id_allocator += 1; squad->set_squad_id(squad_id_allocator);  }
	_FORCE_INLINE_ void remove_squad(RCSSquad* squad) { return squads.erase(squad);  }
};

struct RadarPingRequest{
public:
	RCSCombatant* from;
	RCSCombatant* to;
	Transform self_transform;
	bool detect_result = false;
	bool lock_result = false;
	bool late_recheck = false;

	RadarPingRequest(RCSCombatant* from, RCSCombatant* to){ this->from = from; this->to = to;}
};

class RCSRadarProfile : public Resource{
	GDCLASS(RCSRadarProfile, Resource);
public:
	enum RadarScanMode : unsigned int {
		ScanModeSwarm,
		ScanModeSingle,
	};
private:
	Ref<AdvancedCurve> detection_curve;
	Ref<AdvancedCurve> acquisition_curve;
	StringName rp_name;
	double spread = Math::deg2rad(30.0);
	double frequency = 1.0;
	double scan_contribution = 0.3;
	RadarScanMode scan_method = RadarScanMode::ScanModeSingle;
protected:
	static void _bind_methods();
public:
	RCSRadarProfile();
	~RCSRadarProfile() = default;

	_FORCE_INLINE_ void set_dcurve(const Ref<AdvancedCurve>& curve) { detection_curve = curve; emit_changed(); }
	_FORCE_INLINE_ Ref<AdvancedCurve> get_dcurve() const { return detection_curve; }

	_FORCE_INLINE_ void set_acurve(const Ref<AdvancedCurve>& curve) { acquisition_curve = curve; emit_changed(); }
	_FORCE_INLINE_ Ref<AdvancedCurve> get_acurve() const { return acquisition_curve; }

	_FORCE_INLINE_ void set_pname(const StringName& name) { rp_name = name; emit_changed(); }
	_FORCE_INLINE_ StringName get_pname() const { return rp_name; }

	_FORCE_INLINE_ void set_spread(const double& new_spread) { spread = CLAMP(new_spread, MIN_RADAR_SPREAD, MAX_RADAR_SPREAD); emit_changed(); }
	_FORCE_INLINE_ double get_spread() const { return spread; }

	_FORCE_INLINE_ void set_freq(const double& new_freq) { frequency = CLAMP(new_freq, MIN_RADAR_FREQ, MAX_RADAR_FREQ); emit_changed(); }
	_FORCE_INLINE_ double get_freq() const { return frequency; }

	_FORCE_INLINE_ void set_method(RadarScanMode new_met) { scan_method = new_met; emit_changed(); }
	_FORCE_INLINE_ RadarScanMode get_method() const { return scan_method; }

	_FORCE_INLINE_ void set_contribution(const double& new_con) { scan_contribution = CLAMP(new_con, 0.0001, 0.999); emit_changed(); }
	_FORCE_INLINE_ double get_contribution() const { return scan_contribution; }

	void ping_target(RadarPingRequest* ping_request);
	void swarm_detect(RadarPingRequest* ticket);
	virtual void internal_detect(RadarPingRequest* ping_request);
	virtual void internal_acquire(RadarPingRequest* ping_request);
	virtual void internal_swarm_detect(RadarPingRequest* ticket);
};
class RCSRadar: public RID_RCS{
private:
	RCSSimulation* simulation;
	RCSCombatant* assigned_vessel;
	Ref<RCSRadarProfile> rprofile;

	double timer = 0.0;
	// Transform fallback_transform;
	Vector<RCSCombatant*> detected_combatants;
	Vector<RCSCombatant*> locked_combatants;
	Array detected_rids;
	Array locked_rids;
	bool is_init = false;

	// bool detect(const float& delta, RCSCombatant* combatant) const;
public:
	RCSRadar();
	~RCSRadar();

	_FORCE_INLINE_ void set_profile(const Ref<RCSRadarProfile>& prof) { rprofile = prof; timer = 0.0; }
	_FORCE_INLINE_ Ref<RCSRadarProfile> get_profile() const { return rprofile; }

	// _FORCE_INLINE_ void set_ft(const Transform& trans) { fallback_transform = trans; }
	// _FORCE_INLINE_ Transform get_ft() const { return fallback_transform; }

	friend class Sentrience;

	virtual void set_simulation(RCSSimulation* sim);
	_FORCE_INLINE_ Array get_detected() const { return detected_rids; }
	_FORCE_INLINE_ Array get_locked() const { return locked_rids; }

	void poll(const float& delta) override;
	virtual void late_check(const float& delta, RadarRecheckTicket* recheck);
};

#endif
