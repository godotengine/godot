#ifndef RCS_MAINCOMP_H
#define RCS_MAINCOMP_H



#include <string>

#include "modules/advanced_curve/advanced_curve.h"
#include "scene/resources/world.h"
#include "core/print_string.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "core/string_name.h"
#include "core/math/transform.h"
#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
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

#define MIN_RPD 0.1
#define MAX_RPD 5.0

class RCSCombatantProfile;
class RCSCombatant;
class RCSSquad;
class RCSUnilateralTeamsBind;
class RCSTeam;
class RCSRecording;
class RCSSimulation;
class RCSRadarProfile;
class RCSRadar;
class RCSEngagement;
class RCSEngagementInternal;

class EventReport;
class CombatantEventReport;
class SquadEventReport;
class TeamEventReport;
class RadarEventReport;

#define REPORTER_CLASS(base, inherited)                                        \
	GDCLASS(base, inherited);                                                  \
	friend class RCSSimulation;                                                \
	friend class RCSRecording;                                                 \
protected:                                                                     \
	RCSSimulation *simulation = nullptr;                                       \
	static _FORCE_INLINE_ void _bind_methods() {}                              \
	_FORCE_INLINE_ Dictionary (base::*__get_primitive_describer())() {         \
		return &base::primitive_describe;                                      \
	}                                                                          \
public:                                                                        \
	_FORCE_INLINE_ RCSSimulation* get_simulation() { return simulation; }              \
private:


//--------------------------------------------------------------------------


struct RCSMemoryAllocation{
	uint64_t allocated = 0;
	uint64_t deallocated = 0;
	static RCSMemoryAllocation *tracker_ptr;
	RCSMemoryAllocation();
	~RCSMemoryAllocation();
	uint64_t currently_allocated() const {
		if (allocated < deallocated) return 0;
		return allocated - deallocated;
	}
};

class EventReport : public Reference {
protected:
	_FORCE_INLINE_ Dictionary primitive_describe() { return Dictionary(); }
// -------------------------------------------
	REPORTER_CLASS(EventReport, Reference);
};
class CombatantEventReport : public EventReport {
	REPORTER_CLASS(CombatantEventReport, EventReport);
public:
	CombatantEventReport();
};
class SquadEventReport : public EventReport {
	REPORTER_CLASS(SquadEventReport, EventReport);
public:
	SquadEventReport();
};
class TeamEventReport : public EventReport {
	REPORTER_CLASS(TeamEventReport, EventReport);
public:
	TeamEventReport();
};
class RadarEventReport : public EventReport {
	REPORTER_CLASS(RadarEventReport, EventReport);
public:
	enum RDR_EventType {
		NA = 0,
		BogeyDetected = 1,
		BogeyLocked = 2,
	};

private:
	uint32_t event_type;

	friend class RCSSimulation;
	friend class RCSRadar;
public:
	RadarEventReport();
};

//--------------------------------------------------------------------------

struct EventReportTicket {
	uint64_t timestamp = 0;
	Ref<EventReport> event;
	EventReportTicket(const uint64_t& timestamp, const Ref<EventReport>& event) { this->timestamp = timestamp; this->event = event; }
};

class RCSRecording : public RID_RCS {
private:
	VECTOR<EventReportTicket*> reports_holder;
protected:
	void push_event(const Ref<EventReport>& event);
	bool running = false;
public:
	RCSRecording();
	~RCSRecording();

	friend class RCSSimulation;
	friend class Sentrience;

	void poll(const float& delta) override;
	void purge();
	VECTOR<EventReportTicket*> *events_by_simulation(RCSSimulation* simulation);
};

#ifdef USE_SAFE_RID_COUNT
struct RadarRecheckTicket{
	std::weak_ptr<RCSRadar> request_sender;
	std::weak_ptr<RCSCombatant> bogey;
	RadarRecheckTicket(const std::weak_ptr<RCSRadar> &sender, const std::weak_ptr<RCSCombatant> &reciever) {
		request_sender = sender;
		bogey = reciever;
	}
};
#else
struct RadarRecheckTicket {
	RCSRadar *request_sender;
	RCSCombatant *bogey;
	RadarRecheckTicket(RCSRadar *sender, RCSCombatant *reciever) {
		request_sender = sender;
		bogey = reciever;
	}
};
#endif

/* Opaque class */
class RCSEngagementInternal : public RID_RCS {
private:
	bool engaging;
	uint32_t scale;
private:
	VECTOR<RID_TYPE> participating;
	RID_TYPE offending_team;
	VECTOR<RID_TYPE> deffending_teams;
	VECTOR<RID_TYPE> offending_squads;
	VECTOR<RID_TYPE> deffending_squads;

	friend class RCSSimulation;
	friend class RCSTeam;
	friend class RCSSquad;
private:
	VECTOR<RCSEngagement*> referencing;

	// void cut_ties_team(RCSTeam* team);
	// void cut_ties_squad(RCSSquad* squad);
	// void cut_ties_to_all();

	void erase_reference(RCSEngagement* to);
	void flush_all_references();
	friend class RCSEngagement;
public:
	RCSEngagementInternal();
	~RCSEngagementInternal();
// ---------------------------------------------
	// bool __is_engagement_happening() const;
	// bool __is_engagement_over() const;
	// uint32_t __get_scale() const;
	// VECTOR<RCSTeam*>    *__get_involving_teams() const { return &participating; }
	// VECTOR<RCSSquad*>   *__get_involving_squads() const  { ERR_FAIL_V_MSG(new VECTOR<RCSSquad*>(), "This method is yet to be implemented"); }
	// RCSTeam             *__get_offending_team() const { return offending_team; }
	// VECTOR<RCSTeam*>    *__get_deffending_teams() const { return &deffending_teams; }
	// VECTOR<RCSSquad*>   *__get_offending_squads() const { return  &offending_squads; }
	// VECTOR<RCSSquad*>   *__get_deffending_squads() const { return &deffending_squads; }
// ---------------------------------------------
	bool is_engagement_happening() const;
	bool is_engagement_over() const;
	uint32_t get_scale() const;
	Array get_involving_teams() const;
	Array get_involving_squads() const;
	RID_TYPE get_offending_team() const;
	Array get_deffending_teams() const;
	Array get_offending_squads() const;
	Array get_deffending_squads() const;
// ---------------------------------------------

	Ref<RCSEngagement> spawn_reference();
};

class RCSEngagement : public Reference {
	GDCLASS(RCSEngagement, Reference);
private:
	RCSEngagementInternal *logger;
public:
	enum EngagementScale : unsigned int {
		// No Engagement
		NA,
		Stalk,
		Standoff,
		// Small scale
		Skirmish,
		Ambush,
		// Full scale
		Assault,
		FullEngagement,
		// Siege related
		Landing,
		Encirclement,
		Siege,
	};
protected:
	static void _bind_methods();

	friend class RCSEngagementInternal;
public:
	RCSEngagement();
	~RCSEngagement();

	_FORCE_INLINE_ bool is_engagement_happening() const		{ ERR_FAIL_COND_V(!logger, false)  ; return logger->is_engagement_happening(); }
	_FORCE_INLINE_ bool is_engagement_over() const			{ ERR_FAIL_COND_V(!logger, false)  ; return logger->is_engagement_over(); }
	_FORCE_INLINE_ EngagementScale get_scale() const		{ ERR_FAIL_COND_V(!logger, NA)     ; return (EngagementScale)logger->get_scale(); }
	_FORCE_INLINE_ Array get_involving_teams() const		{ ERR_FAIL_COND_V(!logger, Array()); return logger->get_involving_teams(); }
	_FORCE_INLINE_ Array get_involving_squads() const		{ ERR_FAIL_COND_V(!logger, Array()); return logger->get_involving_squads(); }
	_FORCE_INLINE_ RID_TYPE get_offending_team() const		{ ERR_FAIL_COND_V(!logger, RID_TYPE())  ; return logger->get_offending_team(); }
	_FORCE_INLINE_ Array get_deffending_teams() const		{ ERR_FAIL_COND_V(!logger, Array()); return logger->get_deffending_teams(); }
	_FORCE_INLINE_ Array get_offending_squads() const		{ ERR_FAIL_COND_V(!logger, Array()); return logger->get_offending_squads(); }
	_FORCE_INLINE_ Array get_deffending_squads() const		{ ERR_FAIL_COND_V(!logger, Array()); return logger->get_deffending_squads(); }
};

class RCSSimulation : public RID_RCS{
private:
#ifdef USE_SAFE_RID_COUNT
	WrappedVector<std::weak_ptr<RCSCombatant>> combatants;
	WrappedVector<std::weak_ptr<RCSSquad>> squads;
	WrappedVector<std::weak_ptr<RCSTeam>> teams;
	WrappedVector<std::weak_ptr<RCSRadar>> radars;
	WrappedVector<std::shared_ptr<RadarRecheckTicket>> rrecheck_requests;
	std::weak_ptr<RCSRecording> recorder;
#else
	VECTOR<RCSCombatant *> combatants;
	VECTOR<RCSSquad *> squads;
	VECTOR<RCSTeam *> teams;
	VECTOR<RCSRadar *> radars;
	VECTOR<RadarRecheckTicket *> rrecheck_requests;
	RCSRecording* recorder;
#endif
public:
	RCSSimulation();
	~RCSSimulation();
#ifdef USE_SAFE_RID_COUNT
	void add_combatant(const std::weak_ptr<RCSCombatant>& com);
	void add_squad(const std::weak_ptr<RCSSquad> &squad);
	void add_team(const std::weak_ptr<RCSTeam> &team);
	void add_radar(const std::weak_ptr<RCSRadar> &rad);

	void remove_combatant(const std::weak_ptr<RCSCombatant> &com);
	void remove_squad(const std::weak_ptr<RCSSquad> &squad);
	void remove_team(const std::weak_ptr<RCSTeam> &team);
	void remove_radar(const std::weak_ptr<RCSRadar> &rad);

	VECTOR<std::weak_ptr<RCSCombatant>> get_combatants() const { return combatants; }
	VECTOR<std::weak_ptr<RCSSquad>> get_squads() const { return squads; }
	VECTOR<std::weak_ptr<RCSTeam>> get_teams() const { return teams; }
	VECTOR<std::weak_ptr<RCSRadar>> get_radars() const { return radars; }

	void poll(const float &delta) override;

	void set_recorder(RCSRecording *rec);
	_FORCE_INLINE_ bool has_recorder() const { return !recorder.expired(); }
	_FORCE_INLINE_ bool is_recording() const { return (has_recorder() ? (recorder.lock()->running) : false); }
	void combatant_event(Ref<CombatantEventReport> event);
	void squad_event(Ref<SquadEventReport> event);
	void team_event(Ref<TeamEventReport> event);
	void radar_event(Ref<RadarEventReport> event);

	std::weak_ptr<RCSCombatant> get_combatant_from_iid(const uint64_t &iid) const;
	void radar_request_recheck(const std::shared_ptr<RadarRecheckTicket> &ticket);
#else
	void add_combatant(RCSCombatant *com);
	void add_squad(RCSSquad *squad);
	void add_team(RCSTeam *team);
	void add_radar(RCSRadar *rad);

	void remove_combatant(RCSCombatant *com);
	void remove_squad(RCSSquad *squad);
	void remove_team(RCSTeam *team);
	void remove_radar(RCSRadar *rad);

	VECTOR<RCSCombatant *> *get_combatants() { return &combatants; }
	VECTOR<RCSSquad *> *get_squads() { return &squads; }
	VECTOR<RCSTeam *> *get_teams() { return &teams; }
	VECTOR<RCSRadar *> *get_radars() { return &radars; }

	void poll(const float &delta) override;

	void set_recorder(RCSRecording *rec);
	_FORCE_INLINE_ bool has_recorder() const { return recorder != nullptr; }
	_FORCE_INLINE_ bool is_recording() const { return (has_recorder() ? (recorder->running) : false); }
	void combatant_event(Ref<CombatantEventReport> event);
	void squad_event(Ref<SquadEventReport> event);
	void team_event(Ref<TeamEventReport> event);
	void radar_event(Ref<RadarEventReport> event);

	RCSCombatant *get_combatant_from_iid(const uint64_t &iid) const;
	void radar_request_recheck(RadarRecheckTicket* ticket);
#endif
};

class RCSCombatantProfile : public Resource{
	GDCLASS(RCSCombatantProfile, Resource);
public:
	enum CombatantAttribute : unsigned int {
		NormalCombatant = 0,
		Projectile = 1,
		Guided = 2,
		Undetectable = 4,
	};
	enum CombatantStand : unsigned int {
		Movable = 0,
		Static = 1,
		Passive = 2,
		Deffensive = 4,
		Retaliative = 8,
		Aggressive = 16,
	};
private:
	double detection_threshold = 0.7;
	double acquisition_threshold = 0.5;
	double detection_subvention = 0.5;
	StringName combatant_name;
	// uint32_t combatant_id = 0;
	uint32_t attributes = NormalCombatant;
	uint32_t stand = Movable + Deffensive;
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

	_FORCE_INLINE_ void set_detection_threshold(const double& new_threshold) { detection_threshold = CLAMP(new_threshold, 0.001, 0.98); emit_changed(); }
	_FORCE_INLINE_ double get_detection_threshold() const { return detection_threshold; }

	_FORCE_INLINE_ void set_combatant_attributes(const uint32_t& attr) { attributes = attr; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_combatant_attributes() const { return attributes; }

	_FORCE_INLINE_ void set_acquisition_threshold(const double& new_threshold) { acquisition_threshold = CLAMP(new_threshold, 0.001, 0.98); emit_changed(); }
	_FORCE_INLINE_ double get_acquisition_threshold() const { return acquisition_threshold; }

	_FORCE_INLINE_ void set_subvention(const double& new_sub) { detection_subvention = CLAMP(new_sub, 0.001, 1.0); emit_changed(); }
	_FORCE_INLINE_ double get_subvention() const { return detection_subvention; }
};

#define COMBATANT_DETECTION_LIMIT 10.0

class RCSCombatant : public RID_RCS {
private:
#ifdef USE_SAFE_RID_COUNT
	// RID_TYPE squad_ref;
	std::weak_ptr<RCSSimulation> simulation;
	std::weak_ptr<RCSSquad> my_squad;
#else
	RCSSimulation *simulation;
	RCSSquad *my_squad;
#endif
	Ref<RCSCombatantProfile> combatant_profile;
private:
	Transform space_transform;
	Transform local_transform;
	double detection_meter = 0.0;
	uint32_t combatant_id = 0;
	uint64_t identifier_id = 0;

	uint32_t status;
public:
	RCSCombatant();
	~RCSCombatant() override;

	friend class Sentrience;
	friend class RCSSimulation;

	void poll(const float& delta) override;
	_FORCE_INLINE_ Transform get_combined_transform() const { return space_transform * local_transform;  }

	void set_profile(const Ref<RCSCombatantProfile>& prof) { combatant_profile = prof; }
	Ref<RCSCombatantProfile> get_profile() const { return combatant_profile; }

	virtual Ref<RawRecord> serialize() const;
	virtual bool serialize(const Ref<RawRecord>& from);

	_FORCE_INLINE_ void set_combatant_id(const uint32_t& new_id) { combatant_id = new_id; }
	_FORCE_INLINE_ uint32_t get_combatant_id() const { return combatant_id; }

	_FORCE_INLINE_ void set_iid(const uint64_t& new_id) { identifier_id = new_id; }
	_FORCE_INLINE_ uint64_t get_iid() const { return identifier_id; }

	_FORCE_INLINE_ void _set_detection_meter(const double& dm) { detection_meter = CLAMP(dm, 0.0, COMBATANT_DETECTION_LIMIT); }
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
#ifdef USE_SAFE_RID_COUNT
	_FORCE_INLINE_ std::weak_ptr<RCSCombatant> self_cast_weakptr() const {
		return *((std::weak_ptr<RCSCombatant> *)&self_ref);
	}

	void set_squad(const std::weak_ptr<RCSSquad>& new_squad);
	std::weak_ptr<RCSSquad> get_squad() const { return my_squad; }

	bool is_same_team(const std::weak_ptr<RCSCombatant> &com);
	bool is_hostile(const std::weak_ptr<RCSCombatant> &com);
	bool is_ally(const std::weak_ptr<RCSCombatant> &com);
	bool is_engagable(const std::weak_ptr<RCSCombatant> &bogey);

	virtual void set_simulation(const std::weak_ptr<RCSSimulation>& sim);
#else
	void set_squad(RCSSquad *new_squad);
	RCSSquad *get_squad() { return my_squad; }

	bool is_same_team(RCSCombatant *com);
	bool is_hostile(RCSCombatant *com);
	bool is_ally(RCSCombatant *com);
	bool is_engagable(RCSCombatant *bogey);

	virtual void set_simulation(RCSSimulation *sim);
#endif
};

class RCSSquad : public RID_RCS {
private:
	StringName squad_name;
	uint32_t squad_id = 0;
	SafeRefCount refcount;

private:
#ifdef USE_SAFE_RID_COUNT
	std::weak_ptr<RCSSimulation> simulation;
	std::weak_ptr<RCSTeam> my_team;
	WrappedVector<std::weak_ptr<RCSCombatant>> combatants;
#else
	RCSSimulation* simulation;
	RCSTeam *my_team;

	VECTOR<RCSCombatant*> combatants;

	// VECTOR<RCSEngagementInternal*> engagement_loggers;
	
	// void add_engagement(RCSEngagementInternal* engagement) { engagement_loggers.push_back(engagement); }
	// void remove_engagement(RCSEngagementInternal* engagement) VEC_ERASE(engagement_loggers, engagement)
#endif
	friend class RCSEngagementInternal;
	friend class RCSSimulation;
	friend class Sentrience;
public:
	RCSSquad();
	~RCSSquad();

	_FORCE_INLINE_ void set_squad_name(const StringName& new_name) { squad_name = new_name; }
	_FORCE_INLINE_ StringName get_squad_name() const { return squad_name; }
	_FORCE_INLINE_ void set_squad_id(const uint32_t& new_id) { squad_id = new_id;}
	_FORCE_INLINE_ uint32_t get_squad_id() const { return squad_id; }

#ifdef USE_SAFE_RID_COUNT
	_FORCE_INLINE_ std::weak_ptr<RCSSquad> self_cast_weakptr() const {
		return *((std::weak_ptr<RCSSquad> *)&self_ref);
	}
	_FORCE_INLINE_ WrappedVector<std::weak_ptr<RCSCombatant>> get_combatants() { return combatants; }

	void set_team(const std::weak_ptr<RCSTeam>& new_team);
	std::weak_ptr<RCSTeam> get_team() const { return my_team; }

	void set_simulation(const std::weak_ptr<RCSSimulation>& sim);
	bool has_combatant(const std::weak_ptr<RCSCombatant> &com);
	_FORCE_INLINE_ void add_combatant(std::weak_ptr<RCSCombatant> com) {
		combatants.push_back(com);
		com.lock()->set_combatant_id(refcount.refval());
	}
	void remove_combatant(const std::weak_ptr<RCSCombatant>& com);
	bool is_same_team(const std::weak_ptr<RCSSquad> &bogey);
	bool is_engagable(const std::weak_ptr<RCSSquad> &bogey);
#else
	_FORCE_INLINE_ VECTOR<RCSCombatant *> *get_combatants() { return &combatants; }

	void set_team(RCSTeam *new_team);
	RCSTeam *get_team() { return my_team; }

	virtual void set_simulation(RCSSimulation *sim);
	_FORCE_INLINE_ bool has_combatant(RCSCombatant *com) const VEC_HAS(combatants, com)
			_FORCE_INLINE_ void add_combatant(RCSCombatant *com) {
		combatants.push_back(com);
		com->set_combatant_id(refcount.refval());
	}
	_FORCE_INLINE_ void remove_combatant(RCSCombatant *com) VEC_ERASE(combatants, com)
	bool is_engagable(RCSSquad *bogey);
	bool is_same_team(RCSSquad *bogey) {
		if (!bogey)
			return false;
		return bogey->get_team() == my_team;
	}
#endif
};

// Unilateral Interteam Profile
class RCSUnilateralTeamsBind : public Resource {
	GDCLASS(RCSUnilateralTeamsBind, Resource);
public:
	enum TeamRelationship : unsigned int {
		TeamNeutral = 0,
		TeamAllies,
		TeamHostiles
	};
	enum InterTeamAttribute : unsigned int {
		ITA_None					= 0,
		// Hostile attributes
		ITA_Engagable				= 1,
		ITA_AutoEngage				= 2,
		ITA_DetectionWarning		= 4,
		// Ally attributes
		ITA_ShareRadar				= 32,
		ITA_ShareLocation			= 64,
	};
private:
	uint32_t relationship = TeamNeutral;
	uint32_t attributes = ITA_None;

#ifdef USE_SAFE_RID_COUNT
	std::weak_ptr<RCSTeam> toward;
#else
	RCSTeam *toward = nullptr;
#endif
protected:
	static void _bind_methods();
public:
	RCSUnilateralTeamsBind();
	~RCSUnilateralTeamsBind();

	friend class Sentrience;
	friend class RCSTeam;

	_FORCE_INLINE_ void set_relationship(const uint32_t& rel) { relationship = rel; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_relationship() const { return relationship; }

	_FORCE_INLINE_ void set_attributes(const uint32_t& attr) { attributes = attr; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_attributes() const { return attributes; }

	// _FORCE_INLINE_ void set_from_rid(const RID_TYPE& rid) { from = rid; emit_changed(); }
	// _FORCE_INLINE_ RID_TYPE get_from_rid() const { return (!from ? RID_TYPE() : from->get_self()); }

#ifdef USE_SAFE_RID_COUNT
	_FORCE_INLINE_ void set_to_rid(const std::weak_ptr<RCSTeam> & to) { toward = to; emit_changed(); }
#else
	_FORCE_INLINE_ void set_to_rid(RCSTeam *to) { toward = to; emit_changed(); }
#endif

	RID_TYPE get_to_rid() const;
};

class RCSTeam : public RID_RCS {
private:
	StringName team_name;
	uint32_t team_id = 0;
	SafeRefCount refcount;

private:
#ifdef USE_SAFE_RID_COUNT
	std::weak_ptr<RCSSimulation> simulation;
	WrappedVector<std::weak_ptr<RCSSquad>> squads;
	WrappedVector<Ref<RCSUnilateralTeamsBind>> team_binds;
	// VECTOR<RCSEngagementInternal*> engagement_loggers;

	_FORCE_INLINE_ void add_engagement(RCSEngagementInternal *engagement) { }
	_FORCE_INLINE_ void remove_engagement(RCSEngagementInternal *engagement) { }
#else
	RCSSimulation* simulation;
	VECTOR<RCSSquad*> squads;
	VECTOR<Ref<RCSUnilateralTeamsBind>> team_binds;
	VECTOR<RCSEngagementInternal*> engagement_loggers;

	_FORCE_INLINE_ void add_engagement(RCSEngagementInternal* engagement) { engagement_loggers.push_back(engagement); }
	_FORCE_INLINE_ void remove_engagement(RCSEngagementInternal* engagement) VEC_ERASE(engagement_loggers, engagement)
#endif

	friend class Sentrience;
	friend class RCSEngagementInternal;
	friend class RCSSimulation;
public:
	RCSTeam();
	~RCSTeam();

	_FORCE_INLINE_ void set_team_name(const StringName& new_name) { team_name = new_name; }
	_FORCE_INLINE_ StringName get_team_name() const { return team_name; }

	_FORCE_INLINE_ void set_team_id(const uint32_t& new_id) { team_id = new_id;}
	_FORCE_INLINE_ uint32_t get_team_id() const { return team_id; }

	Array get_engagements_ref() const;
	void purge_all_links();

#ifdef USE_SAFE_RID_COUNT
	_FORCE_INLINE_ std::weak_ptr<RCSTeam> self_cast_weakptr() const {
		return *((std::weak_ptr<RCSTeam> *)&self_ref);
	}
	_FORCE_INLINE_ WrappedVector<std::weak_ptr<RCSSquad>> get_squads() const { return squads; }

	void set_simulation(const std::weak_ptr<RCSSimulation> &sim);
	bool has_squad(const std::weak_ptr<RCSSquad> &squad);
	_FORCE_INLINE_ void add_squad(const std::weak_ptr<RCSSquad>& squad) {
		squads.push_back(squad);
		auto locked = squad.lock();
		locked->set_squad_id(refcount.refval());
	}
	bool remove_squad(const std::weak_ptr<RCSSquad> &squad);

	bool is_engagable(const std::weak_ptr<RCSTeam> &bogey);

	Ref<RCSUnilateralTeamsBind> add_link(const std::weak_ptr<RCSTeam> &toward);
	bool remove_link(const std::weak_ptr<RCSTeam> &to);
	Ref<RCSUnilateralTeamsBind> get_link_to(const std::weak_ptr<RCSTeam> &to) const;
	Ref<RCSUnilateralTeamsBind> get_link_to(const std::shared_ptr<RCSTeam> &to) const;
	_FORCE_INLINE_ bool has_link_to(const std::weak_ptr<RCSTeam> &to) const { return get_link_to(to).is_valid(); }
#else
	_FORCE_INLINE_ VECTOR<RCSSquad*>* get_squads() { return &squads; }

	virtual void set_simulation(RCSSimulation* sim);
	_FORCE_INLINE_ bool has_squad(RCSSquad* squad) const VEC_HAS(squads, squad)
	_FORCE_INLINE_ void add_squad(RCSSquad* squad) { squads.push_back(squad); squad->set_squad_id(refcount.refval());  }
	_FORCE_INLINE_ void remove_squad(RCSSquad* squad) VEC_ERASE(squads, squad);
	bool is_engagable(RCSTeam *bogey) const;

	Ref<RCSUnilateralTeamsBind> add_link(RCSTeam *toward);
	bool remove_link(RCSTeam *to);
	Ref<RCSUnilateralTeamsBind> get_link_to(RCSTeam *to) const;
	_FORCE_INLINE_ bool has_link_to(RCSTeam *to) const { return get_link_to(to).is_valid(); }
#endif
};

struct RadarPingCache {
	Vector3 from_origin;
	Vector3 from_forward;
	Vector3 to_origin;
	Vector3 to_dir;
	float distance = 0.0;
	float bearing = 0.0;
};

struct RadarPingRequest{
public:
#ifdef UngaBunga
	std::weak_ptr<RCSCombatant> from;
	std::weak_ptr<RCSCombatant> to;
#else
	RCSCombatant *from;
	RCSCombatant *to;
#endif
	Transform self_transform;
	bool detect_result = false;
	bool lock_result = false;
	bool late_recheck = false;

#ifdef UngaBunga
	RadarPingRequest(const std::weak_ptr<RCSCombatant>& from, const std::weak_ptr<RCSCombatant>& to) {
		this->from = from;
		this->to = to;
	}
#else
	RadarPingRequest(RCSCombatant *from, RCSCombatant *to) {
		this->from = from;
		this->to = to;
	}
#endif
};

class RCSRadarProfile : public Resource{
	GDCLASS(RCSRadarProfile, Resource);
public:
	enum RadarScanMode : unsigned int {
		ScanModeSwarm,
		ScanModeSingle,
	};
	enum RadarScanBase : unsigned int {
		ScanTransform,
		ScanDirectSpaceState,
	};
	enum RadarScanttributes : unsigned int {
		ScanCollideWithBodies = 0,
		ScanCollideWithAreas = 1,
	};
private:
	Ref<AdvancedCurve> detection_curve;
	Ref<AdvancedCurve> acquisition_curve;
	StringName rp_name;
	double spread = Math::deg2rad(30.0);
	double frequency = 1.0;
	double scan_contribution = 0.3;
	double rays_per_degree = 1.5;
	uint32_t collision_mask = 0xFFFFFFFF;
	uint32_t scan_attributes = ScanCollideWithBodies;
	RadarScanMode scan_method = RadarScanMode::ScanModeSingle;
	RadarScanBase scan_base = RadarScanBase::ScanTransform;
protected:
	static void _bind_methods();

	virtual RadarPingCache *ping_cache(RadarPingRequest* ping_request) const;
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

	_FORCE_INLINE_ void set_rpd(const double& new_rpd) { rays_per_degree = CLAMP(new_rpd, MIN_RPD, MAX_RPD); emit_changed(); }
	_FORCE_INLINE_ double get_rpd() const { return rays_per_degree; }

	_FORCE_INLINE_ void set_method(RadarScanMode new_met) { scan_method = new_met; emit_changed(); }
	_FORCE_INLINE_ RadarScanMode get_method() const { return scan_method; }

	_FORCE_INLINE_ void set_base(RadarScanBase new_base) { scan_base = new_base; emit_changed(); }
	_FORCE_INLINE_ RadarScanBase get_base() const { return scan_base; }

	_FORCE_INLINE_ void set_cmask(const uint32_t& new_mask) { collision_mask = new_mask; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_cmask() const { return collision_mask; }

	_FORCE_INLINE_ void set_attr(const uint32_t& new_attr) { scan_attributes = new_attr; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_attr() const { return scan_attributes; }

	_FORCE_INLINE_ void set_contribution(const double& new_con) { scan_contribution = CLAMP(new_con, 0.0001, 0.999); emit_changed(); }
	_FORCE_INLINE_ double get_contribution() const { return scan_contribution; }

	void ping_target(RadarPingRequest* ping_request);
	void swarm_detect(RadarPingRequest* ticket);
	virtual void internal_detect(RadarPingRequest* ping_request, RadarPingCache* cache);
	virtual void internal_acquire(RadarPingRequest* ping_request, RadarPingCache* cache);
	virtual void internal_swarm_detect(RadarPingRequest* ticket);
};
class RCSRadar: public RID_RCS{
private:
	PhysicsDirectSpaceState *space_state;
	Ref<RCSRadarProfile> rprofile;
	Array detected_rids;
	Array locked_rids;
	bool is_init = false;
	double timer = 0.0;

	void fetch_space_state();

	void ping_base_transform(const float& delta);
	void ping_base_direct_space_state(const float &delta);
	// bool detect(const float& delta, RCSCombatant* combatant) const

	friend class Sentrience;
	friend class RCSSimulation;
private:
#ifdef USE_SAFE_RID_COUNT
	std::weak_ptr<RCSSimulation> simulation;
	std::weak_ptr<RCSCombatant> assigned_vessel;
	WrappedVector<std::weak_ptr<RCSCombatant>> detected_combatants;
	WrappedVector<std::weak_ptr<RCSCombatant>> locked_combatants;
#else
	RCSSimulation* simulation;
	RCSCombatant* assigned_vessel;
	VECTOR<RCSCombatant*> detected_combatants;
	VECTOR<RCSCombatant*> locked_combatants;
#endif

public:
	RCSRadar();
	~RCSRadar();

	_FORCE_INLINE_ void set_profile(const Ref<RCSRadarProfile>& prof) { rprofile = prof; timer = 0.0; }
	_FORCE_INLINE_ Ref<RCSRadarProfile> get_profile() const { return rprofile; }

	// _FORCE_INLINE_ void set_ft(const Transform& trans) { fallback_transform = trans; }
	// _FORCE_INLINE_ Transform get_ft() const { return fallback_transform; }

	void poll(const float& delta) override;

	_FORCE_INLINE_ Array get_detected() const { return detected_rids; }
	_FORCE_INLINE_ Array get_locked() const { return locked_rids; }
#ifdef USE_SAFE_RID_COUNT
	_FORCE_INLINE_ std::weak_ptr<RCSRadar> self_cast_weakptr() const {
		return *((std::weak_ptr<RCSRadar> *)&self_ref);
	}

	void set_simulation(const std::weak_ptr<RCSSimulation> &sim);
	void late_check(const float &delta, const std::weak_ptr<RadarRecheckTicket> &recheck);
	void set_vessel(const std::weak_ptr<RCSCombatant> &new_vessel);
#else
	virtual void set_simulation(RCSSimulation* sim);
	virtual void late_check(const float& delta, RadarRecheckTicket* recheck);
	void set_vessel(RCSCombatant *new_vessel);
#endif
};

#endif

