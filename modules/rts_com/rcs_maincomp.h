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

#define MIN_RADAR_SPREAD 0.008726646
#define MAX_RADAR_SPREAD Math_PI

#define MIN_RADAR_FREQ 0.001
#define MAX_RADAR_FREQ 5.0

#define MIN_RPD 0.1
#define MAX_RPD 5.0

class RCSCombatantProfile;
class RCSCombatant;
class RCSProjectileBind;
class RCSSquad;
class RCSUnilateralTeamsBind;
class RCSTeam;
class RCSRecording;
class RCSSimulation;
class RCSRadarProfile;
class RCSRadar;
class RCSEngagement;
class RCSEngagementInternal;

class RCSWorld;
class RCSStaticWorld;
class RCSSingleWorld;

class EventReport;
class CombatantEventReport;
class SquadEventReport;
class TeamEventReport;
class ProjectileEventReport;
class ComRadarEventReport;

#define REPORTER_CLASS(base, inherited)                                        \
	friend class RCSSimulation;                                                \
	friend class RCSRecording;                                                 \
protected:                                                                     \
	RCSSimulation *simulation = nullptr;                                       \
	static _FORCE_INLINE_ void _bind_methods() {}                              \
	_FORCE_INLINE_ Dictionary (base::*__get_primitive_describer())() {         \
		return &base::primitive_describe;                                      \
	}                                                                          \
public:                                                                        \
	_FORCE_INLINE_ RCSSimulation* get_simulation() { return simulation; }      \
private:


//--------------------------------------------------------------------------


struct RCSMemoryAllocation{
private:
	std::atomic<uint64_t> allocated;
	std::atomic<uint64_t> deallocated;
public:
	static RCSMemoryAllocation *tracker_ptr;
	RCSMemoryAllocation();
	~RCSMemoryAllocation();
	_ALWAYS_INLINE_ uint64_t add_allocated(const uint64_t& amount){
#ifdef DEBUG_ENABLED
		return allocated.fetch_add(amount, std::memory_order_acq_rel) + amount;
#else
		return 0;
#endif
	}
	_ALWAYS_INLINE_ uint64_t add_deallocated(const uint64_t& amount){
#ifdef DEBUG_ENABLED
		return deallocated.fetch_add(amount, std::memory_order_acq_rel) + amount;
#else
		return 0;
#endif
	}
	_ALWAYS_INLINE_ uint64_t currently_allocated() const {
#ifdef DEBUG_ENABLED
		auto al = allocated.load(std::memory_order_acquire);
		auto de = deallocated.load(std::memory_order_acquire);
		if (al < de) return 0;
		return al - de;
#else
		return 0;
#endif
	}

};

class EventReport {
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
class ProjectileEventReport : public EventReport {
	REPORTER_CLASS(ProjectileEventReport, EventReport);
public:
	enum PTR_EventType : unsigned int {
		NA = 0,
		HostSet = 1,
		TargetSet = 2,
		Activation = 3,
		Deactivation = 4,
		Marked = 5,
		Unmarked = 6,
	};
private:
	PTR_EventType event_type;
	RCSCombatant *package_a   = nullptr;
	RCSCombatant *package_b   = nullptr;
	RCSProjectileBind *sender = nullptr;

	uint32_t pcka_id = 0;
	uint32_t pckb_id = 0;
public:
	ProjectileEventReport();

	_FORCE_INLINE_ void set_sender(RCSProjectileBind* bind) { sender = bind; }
	_FORCE_INLINE_ RCSProjectileBind* get_sender() { return sender; }

	_FORCE_INLINE_ void set_event(PTR_EventType ev) { event_type = ev; }
	_FORCE_INLINE_ PTR_EventType get_event() const { return event_type; }

	void set_package_alpha(RCSCombatant *com = nullptr);
	_FORCE_INLINE_ RCSCombatant* get_package_alpha() { return package_a; }

	void set_package_beta(RCSCombatant *com = nullptr);
	_FORCE_INLINE_ RCSCombatant* get_package_beta() { return package_b; }

	// Combatants' pointer might be invalid later on, but their id won't
	_ALWAYS_INLINE_ uint32_t get_packagae_a_id() const noexcept { return pcka_id; }
	_ALWAYS_INLINE_ uint32_t get_packagae_b_id() const noexcept { return pckb_id; }
};
class ComRadarEventReport : public EventReport {
	REPORTER_CLASS(ComRadarEventReport, EventReport);
public:
	// Opaque record
	struct ScanConclusion{
		Vector<RID_TYPE> detected;
		Vector<RID_TYPE> locked;

		Vector<RID_TYPE> newly_detected;
		Vector<RID_TYPE> newly_locked;

		Vector<RID_TYPE> nolonger_detected;
		Vector<RID_TYPE> nolonger_locked;

		_ALWAYS_INLINE_ void add_detected(const RID_TYPE& id) noexcept { detected.push_back(id); }
		_ALWAYS_INLINE_ void add_locked(const RID_TYPE& id) noexcept { locked.push_back(id); }

		_ALWAYS_INLINE_ void add_newly_detected(const RID_TYPE& id) noexcept { newly_detected.push_back(id); }
		_ALWAYS_INLINE_ void add_newly_locked(const RID_TYPE& id) noexcept { newly_locked.push_back(id); }

		_ALWAYS_INLINE_ void add_nolonger_detected(const RID_TYPE& id) noexcept { nolonger_detected.push_back(id); }
		_ALWAYS_INLINE_ void add_nolonger_locked(const RID_TYPE& id) noexcept { nolonger_locked.push_back(id); }
	};
	enum RDR_EventType : unsigned int {
		NA = 0,
		ScanConcluded = 1,
	};

private:
	RDR_EventType event_type;
	RCSRadar* sender = nullptr;

	std::unique_ptr<ScanConclusion> conclusion;

	friend class RCSSimulation;
	friend class RCSRadar;
public:
	ComRadarEventReport();
};

class RCSSpatial : public RID_RCS {
public:
	RCSSpatial() = default;
	virtual ~RCSSpatial() = default;

	virtual Transform get_combined_transform() const { return Transform(); }
	virtual Transform get_global_transform() const { return Transform(); }
	virtual Transform get_local_transform() const { return Transform(); }
};

//--------------------------------------------------------------------------
class RCSWorld : public RID_RCS {
public:
	RCSWorld()  = default;
	virtual ~RCSWorld() = default;

	virtual void allocate_by_static_size(const double& world_width, const uint32_t& cell_per_row) = 0;
	virtual void allocate_by_dynamic_size(const double& cell_width, const uint32_t& cell_per_row) = 0;

	virtual uint32_t get_cell_count() const 		= 0;
	virtual uint32_t get_cell_per_row() const 		= 0;
	virtual double get_world_width() const 			= 0;
	virtual double get_cell_width() const 			= 0;

	virtual void add_unit(RCSSpatial *unit)			= 0;
	virtual void remove_unit(RCSSpatial *unit)		= 0;
	virtual uint32_t get_unit_count() const 		= 0;

};
class RCSSingleWorld : public RCSWorld {
public:
	struct UnitedWorld {
	private:
		// std::recursive_mutex main_lock;
	public:
		VECTOR<RCSSpatial*> combatants;

		_FORCE_INLINE_ uint32_t get_size() const { return combatants.size(); }
	};
private:
	UnitedWorld* inner_world;
public:
	RCSSingleWorld();
	~RCSSingleWorld();

	// Not available
	_FORCE_INLINE_ void allocate_by_static_size(const double& world_width, const uint32_t& cell_per_row) override {}
	_FORCE_INLINE_ void allocate_by_dynamic_size(const double& cell_width, const uint32_t& cell_per_row) override {}

	// Not available
	_FORCE_INLINE_ uint32_t get_cell_count() const override { return 0; }
	_FORCE_INLINE_ uint32_t get_cell_per_row() const override { return 0; }
	_FORCE_INLINE_ double get_world_width() const override { return 0.0F; }
	_FORCE_INLINE_ double get_cell_width() const override { return 0.0F; }

	_FORCE_INLINE_ UnitedWorld* get_inner_world() { return inner_world; }
	_FORCE_INLINE_ const UnitedWorld* get_inner_world() const { return inner_world; }

	void add_unit(RCSSpatial *unit) override{
		inner_world->combatants.push_back(unit);
	}
	void remove_unit(RCSSpatial *unit) override {
		VEC_ERASE(inner_world->combatants, unit);
	}
	_FORCE_INLINE_ uint32_t get_unit_count() const override { return inner_world->get_size(); }
};
class RCSStaticWorld : public RCSWorld {
public:
	struct StaticWorldCell;
	class StaticWorldPartition;
	// struct CellUpdateRequest;
	// struct CellsSynchronizer;
public:
	struct StaticWorldCell {
		VECTOR<RCSSpatial*> units;
	};
	class StaticWorldPartition {
	public:
	private:
		VECTOR<StaticWorldCell> cells_2d_array;
		// Cells per row
		uint32_t unit_count = 0;
		uint32_t cpr = 0;
		// cell_count == cpr
		// uint32_t row_count = 0;
		uint32_t cell_count = 0;
		double world_width = 0.0F;
		double cell_width = 0.0F;
		friend class RCSStaticWorld;
	public:
		StaticWorldPartition();
		~StaticWorldPartition();

		void deallocate_world() { cells_2d_array.clear(); this->unit_count = 0; }

		void allocate_by_static_size(const double& world_width, const uint32_t& cell_per_row);
		void allocate_by_dynamic_size(const double& cell_width, const uint32_t& cell_per_row);	

		_FORCE_INLINE_ StaticWorldCell* get_cell(const Vector2& loc) {
			uint32_t loc_1d = (loc.y * cpr) + loc.x;
			ERR_FAIL_COND_V_MSG(loc_1d >= cell_count, nullptr, "Index overflowed");
			return &cells_2d_array.get(loc_1d);
		}
		_FORCE_INLINE_ const StaticWorldCell* get_cell(const Vector2& loc) const {
			uint32_t loc_1d = (loc.y * cpr) + loc.x;
			ERR_FAIL_COND_V_MSG(loc_1d >= cell_count, nullptr, "Index overflowed");
			return &cells_2d_array.get(loc_1d);
		}
	};
private:
	StaticWorldPartition* static_world;
	// CellsSynchronizer *synchronizer;
public:
	RCSStaticWorld();
	~RCSStaticWorld();

	_FORCE_INLINE_ void allocate_by_static_size(const double& world_width, const uint32_t& cell_per_row) override { static_world->allocate_by_static_size(world_width, cell_per_row); }
	_FORCE_INLINE_ void allocate_by_dynamic_size(const double& cell_width, const uint32_t& cell_per_row) override { static_world->allocate_by_dynamic_size(cell_width, cell_per_row); }

	_FORCE_INLINE_ uint32_t get_cell_count() const override { return static_world->cell_count; }
	_FORCE_INLINE_ uint32_t get_cell_per_row() const override { return static_world->cpr; }
	_FORCE_INLINE_ double get_world_width() const override { return static_world->world_width; }
	_FORCE_INLINE_ double get_cell_width() const override { return static_world->cell_width; }

	_FORCE_INLINE_ void add_unit(RCSSpatial *unit, const Vector2& loc) { 
		auto cell = static_world->get_cell(loc);
		if (!cell) return;
		cell->units.push_back(unit);
		static_world->unit_count += 1;
	}
	_FORCE_INLINE_ void remove_unit(RCSSpatial *unit, const Vector2& loc) { 
		auto cell = static_world->get_cell(loc);
		if (!cell) return;
		VEC_ERASE(cell->units, unit)
		static_world->unit_count -= 1;
	}
	_FORCE_INLINE_ uint32_t get_unit_count() const override { return static_world->unit_count; }
};
//--------------------------------------------------------------------------

struct EventReportTicket {
	uint64_t timestamp = 0;
	std::shared_ptr<EventReport> event;
	EventReportTicket(const uint64_t& timestamp, const std::shared_ptr<EventReport>& event) { this->timestamp = timestamp; this->event = event; }
};

class RCSRecording : public RID_RCS {
private:
	VECTOR<std::shared_ptr<EventReportTicket>> reports_holder;
	uint64_t start_time_usec;
protected:
	void push_event(const std::shared_ptr<EventReport>& event);
	bool running = false;

	_ALWAYS_INLINE_ uint64_t get_timestamp() const { return (OS::get_singleton()->get_ticks_usec() - start_time_usec); }
public:
	RCSRecording();
	~RCSRecording();

	friend class RCSSimulation;
	friend class Sentrience;

	void poll(const float& delta) override;
	void purge();
	VECTOR<std::weak_ptr<EventReportTicket>> events_by_simulation(RCSSimulation* simulation) const;
	VECTOR<std::weak_ptr<EventReportTicket>> get_all_events() const;
};

struct RadarRecheckTicket {
	RCSRadar *request_sender;
	RCSCombatant *bogey;
	RadarRecheckTicket(RCSRadar *sender, RCSCombatant *reciever) {
		request_sender = sender;
		bogey = reciever;
	}
};
struct RadarConcludeTicket {
	RCSRadar *request_sender = nullptr;
	RadarConcludeTicket(RCSRadar* sender){ request_sender = sender; }
};

/* Opaque class */
class RCSEngagementInternal : public RID_RCS {
private:
	bool engaging;
	uint32_t scale;
	float time_since_last_action = 0.0;
	float degrade_to_finished = 120.0;
	float total_heat = 0.0;
	float time_elapsed = 0.0;
private:
	VECTOR<RID_TYPE> participating_teams;
	RID_TYPE offending_team;
	RID_TYPE deffending_team;
	RID_TYPE offending_squads;
	RID_TYPE deffending_squads;

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
	_ALWAYS_INLINE_ bool is_engagement_happening() const noexcept { return engaging; }
	_ALWAYS_INLINE_ bool is_engagement_over() const noexcept { return !engaging; }
	_ALWAYS_INLINE_ uint32_t get_scale() const noexcept { return scale; }
	_ALWAYS_INLINE_ float get_heat_meter() const noexcept { return (total_heat / time_elapsed); }
	_ALWAYS_INLINE_ float get_time_elapsed() const noexcept { return time_elapsed; }
	_ALWAYS_INLINE_ VECTOR<RID_TYPE> *_get_participating() { return &participating_teams; }
	_ALWAYS_INLINE_ RID_TYPE _get_offending_team()  const { return offending_team; }
	_ALWAYS_INLINE_ RID_TYPE _get_deffending_team() const { return deffending_team; }
	_ALWAYS_INLINE_ RID_TYPE _get_offending_squads() { return offending_squads; }
	_ALWAYS_INLINE_ RID_TYPE _get_deffending_squads() { return deffending_squads; }
// ---------------------------------------------
	Array get_involving_teams() const;
	Array get_involving_squads() const;
	_ALWAYS_INLINE_ RID_TYPE get_offending_team() const noexcept { return offending_team; }
	_ALWAYS_INLINE_ RID_TYPE get_deffending_team() const noexcept { return deffending_team; }
	_ALWAYS_INLINE_ RID_TYPE get_offending_squad() const noexcept { return offending_squads; }
	_ALWAYS_INLINE_ RID_TYPE get_deffending_squad() const noexcept { return deffending_squads; }
// ---------------------------------------------

	Ref<RCSEngagement> spawn_reference();

	_ALWAYS_INLINE_ void reset_action_timer() { time_since_last_action = 0.0; }
	void poll(const float& delta) override;
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
	_FORCE_INLINE_ RID_TYPE get_deffending_team() const		{ ERR_FAIL_COND_V(!logger, RID_TYPE()); return logger->get_deffending_team(); }
	_FORCE_INLINE_ RID_TYPE get_offending_squad() const	{ ERR_FAIL_COND_V(!logger, RID_TYPE()); return logger->get_offending_squad(); }
	_FORCE_INLINE_ RID_TYPE get_deffending_squad() const	{ ERR_FAIL_COND_V(!logger, RID_TYPE()); return logger->get_deffending_squad(); }
};

class RCSSimulation : public RID_RCS{
private:
	VECTOR<RCSCombatant *> combatants;
	VECTOR<RCSSquad *> squads;
	VECTOR<RCSTeam *> teams;
	VECTOR<RCSRadar *> radars;
	VECTOR<RadarRecheckTicket *> rrecheck_requests;
	VECTOR<RadarConcludeTicket*> rconclude_requests;
	RCSRecording* recorder;

	VECTOR<std::shared_ptr<RCSEngagementInternal>> engagements;
private:
	void ihandler_projectile_fired(std::shared_ptr<ProjectileEventReport>& event);
	void ihandler_radar_scan_concluded(std::shared_ptr<ComRadarEventReport>& event);
public:
	RCSSimulation();
	~RCSSimulation();

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

	std::weak_ptr<RCSEngagementInternal> find_active_engagement(const RID_TYPE& offending_squad, const RID_TYPE& defending_squad) const;
	VECTOR<std::weak_ptr<RCSEngagementInternal>> find_engagements(const RID_TYPE& offending_squad, const RID_TYPE& defending_squad) const;
	_FORCE_INLINE_ std::weak_ptr<RCSEngagementInternal> create_engagement(){
		auto en = std::make_shared<RCSEngagementInternal>();
		engagements.push_back(en);
		return en;
	}

	// World related
	Vector<RCSSquad*> request_scanable_squads(const RCSSquad* base) const;
	Vector<RCSSquad*> request_reachable_squads(const RCSSquad* from) const;
	Vector<RCSCombatant*> request_reachable_combatants(const RCSSquad* from) const;
	// ---------------------------
	_FORCE_INLINE_ uint32_t get_allocatable_workers() const {
		// auto size = combatants.size();
		// uint32_t re = 1U;
		// if (size < 50) re = 1;
		// else if (size < 200) re = 2;
		// else if (size < 2000) re = 4;
		// else re = 8;
		// return re;
		return MAX_ALLOCATABLE_WORKER_PER_OBJECT;
	}

	void set_recorder(RCSRecording *rec);
	_FORCE_INLINE_ bool has_recorder() const { return recorder != nullptr; }
	_FORCE_INLINE_ bool is_recording() const { return (has_recorder() ? (recorder->running) : false); }
	void combatant_event(std::shared_ptr<CombatantEventReport>& event);
	void squad_event(std::shared_ptr<SquadEventReport>& event);
	void team_event(std::shared_ptr<TeamEventReport>& event);
	void projectile_event(std::shared_ptr<ProjectileEventReport>& event);
	void radar_event(std::shared_ptr<ComRadarEventReport>& event);

	RCSCombatant *get_combatant_from_iid(const uint64_t &iid) const;

	void radar_request_recheck(RadarRecheckTicket* ticket) { rrecheck_requests.push_back(ticket); }
	void radar_request_conclude(RadarConcludeTicket* ticket) { rconclude_requests.push_back(ticket); }


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
	bool is_phantom = false;
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

	_FORCE_INLINE_ void set_phantom_mode(const bool& pm) { is_phantom = pm; emit_changed(); }
	_FORCE_INLINE_ bool get_phantom_mode() const { return is_phantom; }
};

#define COMBATANT_DETECTION_LIMIT 10.0

class RCSCombatant : public RCSSpatial {
private:
	RCSSimulation *simulation;
	RCSSquad *my_squad;
	Ref<RCSCombatantProfile> combatant_profile;
	// RCSProjectileBind *projectile_bind;
	std::unique_ptr<RCSProjectileBind> projectile_bind;
	RCSCombatant* fired_by;
	VECTOR<RCSProjectileBind*> punishers;
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

	_FORCE_INLINE_ void set_fired_by(RCSCombatant* by) { fired_by = by; }
	_FORCE_INLINE_ RCSCombatant* get_fired_by() { return fired_by; }

	// _FORCE_INLINE_ RCSProjectileBind *get_projectile_bind() { return projectile_bind; }
	_FORCE_INLINE_ RCSProjectileBind *get_projectile_bind() { return projectile_bind.get(); }
	_FORCE_INLINE_ void add_punisher(RCSProjectileBind* bind){ punishers.push_back(bind);  }
	_FORCE_INLINE_ void erase_punisher(RCSProjectileBind *bind) VEC_ERASE(punishers, bind)
	_FORCE_INLINE_ bool has_punisher(RCSProjectileBind *bind) const VEC_HAS(punishers, bind)

	virtual void set_st(const Transform& trans) { space_transform = trans;}

	virtual void set_lt(const Transform& trans) { local_transform = trans;}

	_FORCE_INLINE_ Transform get_combined_transform() const override { return space_transform * local_transform;  }
	_FORCE_INLINE_ Transform get_global_transform() const override { return space_transform;  }
	_FORCE_INLINE_ Transform get_local_transform() const override { return local_transform;  }

	virtual void set_cname(const StringName& cname);
	virtual StringName get_cname() const;

	virtual void set_stand(const uint32_t& new_stand);
	virtual uint32_t get_stand() const;

	virtual void set_status(const uint32_t& new_status) { status = new_status;  }
	virtual uint32_t get_status() const { return status; }

	void set_squad(RCSSquad *new_squad);
	RCSSquad *get_squad() { return my_squad; }

	bool is_same_team(RCSCombatant *com);
	bool is_hostile(RCSCombatant *com);
	bool is_ally(RCSCombatant *com);
	bool is_engagable(RCSCombatant *bogey) const;
	bool is_scannable(RCSCombatant *bogey) const;

	virtual void set_simulation(RCSSimulation *sim);
	RCSSimulation* get_simulation() { return simulation; }
};

class RCSProjectileBind {
private:
	RCSCombatant *host = nullptr;
	RCSCombatant *fired_by = nullptr;
	RCSCombatant *target = nullptr;
	bool is_marked = false;
	bool is_active = false;

	// friend class RCSCombatant;
public:
	RCSProjectileBind() = default;

	_ALWAYS_INLINE_ void set_host(RCSCombatant *new_host){
		auto old_host = host;
		host = new_host;
		if (!host) return;
		auto sim = host->get_simulation();
		if (!sim) return;
		std::shared_ptr<ProjectileEventReport> event = std::make_shared<ProjectileEventReport>();
		event->set_sender(this);
		event->set_event(ProjectileEventReport::HostSet);
		event->set_package_alpha(old_host);
		event->set_package_beta(new_host);
		sim->projectile_event(event);
	}
	_ALWAYS_INLINE_ RCSCombatant* get_host() noexcept { return host; }

	_ALWAYS_INLINE_ void set_marked(const bool& mark) {
		is_marked = mark;
		if (!host) return;
		auto sim = host->get_simulation();
		if (!sim) return;
		std::shared_ptr<ProjectileEventReport> event = std::make_shared<ProjectileEventReport>();
		event->set_sender(this);
		if (is_marked) event->set_event(ProjectileEventReport::Marked);
		else event->set_event(ProjectileEventReport::Unmarked);
		sim->projectile_event(event);
	}
	_ALWAYS_INLINE_ bool get_marked() const noexcept { return is_marked; }

	_ALWAYS_INLINE_ void set_state(const bool& state) {
		is_active = state;
		if (!host) return;
		auto sim = host->get_simulation();
		if (!sim) return;
		std::shared_ptr<ProjectileEventReport> event = std::make_shared<ProjectileEventReport>();
		event->set_sender(this);
		if (is_active) event->set_event(ProjectileEventReport::Activation);
		else event->set_event(ProjectileEventReport::Deactivation);
		sim->projectile_event(event);
	}
	_ALWAYS_INLINE_ bool get_state() const noexcept { return is_active; }

	_ALWAYS_INLINE_ void set_target(RCSCombatant *tar) { 
		auto old_target = target;
		if (target){
			target->erase_punisher(this);
		}
		target = tar;
		if (target) target->add_punisher(this);
		if (!host) return;
		auto sim = host->get_simulation();
		if (!sim) return;
		std::shared_ptr<ProjectileEventReport> event = std::make_shared<ProjectileEventReport>();
		event->set_sender(this);
		event->set_event(ProjectileEventReport::TargetSet);
		event->set_package_alpha(old_target);
		event->set_package_beta(tar);
		sim->projectile_event(event);
	}
	_ALWAYS_INLINE_ RCSCombatant* get_target() noexcept { return target; }
};

class RCSSquad : public RCSSpatial {
public:
	struct SquadSummarization {
		Vector3 center;
		Vector2 height_differences;
		real_t radius_squared = 0.0F;
	};
private:
	StringName squad_name;
	uint32_t squad_id = 0;
	SafeRefCount refcount;
	VECTOR<std::weak_ptr<RCSEngagementInternal>> participating;
private:
	RCSSimulation* simulation;
	RCSTeam *my_team;

	VECTOR<RCSCombatant*> combatants;

	// friend class RCSEngagementInternal;
	friend class RCSSimulation;
	friend class Sentrience;
public:
	RCSSquad();
	~RCSSquad();

	_FORCE_INLINE_ void add_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		participating.push_back(engagement);
	}
	_FORCE_INLINE_ int64_t find_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		for (uint32_t i = 0, size = participating.size(); i < size; i++){
			auto curr = participating[i];
			auto locked = curr.lock();
			if (!(locked.operator bool())){
				VEC_REMOVE(participating, i);
				i -= 1;
				continue;
			} else if (locked == engagement){
				return i;
			}
		}
		return -1;
	}
	_FORCE_INLINE_ bool has_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		return find_participating(engagement) != -1;
	}
	_FORCE_INLINE_ void remove_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		auto idx = find_participating(engagement);
		if (idx == -1) return;
		VEC_REMOVE(participating, idx);
	}

	_FORCE_INLINE_ void set_squad_name(const StringName& new_name) { squad_name = new_name; }
	_FORCE_INLINE_ StringName get_squad_name() const { return squad_name; }
	_FORCE_INLINE_ void set_squad_id(const uint32_t& new_id) { squad_id = new_id;}
	_FORCE_INLINE_ uint32_t get_squad_id() const { return squad_id; }

	_FORCE_INLINE_ Vector3 get_spatial_position(const uint32_t& interval = 1) const {
		Vector3 raw_combined;
		uint32_t i = 0, size = combatants.size(), count = 0;
		for ( ; i < size; i += interval){
			auto com = combatants[i];
			if (!com) continue;
			raw_combined += com->get_combined_transform().get_origin();
			count++;
		}
		return (raw_combined / count);
	}
	_FORCE_INLINE_ real_t get_combatants_radius_squared(const Vector3& center, const real_t& padding = 0.0F) const {
		real_t largest = 0.0F;
		uint32_t i = 0, size = combatants.size(), count = 0;
		for ( ; i < size; i += 1){
			auto com = combatants[i];
			if (!com) continue;
			//----------------------------------
			const Vector3& com_loc = combatants[i]->get_combined_transform().get_origin();
			auto distance_squared = center.distance_squared_to(com_loc);
			//----------------------------------
			count++;
		}
		return (largest / count) + padding;
	}
	_FORCE_INLINE_ Vector2 get_combatants_displacement(const Vector3& center, const real_t& padding = 0.0F) const {
		real_t largest_pos = 0.0F, larget_neg = 0.0F;
		uint32_t i = 0, size = combatants.size();
		for ( ; i < size; i += 1){
			auto com = combatants[i];
			if (!com) continue;
			//----------------------------------
			const Vector3& com_loc = combatants[i]->get_combined_transform().get_origin();
			if (com_loc.y < center.y){
				if (com_loc.y < larget_neg) larget_neg = com_loc.y;
			} else if ( com_loc.y > center.y){
				if (com_loc.y > largest_pos) largest_pos = com_loc.y;
			}
			//----------------------------------
		}
		return Vector2(largest_pos + padding, larget_neg - padding);
	}
	_FORCE_INLINE_ SquadSummarization get_summary(const uint32_t& mask = 3U,const real_t& padding = 0.0F) const {
		SquadSummarization sum;
		sum.center = get_spatial_position();
		if (mask & 1U)
			sum.radius_squared = get_combatants_radius_squared(sum.center, padding);
		if (mask & 2U)
			sum.height_differences = get_combatants_displacement(sum.center, padding);
		return sum;
	}
	_FORCE_INLINE_ Transform get_global_transform() const override {
		return Transform(Basis(), get_spatial_position());
	}
	_FORCE_INLINE_ VECTOR<RCSCombatant *> *get_combatants() { return &combatants; }
	_FORCE_INLINE_ const VECTOR<RCSCombatant *> *get_combatants() const { return &combatants; }
	_FORCE_INLINE_ uint32_t get_combatant_count() const { return combatants.size(); }

	void set_team(RCSTeam *new_team);
	RCSTeam *get_team() { return my_team; }

	virtual void set_simulation(RCSSimulation *sim);
	_FORCE_INLINE_ bool has_combatant(RCSCombatant *com) const VEC_HAS(combatants, com)
			_FORCE_INLINE_ void add_combatant(RCSCombatant *com) {
		combatants.push_back(com);
		com->set_combatant_id(refcount.refval());
	}
	_FORCE_INLINE_ void remove_combatant(RCSCombatant *com) VEC_ERASE(combatants, com)
	bool is_engagable(RCSSquad *bogey) const ;
	bool is_scannable(RCSSquad *bogey) const ;
	bool is_same_team(RCSSquad *bogey) {
		if (!bogey)
			return false;
		return bogey->get_team() == my_team;
	}
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
		ITA_Scanable				= 8,
		// Ally attributes
		ITA_ShareRadar				= 32,
		ITA_ShareLocation			= 64,
	};
private:
	uint32_t relationship = TeamNeutral;
	uint32_t attributes = ITA_Scanable;

	RCSTeam *toward = nullptr;
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

	_FORCE_INLINE_ void set_to_rid(RCSTeam *to) { toward = to; emit_changed(); }

	RID_TYPE get_to_rid() const;
};

class RCSTeam : public RID_RCS {
private:
	StringName team_name;
	uint32_t team_id = 0;
	SafeRefCount refcount;
	VECTOR<std::weak_ptr<RCSEngagementInternal>> participating;
private:
	RCSSimulation* simulation;
	VECTOR<RCSSquad*> squads;
	VECTOR<Ref<RCSUnilateralTeamsBind>> team_binds;
	VECTOR<RCSEngagementInternal*> engagement_loggers;

	_FORCE_INLINE_ void add_engagement(RCSEngagementInternal* engagement) { engagement_loggers.push_back(engagement); }
	_FORCE_INLINE_ void remove_engagement(RCSEngagementInternal* engagement) VEC_ERASE(engagement_loggers, engagement)

	friend class Sentrience;
	friend class RCSEngagementInternal;
	friend class RCSSimulation;
public:
	RCSTeam();
	~RCSTeam();

	_FORCE_INLINE_ void add_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		participating.push_back(engagement);
	}
	_FORCE_INLINE_ int64_t find_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		for (uint32_t i = 0, size = participating.size(); i < size; i++){
			auto curr = participating[i];
			auto locked = curr.lock();
			if (!locked){
				VEC_REMOVE(participating, i);
				i -= 1;
				continue;
			} else if (locked == engagement){
				return i;
			}
		}
		return -1;
	}
	_FORCE_INLINE_ bool has_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		return find_participating(engagement) != -1;
	}
	_FORCE_INLINE_ void remove_participating(const std::shared_ptr<RCSEngagementInternal>& engagement){
		auto idx = find_participating(engagement);
		if (idx == -1) return;
		VEC_REMOVE(participating, idx);
	}

	_FORCE_INLINE_ void set_team_name(const StringName& new_name) { team_name = new_name; }
	_FORCE_INLINE_ StringName get_team_name() const { return team_name; }

	_FORCE_INLINE_ void set_team_id(const uint32_t& new_id) { team_id = new_id;}
	_FORCE_INLINE_ uint32_t get_team_id() const { return team_id; }

	Array get_engagements_ref() const;
	void purge_all_links();

	_FORCE_INLINE_ VECTOR<RCSSquad*>* get_squads() { return &squads; }

	virtual void set_simulation(RCSSimulation* sim);
	_FORCE_INLINE_ bool has_squad(RCSSquad* squad) const VEC_HAS(squads, squad)
	_FORCE_INLINE_ void add_squad(RCSSquad* squad) { squads.push_back(squad); squad->set_squad_id(refcount.refval());  }
	_FORCE_INLINE_ void remove_squad(RCSSquad* squad) VEC_ERASE(squads, squad);
	_FORCE_INLINE_ bool is_engagable(RCSTeam *bogey) const {
		if (!bogey || bogey == this) return false;
		auto relation = get_link_to(bogey);
		if (relation.is_null()) return false;
		return (relation->get_attributes() & RCSUnilateralTeamsBind::ITA_Engagable);
	}
	_FORCE_INLINE_ bool is_scannable(RCSTeam *bogey) const {
		if (!bogey || bogey == this) return false;
		auto relation = get_link_to(bogey);
		if (relation.is_null()) return false;
		return (relation->get_attributes() & RCSUnilateralTeamsBind::ITA_Scanable);
	}

	Ref<RCSUnilateralTeamsBind> add_link(RCSTeam *toward);
	bool remove_link(RCSTeam *to);
	Ref<RCSUnilateralTeamsBind> get_link_to(RCSTeam *to) const;
	_FORCE_INLINE_ bool has_link_to(RCSTeam *to) const { return get_link_to(to).is_valid(); }
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
	bool compute_lock = false;
	bool detect_result = false;
	bool lock_result = false;
	bool late_recheck = false;

#ifdef UngaBunga
	RadarPingRequest(const std::weak_ptr<RCSCombatant>& from, const std::weak_ptr<RCSCombatant>& to, const bool& cl = false) {
		this->from = from;
		this->to = to;
		this->compute_lock = cl;
	}
#else
	RadarPingRequest(RCSCombatant *from, RCSCombatant *to, const bool& cl = false) {
		this->from = from;
		this->to = to;
		this->compute_lock = cl;
	}
#endif
};

class RCSRadarProfile : public Resource{
	GDCLASS(RCSRadarProfile, Resource);
public:
	struct RadarPingCache {
		Vector3 from_origin;
		Vector3 from_forward;
		Vector3 to_origin;
		Vector3 to_dir;
		float distance = 0.0;
		float bearing = 0.0;
		// Vector<RCSSquad*> auto_detect;
	};
public:
	enum RadarScanMode : unsigned int {
		ScanModeSwarm,
		ScanModeSingle,
	};
	enum RadarScanBase : unsigned int {
		ScanTransform,
		ScanDirectSpaceState,
	};
	enum RadarScanAttributes : unsigned int {
		ScanCollideWithBodies = 0,
		ScanCollideWithAreas = 1,
	};
	enum RadarTargetMode : unsigned int {
		TargetCombatants,
		TargetSquadPartial,
		// TargetSquadCompleted,
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
	RadarTargetMode scan_target = RadarTargetMode::TargetCombatants;
protected:
	static void _bind_methods();

	virtual RadarPingCache *ping_cache(RadarPingRequest* ping_request) const;
public:
	RCSRadarProfile();
	~RCSRadarProfile() = default;

	_FORCE_INLINE_ void set_dcurve(const Ref<AdvancedCurve>& curve) { detection_curve = curve; emit_changed(); }
	_FORCE_INLINE_ Ref<AdvancedCurve> get_dcurve() const noexcept { return detection_curve; }

	_FORCE_INLINE_ void set_acurve(const Ref<AdvancedCurve>& curve) { acquisition_curve = curve; emit_changed(); }
	_FORCE_INLINE_ Ref<AdvancedCurve> get_acurve() const noexcept { return acquisition_curve; }

	_FORCE_INLINE_ void set_pname(const StringName& name) { rp_name = name; emit_changed(); }
	_FORCE_INLINE_ StringName get_pname() const noexcept { return rp_name; }

	_FORCE_INLINE_ void set_spread(const double& new_spread) { spread = CLAMP(new_spread, MIN_RADAR_SPREAD, MAX_RADAR_SPREAD); emit_changed(); }
	_FORCE_INLINE_ double get_spread() const noexcept { return spread; }

	_FORCE_INLINE_ void set_freq(const double& new_freq) { frequency = CLAMP(new_freq, MIN_RADAR_FREQ, MAX_RADAR_FREQ); emit_changed(); }
	_FORCE_INLINE_ double get_freq() const noexcept { return frequency; }

	_FORCE_INLINE_ void set_rpd(const double& new_rpd) { rays_per_degree = CLAMP(new_rpd, MIN_RPD, MAX_RPD); emit_changed(); }
	_FORCE_INLINE_ double get_rpd() const noexcept { return rays_per_degree; }

	_FORCE_INLINE_ void set_method(RadarScanMode new_met) { scan_method = new_met; emit_changed(); }
	_FORCE_INLINE_ RadarScanMode get_method() const noexcept { return scan_method; }

	_FORCE_INLINE_ void set_base(RadarScanBase new_base) { scan_base = new_base; emit_changed(); }
	_FORCE_INLINE_ RadarScanBase get_base() const noexcept { return scan_base; }

	_FORCE_INLINE_ void set_target_mode(RadarTargetMode new_tm) { scan_target = new_tm; emit_changed(); }
	_FORCE_INLINE_ RadarTargetMode get_target_mode() const noexcept { return scan_target; }

	_FORCE_INLINE_ void set_cmask(const uint32_t& new_mask) { collision_mask = new_mask; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_cmask() const noexcept { return collision_mask; }

	_FORCE_INLINE_ void set_attr(const uint32_t& new_attr) { scan_attributes = new_attr; emit_changed(); }
	_FORCE_INLINE_ uint32_t get_attr() const noexcept { return scan_attributes; }

	_FORCE_INLINE_ void set_contribution(const double& new_con) { scan_contribution = CLAMP(new_con, 0.0001, 0.999); emit_changed(); }
	_FORCE_INLINE_ double get_contribution() const noexcept { return scan_contribution; }

	void ping_target(RadarPingRequest* ping_request);
	void swarm_detect(RadarPingRequest* ticket);
	virtual void internal_detect(RadarPingRequest* ping_request, RadarPingCache* cache);
	virtual void internal_acquire(RadarPingRequest* ping_request, RadarPingCache* cache);
	virtual void internal_swarm_detect(RadarPingRequest* ticket);
};
class RCSRadar : public RID_RCS{
public:
	struct TargetScreeningResult {
		// Reference-able Vectors
		Vector<RCSSquad*> scanable_squads;
		Vector<RCSCombatant*> scanable_combatants;
		_FORCE_INLINE_ Vector<RCSCombatant*> get_combatants() const {
			Vector<RCSCombatant*> re;
			for (uint32_t i = 0, size = scanable_squads.size(); i < size; i++){
				auto squad = scanable_squads.get(i);
				auto combatants = squad->get_combatants();
				for (uint32_t j = 0, inner_size = combatants->size(); j < inner_size; j++){
					re.push_back(combatants->get(j));
				}
			}
			for (uint32_t i = 0, size = scanable_combatants.size(); i < size; i++){
				re.push_back(scanable_combatants.get(i));
			}
			return re;
		}
		_FORCE_INLINE_ Vector<RCSSquad*> get_squads() const {
			Vector<RCSSquad*> re;
			for (uint32_t i = 0, size = scanable_squads.size(); i < size; i++){
				re.push_back(scanable_squads.get(i));
			}
			for (uint32_t i = 0, size = scanable_combatants.size(); i < size; i++){
				RCSSquad* squad = scanable_combatants.get(i)->get_squad();
				if (re.find(squad) != -1) continue;
				re.push_back(squad);
			}
			return re;
		}
		_FORCE_INLINE_ uint32_t get_total_size() const {
			uint32_t re = scanable_combatants.size();
			for (uint32_t i = 0, size = scanable_squads.size(); i < size; i++){
				re += scanable_squads.get(i)->get_combatant_count();
			}
			return re;
		}
	};
private:
	TargetScreeningResult screening_result;
	PhysicsDirectSpaceState *space_state;
	Ref<RCSRadarProfile> rprofile;

	// All Lists are opaque to avoid referencing dangling pointers

	// Up-to-date list, reset after one radar cycle
	List<RID_TYPE> detected_squads;

	List<RID_TYPE> detected;
	List<RID_TYPE> locked;

	// Fodder lists, become unusable after 1st iteration
	List<RID_TYPE> prev_detected;
	List<RID_TYPE> prev_locked;

	// Positive-changes list, reset after one radar cycle
	List<RID_TYPE> newly_detected;
	List<RID_TYPE> newly_locked;

	// Negative-changes list, reset after one radar cycle
	List<RID_TYPE> nolonger_detected;
	List<RID_TYPE> nolonger_locked;

	bool is_init = false;
	double timer = 0.0;

	void fetch_space_state();
	void combatant_detected(RCSCombatant* com);
	void combatant_locked(RCSCombatant* com);
	void send_conclusion();

	void ping_base_transform(const float& delta);
	void ping_base_direct_space_state(const float &delta);
	// bool detect(const float& delta, RCSCombatant* combatant) const

	friend class Sentrience;
	friend class RCSSimulation;
private:
	RCSSimulation* simulation;
	RCSCombatant* assigned_vessel;
public:
	RCSRadar();
	~RCSRadar();

	_FORCE_INLINE_ void set_profile(const Ref<RCSRadarProfile>& prof) { rprofile = prof; timer = 0.0; }
	_FORCE_INLINE_ Ref<RCSRadarProfile> get_profile() const { return rprofile; }

	void poll(const float& delta) override;


	// No Ambigous Vector type as these methods is used by Sentrience
	_FORCE_INLINE_ Vector<RID_TYPE> get_detected() const {
		Vector<RID_TYPE> re;
		P2PVectorCopy(detected, re);
		return re;
	}
	_FORCE_INLINE_ Vector<RID_TYPE> get_locked() const { 
		Vector<RID_TYPE> re;
		P2PVectorCopy(locked, re);
		return re;
	}
	_FORCE_INLINE_ Vector<RID_TYPE> get_newly_detected() const {
		Vector<RID_TYPE> re;
		P2PVectorCopy(newly_detected, re);
		return re;
	}
	_FORCE_INLINE_ Vector<RID_TYPE> get_newly_locked() const { 
		Vector<RID_TYPE> re;
		P2PVectorCopy(newly_locked, re);
		return re;
	}
	_FORCE_INLINE_ Vector<RID_TYPE> get_nolonger_detected() const {
		Vector<RID_TYPE> re;
		P2PVectorCopy(nolonger_detected, re);
		return re;
	}
	_FORCE_INLINE_ Vector<RID_TYPE> get_nolonger_locked() const { 
		Vector<RID_TYPE> re;
		P2PVectorCopy(nolonger_locked, re);
		return re;
	}
	virtual void set_simulation(RCSSimulation* sim);
	virtual void late_check(const float& delta, RadarRecheckTicket* recheck);
	virtual void conclude(const float& delta);
	void set_vessel(RCSCombatant *new_vessel);

	TargetScreeningResult target_screening() const;
	_FORCE_INLINE_ Vector<RID_TYPE> target_screening_proxy() const {
		auto res = target_screening().get_combatants();
		Vector<RID_TYPE> re;
		for (uint32_t i = 0, size = res.size(); i < 0; i++){
			re.push_back(res.get(i)->get_self());
		}
		return re;
	}
};

#endif

