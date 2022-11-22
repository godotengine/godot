#include "rcs_maincomp.h"

#define MainPoll(what, iter, delta)             \
	auto target = what[iter];                   \
	if (target) target->poll(delta)

#define RemoveReference(iter, inc, func, pool)  \
	if (iter < pool.size()){                    \
		pool[iter]->func(nullptr);              \
		inc++;                                  \
	}

VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanMode);

RCSSimulation::RCSSimulation(){}
RCSSimulation::~RCSSimulation(){
	uint32_t i = 0;
	while (true){
		uint16_t cond = 0;
		RemoveReference(i, cond, set_simulation, combatants);
		RemoveReference(i, cond, set_simulation, squads);
		RemoveReference(i, cond, set_simulation, teams);
		RemoveReference(i, cond, set_simulation, radars);
		if (cond == 0) break;
		i++;
	}
}

RCSRecording::RCSRecording(){

}
RCSRecording::~RCSRecording(){

}
void RCSRecording::recieve_event(RCSSimulation* handler, RID_RCS* from, void* event){

}

void RCSRecording::add_simulation(RCSSimulation* simul){
	simulations.push_back(simul);
	rids.push_back(simul->get_self());
	simul->set_recorder(this);
}
void RCSRecording::remove_simulation(RCSSimulation* simul){
	simulations.erase(simul);
	rids.erase(simul->get_self());
}
bool RCSRecording::has_simulation(RCSSimulation* simul) const{
	return (simulations.find(simul) != -1);
}

Vector<RID> RCSRecording::get_rids() const{
	return rids;
}

void RCSRecording::poll(const float& delta){

}

void RCSSimulation::add_combatant(RCSCombatant* com){
	combatants.push_back(com);
}
void RCSSimulation::add_squad(RCSSquad* squad){
	squads.push_back(squad);
}

void RCSSimulation::add_team(RCSTeam* team){
	teams.push_back(team);
}

void RCSSimulation::add_radar(RCSRadar* rad){
	radars.push_back(rad);
}

void RCSSimulation::remove_combatant(RCSCombatant* com){
	combatants.erase(com);
}
void RCSSimulation::remove_squad(RCSSquad* squad){
	squads.erase(squad);
}

void RCSSimulation::remove_team(RCSTeam* team){
	teams.erase(team);
}

void RCSSimulation::remove_radar(RCSRadar* rad){
	radars.erase(rad);
}

void RCSSimulation::set_recorder(RCSRecording* rec){
	recorder = rec;
}

void RCSSimulation::radar_request_recheck(RadarRecheckTicket* ticket){
	rrecheck_requests.push_back(ticket);
}

void RCSSimulation::poll(const float& delta){
	RID_RCS::poll(delta);
	// Radars run first for rechecks
	for (uint32_t k = 0; k < radars.size(); k++){
		MainPoll(radars, k, delta);
	}
	for (uint32_t u = 0; u < rrecheck_requests.size(); u++){
		auto ticket = rrecheck_requests[u];
		if (!ticket->request_sender) continue;
		ticket->request_sender->late_check(delta, ticket);
		delete ticket;
	}
	rrecheck_requests.clear();
	for (uint32_t j = 0; j < combatants.size(); j++){
		MainPoll(combatants, j, delta);
	}
	for (uint32_t i = 0; i < squads.size(); i++){
		MainPoll(squads, i, delta);
	}
}

RCSCombatantProfile::RCSCombatantProfile(){}

void RCSCombatantProfile::_bind_methods(){
	// ClassDB::bind_method(D_METHOD("set_id", "new_id"), &RCSCombatantProfile::set_id);
	// ClassDB::bind_method(D_METHOD("get_id"), &RCSCombatantProfile::get_id);

	ClassDB::bind_method(D_METHOD("set_stand", "new_stand"), &RCSCombatantProfile::set_stand);
	ClassDB::bind_method(D_METHOD("get_stand"), &RCSCombatantProfile::get_stand);

	ClassDB::bind_method(D_METHOD("set_pname", "name"), &RCSCombatantProfile::set_pname);
	ClassDB::bind_method(D_METHOD("get_pname"), &RCSCombatantProfile::get_pname);

	ClassDB::bind_method(D_METHOD("set_detection_threshold", "new_threshold"), &RCSCombatantProfile::set_detection_threshold);
	ClassDB::bind_method(D_METHOD("get_detection_threshold"), &RCSCombatantProfile::get_detection_threshold);

	ClassDB::bind_method(D_METHOD("set_subvention", "new_subvention"), &RCSCombatantProfile::set_subvention);
	ClassDB::bind_method(D_METHOD("get_subvention"), &RCSCombatantProfile::get_subvention);

	// ClassDB::bind_method(D_METHOD("_set_detection_meter", "new_param"), &RCSCombatantProfile::_set_detection_meter);
	// ClassDB::bind_method(D_METHOD("_get_detection_meter"), &RCSCombatantProfile::_get_detection_meter);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "combatant_name"), "set_pname", "get_pname");
	// ADD_PROPERTY(PropertyInfo(Variant::INT, "combatant_id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "combatant_stand"), "set_stand", "get_stand");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_threshold"), "set_detection_threshold", "get_detection_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_subvention"), "set_subvention", "get_subvention");
	// ADD_PROPERTY(PropertyInfo(Variant::REAL, "_detection_meter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_detection_meter", "_get_detection_meter");
}

RCSCombatant::RCSCombatant(){
	simulation = nullptr;
	my_squad = nullptr;
}

RCSCombatant::~RCSCombatant(){
	set_simulation(nullptr);
	set_squad(nullptr);
}

#define COMBATANT_DETECTION_LIMIT 10.0

void RCSCombatant::set_squad(RCSSquad* new_squad){
	if (my_squad){
		my_squad->remove_combatant(this);
		my_squad = nullptr;
	}
	my_squad = new_squad;
	combatant_id = 0;
	if (my_squad) my_squad->add_combatant(this);
}

bool RCSCombatant::is_same_team(RCSCombatant* com){
	if (!com) return false;
	if (!my_squad) return false;
	auto my_team = my_squad->get_team();
	if (!my_team) return false;
	auto bogey_squad = com->get_squad();
	if (!bogey_squad) return false;
	return (my_team == bogey_squad->get_team());
}

bool RCSCombatant::is_hostile(RCSCombatant* com){
	return !is_same_team(com);
}

bool RCSCombatant::is_ally(RCSCombatant* com){
	return is_same_team(com);
}

void RCSCombatant::poll(const float& delta){
	if (combatant_profile.is_null()) return;
	auto new_meter = detection_meter - combatant_profile->get_subvention();
	detection_meter = CLAMP(new_meter, 0.0, COMBATANT_DETECTION_LIMIT);
}

void RCSCombatant::set_cname(const StringName& cname){
	ERR_FAIL_COND(combatant_profile.is_null());
	combatant_profile->set_pname(cname);
}
StringName RCSCombatant::get_cname() const{
	ERR_FAIL_COND_V(combatant_profile.is_null(), StringName());
	return combatant_profile->get_pname();
}
// void RCSCombatant::set_id(const uint32_t& new_id){
// 	ERR_FAIL_COND(combatant_profile.is_null());
// 	combatant_profile->set_id(new_id);
// }
// uint32_t RCSCombatant::get_id() const{
// 	ERR_FAIL_COND_V(combatant_profile.is_null(), 0);
// 	return combatant_profile->get_id();
// }
void RCSCombatant::set_stand(const uint32_t& new_stand){
	ERR_FAIL_COND(combatant_profile.is_null());
	combatant_profile->set_stand(new_stand);
}
uint32_t RCSCombatant::get_stand() const{
	ERR_FAIL_COND_V(combatant_profile.is_null(), 0);
	return combatant_profile->get_stand();
}

Ref<RawRecord> RCSCombatant::serialize() const{
	Ref<RawRecord> rrec = memnew(RawRecord);
	Ref<RawRecordData> rdata = memnew(RawRecordData);
	rdata->name = StringName("__main");

	PUSH_RECORD_PRIMITIVE(rdata, space_transform);
	PUSH_RECORD_PRIMITIVE(rdata, local_transform);
	// PUSH_RECORD_PRIMITIVE(rdata, stand);

	rdata->external_refs.resize(rdata->table.size());
	rdata->external_refs.fill(0);
	return rrec;
}
bool RCSCombatant::serialize(const Ref<RawRecord>& from){
	return false;
}

void RCSCombatant::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_combatant(this);
		simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_combatant(this);
}

RCSSquad::RCSSquad(){
	simulation = nullptr;
	my_team = nullptr;
}

RCSSquad::~RCSSquad(){
	set_simulation(nullptr);
	set_team(nullptr);
	uint32_t i = 0;
	while (true) {
		uint16_t cond = 0;
		RemoveReference(i, cond, set_squad, combatants);
		if (cond == 0) break;
		i++;
	}
}

void RCSSquad::set_team(RCSTeam* new_team){
	if (my_team){
		my_team->remove_squad(this);
		my_team = nullptr;
	}
	my_team = new_team;
	squad_id = 0;
	if (my_team) my_team->add_squad(this);
}

void RCSSquad::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_squad(this);
		simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_squad(this);
}

RCSTeam::RCSTeam(){
	simulation = nullptr;
}

RCSTeam::~RCSTeam(){
	set_simulation(nullptr);
	uint32_t i = 0;
	while (true) {
		uint16_t cond = 0;
		RemoveReference(i, cond, set_team, squads);
		if (cond == 0) break;
		i++;
	}
}

void RCSTeam::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_team(this);
		simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_team(this);
}

RCSRadarProfile::RCSRadarProfile(){

}

#define RADAR_PROFILE_DETECT_VMETHOD StringName("_detect")
#define RADAR_PROFILE_LATE_DETECT_VMETHOD StringName("_late_detect")
#define RADAR_PROFILE_ACQUIRE_VMETHOD StringName("_acquire_lock")

void RCSRadarProfile::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_dcurve", "curve"), &RCSRadarProfile::set_dcurve);
	ClassDB::bind_method(D_METHOD("get_dcurve"), &RCSRadarProfile::get_dcurve);

	ClassDB::bind_method(D_METHOD("set_pname", "name"), &RCSRadarProfile::set_pname);
	ClassDB::bind_method(D_METHOD("get_pname"), &RCSRadarProfile::get_pname);

	ClassDB::bind_method(D_METHOD("set_spread", "new_spread"), &RCSRadarProfile::set_spread);
	ClassDB::bind_method(D_METHOD("get_spread"), &RCSRadarProfile::get_spread);

	ClassDB::bind_method(D_METHOD("set_freq", "new_freq"), &RCSRadarProfile::set_freq);
	ClassDB::bind_method(D_METHOD("get_freq"), &RCSRadarProfile::get_freq);

	ClassDB::bind_method(D_METHOD("set_method", "new_met"), &RCSRadarProfile::set_method);
	ClassDB::bind_method(D_METHOD("get_method"), &RCSRadarProfile::get_method);

	ClassDB::bind_method(D_METHOD("set_contribution", "new_con"), &RCSRadarProfile::set_contribution);
	ClassDB::bind_method(D_METHOD("get_contribution"), &RCSRadarProfile::get_contribution);

	BIND_ENUM_CONSTANT(ScanModeSwarm);
	BIND_ENUM_CONSTANT(ScanModeSingle);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "detection_curve", PROPERTY_HINT_RESOURCE_TYPE, "", PROPERTY_USAGE_DEFAULT, "AdvancedCurve"), "set_dcurve", "get_dcurve");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "profile_name"), "set_pname", "get_pname");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_spread", PROPERTY_HINT_RANGE, "0.009,3.141,0.001"), "set_spread", "get_spread");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_frequency", PROPERTY_HINT_RANGE, "0.001,5.0,0.001"), "set_freq", "get_freq");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "scan_contribution", PROPERTY_HINT_RANGE, "0.0001,0.999,0.001"), "set_contribution", "get_contribution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_method", PROPERTY_HINT_ENUM, "ScanModeSwarm,ScanModeSingle"), "set_method", "get_method");

	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_ACQUIRE_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_LATE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
}

void RCSRadarProfile::ping_target(RadarPingRequest* ping_request){
	auto script = get_script_instance();
	if (script && script->has_method(RADAR_PROFILE_DETECT_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_DETECT_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_detect(ping_request);
	if (script && script->has_method(RADAR_PROFILE_ACQUIRE_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_ACQUIRE_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_acquire(ping_request);
}
void RCSRadarProfile::swarm_detect(RadarPingRequest* ping_request){
	auto script = get_script_instance();
	if (script && script->has_method(RADAR_PROFILE_LATE_DETECT_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_LATE_DETECT_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_swarm_detect(ping_request);
}
void RCSRadarProfile::internal_detect(RadarPingRequest* ping_request){
	auto self_origin = ping_request->self_transform.get_origin();
	auto bearing = -(ping_request->self_transform.get_basis()[2]);
	auto bogey_transform = ping_request->to->get_combined_transform();
	auto bogey_origin = bogey_transform.get_origin();
	auto target_distance = self_origin.distance_to(bogey_origin);
	// Null safety is handled by RCSRadar
	// Check if bogey is out-of-range
	if (target_distance > detection_curve->get_range()) return;
	auto target_bearing = self_origin.direction_to(bogey_origin);
	// Check for bogey's bearing
	if (bearing.angle_to(target_bearing) > spread) return;
	auto bogey_profile = ping_request->to->get_profile();
	// If there is no profile presented for bogey,
	// do no further test and marked as detected
	if (bogey_profile.is_null()) {
		ping_request->detect_result = true;
		return;
	}
	auto dst_percentage = target_distance / detection_curve->get_range();
	auto scan_power = detection_curve->interpolate_baked(dst_percentage);
	switch (scan_method){
		case RadarScanMode::ScanModeSingle:
			// Simple scan check, if radar power overwhelm
			// bogey's threshold, mark as detected
			if (scan_power > bogey_profile->get_detection_threshold())
				ping_request->detect_result = true;
			break;
		case RadarScanMode::ScanModeSwarm:
			// More complicated swarm-based scan method
			// The bogey is eligible for "scan contribution" as soon as
			// it steps into the scan cone.
			// The base can contribution is always less than 1.0,
			// and the bogey can rapidly decrease its detection meter
			// so to detect a target, the radar must overwhelm it or have many radars
			// contributing at the same time
			// The bogey is detected once the detection meter is greater than 1.0
			// Since contribution runs in sequence, the detection meter must be rechecked
			// once every radars have done contributing.
			auto bogey_curr_dmeter = ping_request->to->_get_detection_meter();
			bogey_curr_dmeter += scan_contribution * scan_power;
			ping_request->to->_set_detection_meter(bogey_curr_dmeter);
			ping_request->late_recheck = true;
			break;
		// default: break;
	}
}

void RCSRadarProfile::internal_acquire(RadarPingRequest* ping_request){

}

void RCSRadarProfile::internal_swarm_detect(RadarPingRequest* ping_request){
	auto bogey_curr_dmeter = ping_request->to->_get_detection_meter();
	if (bogey_curr_dmeter >= 1.0)
		ping_request->detect_result = true;
}

RCSRadar::RCSRadar(){
	simulation = nullptr;
	assigned_vessel = nullptr;
}
RCSRadar::~RCSRadar(){}

void RCSRadar::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_radar(this);
		simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_radar(this);
}

void RCSRadar::poll(const float& delta){
	RID_RCS::poll(delta);
	if (rprofile.is_null() || !simulation || !assigned_vessel) return;
	timer += delta;
	if (timer >= ( 1.0 / rprofile->get_freq())){
		timer = 0.0;
		detected_combatants.clear();
		detected_rids.clear();
		locked_combatants.clear();
		locked_rids.clear();
		auto combatants = simulation->get_combatants();
		for (uint32_t i = 0; i < combatants->size(); i++){
			auto com = combatants->get(i);
			if (!com || assigned_vessel->is_ally(com)) continue;
			RadarPingRequest req(assigned_vessel, com);
			req.self_transform = assigned_vessel->get_combined_transform();
			rprofile->ping_target(&req);

			if (req.late_recheck){
				simulation->radar_request_recheck(new RadarRecheckTicket(this, com));
			} else if (req.detect_result) {
				detected_combatants.push_back(com);
				detected_rids.push_back(com->get_self());
			}
			if (req.lock_result){
				locked_combatants.push_back(com);
				locked_rids.push_back(com->get_self());
			}
		}
	}
}

void RCSRadar::late_check(const float& delta, RadarRecheckTicket* recheck){
	if (rprofile.is_null() || !simulation || !assigned_vessel) return;
	if (!recheck->bogey) return;
	RadarPingRequest req(assigned_vessel, recheck->bogey);
	req.self_transform = assigned_vessel->get_combined_transform();
	rprofile->swarm_detect(&req);
	if (req.detect_result){
		detected_combatants.push_back(recheck->bogey);
		detected_rids.push_back(recheck->bogey->get_self());
	}
}
