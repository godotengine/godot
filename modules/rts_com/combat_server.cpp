#include "combat_server.h"

Sentrience* Sentrience::singleton = nullptr;

#define RCSMakeRID(rid_owner, rdata)                                    \
	RID rid = rid_owner.make_rid(rdata);                                \
	rdata->set_self(rid);                                               \
	rdata->_set_combat_server(this);                                    \
	all_rids.push_back(rid);                                            \
	return rid
#define GetSimulation(rid_owner, rid)                                   \
	auto target = rid_owner.get(rid);                                   \
	if (target == nullptr) return RID();                                \
	auto sim = target->simulation;                                      \
	if (sim == nullptr) return RID();                                   \
	return sim->get_self()
#define FreeLog(component_name, rid)                                    \
	log(std::string("Freeing ") + std::string(#component_name)          \
	 + std::string(" with RID ") + std::to_string(rid.get_id()))
#define RIDSort(owner, type, rid)                                       \
	if (owner.owns(rid)) return String(#type);
#define OverweightAssert(container)                                     \
	ERR_FAIL_COND(container.size() >= MAX_OBJECT_PER_CONTAINER)
#define OverweightAssertReturn(container, re)                           \
	ERR_FAIL_COND_V(container.size() >= MAX_OBJECT_PER_CONTAINER, re)

#define RCS_DEBUG

Sentrience::Sentrience(){
	ERR_FAIL_COND(singleton);
	singleton = this;
	// RCSMemoryAllocationPtr = rcsnew(RCSMemoryAllocation);
	active = false;
}

Sentrience::~Sentrience(){
	// flush_instances_pool();
	if (!all_rids.empty()) {
		auto size = all_rids.size();
		log(String("Combat Server exitted with ") + itos(size) + String(" instance(s) still in use."));
		for (uint32_t i = 0; i < size; i++){
			auto rid = all_rids[i];
			log(String("In use: ") + rid_sort(rid) + ":" + itos(rid.get_id()));
		}
	}
	auto rcs_alloc = RCSMemoryAllocation::tracker_ptr->currently_allocated();
	if (rcs_alloc > 0){
		log(String("There is currently ") + String::humanize_size(rcs_alloc) + String(" of memory spawned by Sentrience that hasn\'t been freed."));
	}
	singleton = nullptr;
	// rcsdel(RCSMemoryAllocationPtr);
}

String Sentrience::rid_sort(const RID& target){
	RIDSort(recording_owner, RCSRecording, target)
	else RIDSort(simulation_owner, RCSSimulation, target)
	else RIDSort(combatant_owner, RCSCombatant, target)
	else RIDSort(squad_owner, RCSSquad, target)
	else RIDSort(team_owner, RCSTeam, target)
	else RIDSort(radar_owner, RCSRadar, target)
	return String("<unknown>");
}

void Sentrience::log(const String& msg){
	print_verbose(String("[Sentrience] ") + msg);
}
void Sentrience::log(const std::string& msg){
	print_verbose(String("[Sentrience] ") + String(msg.c_str()));
}

void Sentrience::log(const char* msg){
	print_verbose(String("[Sentrience] ") + String(msg));
}

void Sentrience::_bind_methods(){
	ClassDB::bind_method(D_METHOD("free_rid", "target"), &Sentrience::free_rid);
	ClassDB::bind_method(D_METHOD("free_all_instances"), &Sentrience::free_all_instances);
	ClassDB::bind_method(D_METHOD("get_memory_usage"), &Sentrience::get_memory_usage);
	ClassDB::bind_method(D_METHOD("get_memory_usage_humanized"), &Sentrience::get_memory_usage_humanized);
	ClassDB::bind_method(D_METHOD("get_instances_count"), &Sentrience::get_instances_count);
	ClassDB::bind_method(D_METHOD("flush_instances_pool"), &Sentrience::flush_instances_pool);
	ClassDB::bind_method(D_METHOD("set_active", "is_active"), &Sentrience::set_active);
	ClassDB::bind_method(D_METHOD("get_activation_state"), &Sentrience::get_state);

	ClassDB::bind_method(D_METHOD("recording_create"), &Sentrience::recording_create);
	ClassDB::bind_method(D_METHOD("recording_assert", "r_rec"), &Sentrience::recording_assert);
	ClassDB::bind_method(D_METHOD("recording_start", "r_rec"), &Sentrience::recording_start);
	ClassDB::bind_method(D_METHOD("recording_end", "r_rec"), &Sentrience::recording_end);
	ClassDB::bind_method(D_METHOD("recording_running", "r_rec"), &Sentrience::recording_running);

	ClassDB::bind_method(D_METHOD("simulation_create"), &Sentrience::simulation_create);
	ClassDB::bind_method(D_METHOD("simulation_assert", "r_simul"), &Sentrience::simulation_assert);
	ClassDB::bind_method(D_METHOD("simulation_set_active", "r_simul", "p_active"), &Sentrience::simulation_set_active);
	ClassDB::bind_method(D_METHOD("simulation_is_active", "r_simul"), &Sentrience::simulation_is_active);
	ClassDB::bind_method(D_METHOD("simulation_get_all_engagements", "r_simul"), &Sentrience::simulation_get_all_engagements);
	ClassDB::bind_method(D_METHOD("simulation_bind_recording", "r_simul", "r_rec"), &Sentrience::simulation_bind_recording);
	ClassDB::bind_method(D_METHOD("simulation_unbind_recording", "r_simul"), &Sentrience::simulation_unbind_recording);
	ClassDB::bind_method(D_METHOD("simulation_count_combatant", "r_simul"), &Sentrience::simulation_count_combatant);
	ClassDB::bind_method(D_METHOD("simulation_count_squad", "r_simul"), &Sentrience::simulation_count_squad);
	ClassDB::bind_method(D_METHOD("simulation_count_team", "r_simul"), &Sentrience::simulation_count_team);
	ClassDB::bind_method(D_METHOD("simulation_count_radar", "r_simul"), &Sentrience::simulation_count_radar);
	ClassDB::bind_method(D_METHOD("simulation_count_engagement", "r_simul"), &Sentrience::simulation_count_engagement);
	ClassDB::bind_method(D_METHOD("simulation_count_all_instances", "r_simul"), &Sentrience::simulation_count_all_instances);

	ClassDB::bind_method(D_METHOD("combatant_create"), &Sentrience::combatant_create);
	ClassDB::bind_method(D_METHOD("combatant_assert", "r_com"), &Sentrience::combatant_assert);
	ClassDB::bind_method(D_METHOD("combatant_get_simulation", "r_com"), &Sentrience::combatant_get_simulation);
	ClassDB::bind_method(D_METHOD("combatant_set_simulation", "r_com", "r_simul"), &Sentrience::combatant_set_simulation);
	ClassDB::bind_method(D_METHOD("combatant_is_squad", "r_com", "r_squad"), &Sentrience::combatant_is_squad);
	ClassDB::bind_method(D_METHOD("combatant_is_team", "r_com", "r_team"), &Sentrience::combatant_is_team);
	ClassDB::bind_method(D_METHOD("combatant_get_involving_engagements", "r_com"), &Sentrience::combatant_get_involving_engagements);
	ClassDB::bind_method(D_METHOD("combatant_set_local_transform", "r_com", "trans"), &Sentrience::combatant_set_local_transform);
	ClassDB::bind_method(D_METHOD("combatant_set_space_transform", "r_com", "trans"), &Sentrience::combatant_set_space_transform);
	ClassDB::bind_method(D_METHOD("combatant_get_local_transform", "r_com"), &Sentrience::combatant_get_local_transform);
	ClassDB::bind_method(D_METHOD("combatant_get_space_transform", "r_com"), &Sentrience::combatant_get_space_transform);
	ClassDB::bind_method(D_METHOD("combatant_get_combined_transform", "r_com"), &Sentrience::combatant_get_combined_transform);
	ClassDB::bind_method(D_METHOD("combatant_set_stand", "r_com", "stand"), &Sentrience::combatant_set_stand);
	ClassDB::bind_method(D_METHOD("combatant_get_stand", "r_com"), &Sentrience::combatant_get_stand);
	ClassDB::bind_method(D_METHOD("combatant_get_status", "r_com"), &Sentrience::combatant_get_status);
	ClassDB::bind_method(D_METHOD("combatant_set_iid", "r_com", "iid"), &Sentrience::combatant_set_iid);
	ClassDB::bind_method(D_METHOD("combatant_get_iid", "r_com"), &Sentrience::combatant_get_iid);
	ClassDB::bind_method(D_METHOD("combatant_set_detection_meter", "r_com", "meter"), &Sentrience::combatant_set_detection_meter);
	ClassDB::bind_method(D_METHOD("combatant_get_detection_meter", "r_com"), &Sentrience::combatant_get_detection_meter);
	ClassDB::bind_method(D_METHOD("combatant_engagable", "from", "to"), &Sentrience::combatant_engagable);
	ClassDB::bind_method(D_METHOD("combatant_bind_chip", "r_com", "chip", "auto_unbind"), &Sentrience::combatant_bind_chip);
	ClassDB::bind_method(D_METHOD("combatant_unbind_chip", "r_com"), &Sentrience::combatant_unbind_chip);
	ClassDB::bind_method(D_METHOD("combatant_set_profile", "r_com", "profile"), &Sentrience::combatant_set_profile);
	ClassDB::bind_method(D_METHOD("combatant_get_profile", "r_com"), &Sentrience::combatant_get_profile);

	ClassDB::bind_method(D_METHOD("squad_create"), &Sentrience::squad_create);
	ClassDB::bind_method(D_METHOD("squad_assert", "r_squad"), &Sentrience::squad_assert);
	ClassDB::bind_method(D_METHOD("squad_get_simulation", "r_squad"), &Sentrience::squad_get_simulation);
	ClassDB::bind_method(D_METHOD("squad_set_simulation", "r_squad", "r_simul"), &Sentrience::squad_set_simulation);
	ClassDB::bind_method(D_METHOD("squad_is_team", "r_squad", "r_team"), &Sentrience::squad_is_team);
	ClassDB::bind_method(D_METHOD("squad_get_involving_engagements", "r_squad"), &Sentrience::squad_get_involving_engagements);
	ClassDB::bind_method(D_METHOD("squad_add_combatant", "r_squad", "r_com"), &Sentrience::squad_add_combatant);
	ClassDB::bind_method(D_METHOD("squad_remove_combatant", "r_squad", "r_com"), &Sentrience::squad_remove_combatant);
	ClassDB::bind_method(D_METHOD("squad_has_combatant", "r_squad", "r_com"), &Sentrience::squad_has_combatant);
	ClassDB::bind_method(D_METHOD("squad_engagable", "from", "to"), &Sentrience::squad_engagable);
	ClassDB::bind_method(D_METHOD("squad_count_combatant", "r_squad"), &Sentrience::squad_count_combatant);
	ClassDB::bind_method(D_METHOD("squad_bind_chip", "r_squad", "chip", "auto_unbind"), &Sentrience::squad_bind_chip);
	ClassDB::bind_method(D_METHOD("squad_unbind_chip", "r_squad"), &Sentrience::squad_unbind_chip);

	ClassDB::bind_method(D_METHOD("team_create"), &Sentrience::team_create);
	ClassDB::bind_method(D_METHOD("team_assert", "r_team"), &Sentrience::team_assert);
	ClassDB::bind_method(D_METHOD("team_get_simulation", "r_team"), &Sentrience::team_get_simulation);
	ClassDB::bind_method(D_METHOD("team_set_simulation", "r_team", "r_simul"), &Sentrience::team_set_simulation);
	ClassDB::bind_method(D_METHOD("team_add_squad", "r_team", "r_squad"), &Sentrience::team_add_squad);
	ClassDB::bind_method(D_METHOD("team_remove_squad", "r_team", "r_squad"), &Sentrience::team_remove_squad);
	ClassDB::bind_method(D_METHOD("team_get_involving_engagements", "r_team"), &Sentrience::team_get_involving_engagements);
	ClassDB::bind_method(D_METHOD("team_has_squad", "r_team", "r_squad"), &Sentrience::team_has_squad);
	ClassDB::bind_method(D_METHOD("team_engagable", "from", "to"), &Sentrience::team_engagable);
	ClassDB::bind_method(D_METHOD("team_create_link", "from", "to"), &Sentrience::team_create_link);
	ClassDB::bind_method(D_METHOD("team_create_link_bilateral", "from", "to"), &Sentrience::team_create_link_bilateral);
	ClassDB::bind_method(D_METHOD("team_get_link", "from", "to"), &Sentrience::team_get_link);
	ClassDB::bind_method(D_METHOD("team_unlink", "from", "to"), &Sentrience::team_unlink);
	ClassDB::bind_method(D_METHOD("team_unlink_bilateral", "from", "to"), &Sentrience::team_unlink_bilateral);
	ClassDB::bind_method(D_METHOD("team_purge_links_multilateral", "from"), &Sentrience::team_purge_links_multilateral);
	ClassDB::bind_method(D_METHOD("team_count_squad", "r_team"), &Sentrience::team_count_squad);
	ClassDB::bind_method(D_METHOD("team_bind_chip", "r_team", "chip", "auto_unbind"), &Sentrience::team_bind_chip);
	ClassDB::bind_method(D_METHOD("team_unbind_chip", "r_team"), &Sentrience::team_unbind_chip);

	ClassDB::bind_method(D_METHOD("radar_create"), &Sentrience::radar_create);
	ClassDB::bind_method(D_METHOD("radar_assert", "r_rad"), &Sentrience::radar_assert);
	ClassDB::bind_method(D_METHOD("radar_get_simulation", "r_rad"), &Sentrience::radar_get_simulation);
	ClassDB::bind_method(D_METHOD("radar_set_simulation", "r_rad", "r_simul"), &Sentrience::radar_set_simulation);
	ClassDB::bind_method(D_METHOD("radar_set_profile", "r_rad", "profile"), &Sentrience::radar_set_profile);
	ClassDB::bind_method(D_METHOD("radar_get_profile", "r_rad"), &Sentrience::radar_get_profile);
	ClassDB::bind_method(D_METHOD("radar_request_recheck_on", "r_rad", "r_com"), &Sentrience::radar_request_recheck_on);
	ClassDB::bind_method(D_METHOD("radar_get_detected", "r_rad"), &Sentrience::radar_get_detected);
	ClassDB::bind_method(D_METHOD("radar_get_locked", "r_rad"), &Sentrience::radar_get_locked);
}

void Sentrience::poll(const float& delta){
	if (!active) return;
	for (uint32_t i = 0; i < active_simulations.size(); i++){
		auto space = active_simulations[i];
		if (space) space->poll(delta);
		// else {
		// 	active_simulations.remove(i);
		// 	i -= 1;
		// }
	}
	for (uint32_t i = 0; i < active_rec.size(); i++){
		auto rec = active_rec[i];
		if (rec) rec->poll(delta);
		// else {
		// 	active_rec.remove(i);
		// 	i -= 1;
		// }
	}
}

void Sentrience::free_all_instances(){
	log("This method has been deprecated.");
}

void Sentrience::flush_instances_pool(){
	uint32_t count = 0;
	while (!all_rids.empty()) {
		auto rid = all_rids[0];
		// log(String("Freeing RID No.") + String(std::to_string(count).c_str()) + String(" with id.") + String(std::to_string(rid.get_id()).c_str()));
		free_rid(rid);
		VEC_REMOVE(all_rids, 0);
	}
}

void Sentrience::free_rid(const RID& target){
	if (simulation_owner.owns(target)){
		FreeLog(RCSSimulation, target);
		auto stuff = simulation_owner.get(target);
		simulation_set_active(target, false);
		simulation_owner.free(target);
		rcsdel(stuff);
	} else if(combatant_owner.owns(target)){
		FreeLog(RCSCombatant, target);
		auto stuff = combatant_owner.get(target);
		combatant_owner.free(target);
		rcsdel(stuff);
	} else if(squad_owner.owns(target)){
		FreeLog(RCSSquad, target);
		auto stuff = squad_owner.get(target);
		squad_owner.free(target);
		rcsdel(stuff);
	} else if (team_owner.owns(target)){
		FreeLog(RCSTeam, target);
		auto stuff = team_owner.get(target);
		team_purge_links_multilateral(target);
		team_owner.free(target);
		rcsdel(stuff);
	} else if(radar_owner.owns(target)){
		FreeLog(RCSRadar, target);
		auto stuff = radar_owner.get(target);
		radar_owner.free(target);
		rcsdel(stuff);
	} else if(recording_owner.owns(target)){
		FreeLog(RCSRecording, target);
		auto stuff = recording_owner.get(target);
		recording_end(target);
		recording_owner.free(target);
		rcsdel(stuff);
	} 
}

RID Sentrience::recording_create(){
	auto subject = rcsnew(RCSRecording);
	RCSMakeRID(recording_owner, subject);
}
bool Sentrience::recording_assert(const RID& r_rec){
	return recording_owner.owns(r_rec);
}
bool Sentrience::recording_start(const RID& r_rec){
	auto recording = recording_owner.get(r_rec);
	ERR_FAIL_COND_V(!recording, false);
	int search_res = 0;
	VEC_FIND(active_rec, recording, search_res);
	if (search_res != -1) return false;
	OverweightAssertReturn(active_rec, false);
	active_rec.push_back(recording);
	recording->running = true;
	return true;
}
bool Sentrience::recording_end(const RID& r_rec){
	auto recording = recording_owner.get(r_rec);
	ERR_FAIL_COND_V(!recording, false);
	int search_res = 0;
	VEC_FIND(active_rec, recording, search_res);
	if (search_res != -1) return false;

	VEC_ERASE(active_rec, recording);
	recording->running = false;
	return false;
}

bool Sentrience::recording_running(const RID& r_rec){
	auto recording = recording_owner.get(r_rec);
	ERR_FAIL_COND_V(!recording, false);
	return recording->running;
}

void Sentrience::recording_purge(const RID& r_rec){
	auto recording = recording_owner.get(r_rec);
	ERR_FAIL_COND(!recording);
	recording->purge();
}

RID Sentrience::combatant_create(){
	auto subject = rcsnew(RCSCombatant);
	
	RCSMakeRID(combatant_owner, subject);
}

RID Sentrience::simulation_create(){
	auto subject = rcsnew(RCSSimulation);
	RCSMakeRID(simulation_owner, subject);
}

bool Sentrience::simulation_assert(const RID& r_simul){
	return simulation_owner.owns(r_simul);
}

void Sentrience::simulation_set_active(const RID& r_simul, const bool& p_active){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND(!simulation);
	if (simulation_is_active(r_simul) == p_active) return;

	if (p_active) {
		OverweightAssert(active_simulations);
		active_simulations.push_back(simulation);
	}
	// else active_simulations.erase(simulation);
	else VEC_ERASE(active_simulations, simulation)
}

bool Sentrience::simulation_is_active(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, false);
	VEC_HAS(active_simulations, simulation)
}
Array Sentrience::simulation_get_all_engagements(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, Array());
	return Array();
}
void Sentrience::simulation_bind_recording(const RID& r_simul, const RID& r_rec){
	auto simulation = simulation_owner.get(r_simul);
	auto recording = recording_owner.get(r_rec);
	ERR_FAIL_COND(!simulation);
	simulation->set_recorder(recording);
}
void Sentrience::simulation_unbind_recording(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND(!simulation);
	simulation->set_recorder(nullptr);
}
uint32_t Sentrience::simulation_count_combatant(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return simulation->get_combatants()->size();
}
uint32_t Sentrience::simulation_count_squad(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return simulation->get_squads()->size();
}
uint32_t Sentrience::simulation_count_team(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return simulation->get_teams()->size();
}
uint32_t Sentrience::simulation_count_radar(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return simulation->get_radars()->size();
}
uint32_t Sentrience::simulation_count_engagement(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return 0;
}
uint32_t Sentrience::simulation_count_all_instances(const RID& r_simul){
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND_V(!simulation, 0);
	return (simulation->get_combatants()->size()	+
			simulation->get_squads()->size()		+
			simulation->get_teams()->size()			+
			simulation->get_radars()->size()		 );
}

bool Sentrience::combatant_assert(const RID& r_com){
	return combatant_owner.owns(r_com);
}

RID Sentrience::combatant_get_simulation(const RID& r_com){
	GetSimulation(combatant_owner, r_com);
}

bool Sentrience::combatant_is_squad(const RID& r_com, const RID& r_squad){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, false);
	auto squad = combatant->get_squad();
	if (!squad) return false;
	return (squad->get_self() == r_squad);
}

bool Sentrience::combatant_is_team(const RID& r_com, const RID& r_team){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, false);
	auto squad = combatant->get_squad();
	if (!squad) return false;
	auto team = squad->get_team();
	if (!team) return false;
	return (team->get_self() == r_team);
}
Array Sentrience::combatant_get_involving_engagements(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	Array re;
	ERR_FAIL_COND_V(!combatant, re);
	// Manually doing this to avoid prompting errors
	auto squad = combatant->get_squad();
	if (!squad) return re;
	auto team = squad->get_team();
	if (!team) return re;
	return team->get_engagements_ref();
}
void Sentrience::combatant_set_simulation(const RID& r_com, const RID& r_simul){
	auto combatant = combatant_owner.get(r_com);
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND((!combatant || !simulation));
	combatant->set_simulation(simulation);
}

void Sentrience::combatant_set_local_transform(const RID& r_com, const Transform& trans){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_lt(trans);
}
void Sentrience::combatant_set_space_transform(const RID& r_com, const Transform& trans){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_st(trans);
}

Transform Sentrience::combatant_get_space_transform(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, Transform());
	return combatant->get_st();
}
Transform Sentrience::combatant_get_local_transform(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, Transform());
	return combatant->get_lt();
}

Transform Sentrience::combatant_get_combined_transform(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, Transform());
	return combatant->get_combined_transform();
}

void Sentrience::combatant_set_stand(const RID& r_com, const uint32_t& stand){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_stand(stand);
}
uint32_t Sentrience::combatant_get_stand(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, 0);
	return combatant->get_stand();
}
uint32_t Sentrience::combatant_get_status(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, 0);
	return combatant->get_status();
}

void Sentrience::combatant_set_iid(const RID& r_com, const uint64_t& iid){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_iid(iid);
}
uint64_t Sentrience::combatant_get_iid(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, 0);
	return combatant->get_iid();
}
void Sentrience::combatant_set_detection_meter(const RID& r_com, const double& dmeter){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->_set_detection_meter(dmeter);
}
double Sentrience::combatant_get_detection_meter(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, 0.0);
	return combatant->_get_detection_meter();
}

bool Sentrience::combatant_engagable(const RID& from, const RID& to){
	auto combatant_1 = combatant_owner.get(from);
	auto combatant_2 = combatant_owner.get(to);
	ERR_FAIL_COND_V(!combatant_1 || !combatant_2, false);
	return combatant_1->is_engagable(combatant_2);
}
void Sentrience::combatant_bind_chip(const RID& r_com, const Ref<RCSChip>& chip, const bool& auto_unbind){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	if (auto_unbind) combatant->set_chip(Ref<RCSChip>());
	ERR_FAIL_COND(combatant->get_chip().is_valid());
	combatant->set_chip(chip);
}

void Sentrience::combatant_unbind_chip(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_chip(Ref<RCSChip>());
}

void Sentrience::combatant_set_profile(const RID& r_com, const Ref<RCSCombatantProfile>& profile){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!combatant);
	combatant->set_profile(profile);
}
Ref<RCSCombatantProfile> Sentrience::combatant_get_profile(const RID& r_com){
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V(!combatant, Ref<RCSCombatantProfile>());
	return combatant->get_profile();
}

RID Sentrience::squad_create(){
	auto subject = rcsnew(RCSSquad);
	RCSMakeRID(squad_owner, subject);
}

void Sentrience::squad_set_simulation(const RID& r_squad, const RID& r_simul){
	auto squad = squad_owner.get(r_squad);
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND(!squad || !simulation);
	squad->set_simulation(simulation);
}

bool Sentrience::squad_is_team(const RID& r_squad, const RID& r_team){
	auto squad = squad_owner.get(r_squad);
	// auto team = team_owner.get(r_team);
	ERR_FAIL_COND_V(!squad, false);
	auto team = squad->get_team();
	if (!team) return false;
	return (team->get_self() == r_team);
}

Array Sentrience::squad_get_involving_engagements(const RID& r_squad){
	auto squad = squad_owner.get(r_squad);
	Array re;
	ERR_FAIL_COND_V(!squad, re);
	auto team = squad->get_team();
	if (!team) return re;
	return team->get_engagements_ref();
}

void Sentrience::squad_add_combatant(const RID& r_squad, const RID& r_com){
	auto squad = squad_owner.get(r_squad);
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND((!squad || !combatant));
	if (squad->has_combatant(combatant)) return;
	OverweightAssert(squad->combatants);
	combatant->set_squad(squad);
}
void Sentrience::squad_remove_combatant(const RID& r_squad, const RID& r_com){
	auto squad = squad_owner.get(r_squad);
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND((!squad || !combatant));
	if (!squad->has_combatant(combatant)) return;
	combatant->set_squad(nullptr);
}
bool Sentrience::squad_has_combatant(const RID& r_squad, const RID& r_com){
	auto squad = squad_owner.get(r_squad);
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND_V((!squad || !combatant), false);
	return squad->has_combatant(combatant);
}
bool Sentrience::squad_engagable(const RID& from, const RID& to){
	auto squad_1 = squad_owner.get(from);
	auto squad_2 = squad_owner.get(to);
	ERR_FAIL_COND_V((!squad_1 || !squad_2), false);
	return squad_1->is_engagable(squad_2);
}
uint32_t Sentrience::squad_count_combatant(const RID& r_squad){
	auto squad = squad_owner.get(r_squad);
	ERR_FAIL_COND_V(!squad, 0);
	return squad->get_combatants()->size();
}
void Sentrience::squad_bind_chip(const RID& r_com, const Ref<RCSChip>& chip, const bool& auto_unbind){
	auto squad = squad_owner.get(r_com);
	ERR_FAIL_COND(!squad);
	if (auto_unbind) squad->set_chip(Ref<RCSChip>());
	ERR_FAIL_COND(squad->get_chip().is_valid());
	squad->set_chip(chip);
}
void Sentrience::squad_unbind_chip(const RID& r_com){
	auto squad = squad_owner.get(r_com);
	ERR_FAIL_COND(!squad);
	squad->set_chip(Ref<RCSChip>());
}

RID Sentrience::squad_get_simulation(const RID& r_com){
	GetSimulation(squad_owner, r_com);
}

bool Sentrience::squad_assert(const RID& r_squad){
	return squad_owner.owns(r_squad);
}

RID Sentrience::team_create(){
	auto subject = rcsnew(RCSTeam);
	RCSMakeRID(team_owner, subject);
}
bool Sentrience::team_assert(const RID& r_team){
	return team_owner.owns(r_team);
}
void Sentrience::team_set_simulation(const RID& r_team, const RID& r_simul){
	auto team = team_owner.get(r_team);
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND(!team || !simulation);
	team->set_simulation(simulation);
}
RID Sentrience::team_get_simulation(const RID& r_team){
	GetSimulation(team_owner, r_team);
}
void Sentrience::team_add_squad(const RID& r_team, const RID& r_squad){
	auto team = team_owner.get(r_team);
	auto squad = squad_owner.get(r_squad);
	ERR_FAIL_COND(!team || !squad);
	OverweightAssert(team->squads);
	if (!team->has_squad(squad))
		squad->set_team(team);
}
void Sentrience::team_remove_squad(const RID& r_team, const RID& r_squad){
	auto team = team_owner.get(r_team);
	auto squad = squad_owner.get(r_squad);
	ERR_FAIL_COND(!team || !squad);
	if (team->has_squad(squad))
		squad->set_team(nullptr);
}
Array Sentrience::team_get_involving_engagements(const RID& r_team){
	auto team = team_owner.get(r_team);
	ERR_FAIL_COND_V(!team, Array());
	return team->get_engagements_ref();
}
bool Sentrience::team_has_squad(const RID& r_team, const RID& r_squad){
	auto team = team_owner.get(r_team);
	auto squad = squad_owner.get(r_squad);
	ERR_FAIL_COND_V(!team || !squad, false);
	return team->has_squad(squad);
}
bool Sentrience::team_engagable(const RID& from, const RID& to){
	auto team_from	= team_owner.get(from);
	auto team_to	= team_owner.get(to);
	ERR_FAIL_COND_V(!team_from || !team_to, false);
	return team_from->is_engagable(team_to);
}
Ref<RCSUnilateralTeamsBind> Sentrience::team_create_link(const RID& from, const RID& to){
	auto team_from	= team_owner.get(from);
	auto team_to	= team_owner.get(to);
	ERR_FAIL_COND_V(!team_from || !team_to, Ref<RCSUnilateralTeamsBind>());
	auto preallocated_link = team_from->get_link_to(team_to);
	if (preallocated_link.is_valid()) return preallocated_link;
	OverweightAssertReturn(team_from->team_binds, preallocated_link);
	return team_from->add_link(team_to);
}
void Sentrience::team_create_link_bilateral(const RID& from, const RID& to){
	team_create_link(from, to);
	team_create_link(to, from);
}
Ref<RCSUnilateralTeamsBind> Sentrience::team_get_link(const RID& from, const RID& to){
	auto team_from	= team_owner.get(from);
	auto team_to	= team_owner.get(to);
	ERR_FAIL_COND_V(!team_from || !team_to, Ref<RCSUnilateralTeamsBind>());
	return team_from->get_link_to(team_to);
}
bool Sentrience::team_has_link(const RID& from, const RID& to){
	return team_get_link(from, to).is_valid();
}
bool Sentrience::team_unlink(const RID& from, const RID& to){
	auto team_from	= team_owner.get(from);
	auto team_to	= team_owner.get(to);
	ERR_FAIL_COND_V(!team_from || !team_to, false);
	return team_from->remove_link(team_to);
}
bool Sentrience::team_unlink_bilateral(const RID& from, const RID& to){
	return (team_unlink(from, to) && team_unlink(to, from));
}

void Sentrience::team_purge_links_multilateral(const RID& from){
	auto team_from	= team_owner.get(from);
	ERR_FAIL_COND(!team_from);
	team_from->purge_all_links();
}

uint32_t Sentrience::team_count_squad(const RID& r_team){
	auto team = team_owner.get(r_team);
	ERR_FAIL_COND_V(!team, 0);
	return team->get_squads()->size();
}

void Sentrience::team_bind_chip(const RID& r_team, const Ref<RCSChip>& chip, const bool& auto_unbind){
	auto team = team_owner.get(r_team);
	ERR_FAIL_COND(!team);
	if (auto_unbind) team->set_chip(Ref<RCSChip>());
	ERR_FAIL_COND(team->get_chip().is_valid());
	team->set_chip(chip);
}
void Sentrience::team_unbind_chip(const RID& r_team){
	auto team = team_owner.get(r_team);
	ERR_FAIL_COND(!team);
	team->set_chip(Ref<RCSChip>());
}

RID Sentrience::radar_create(){
	auto subject = rcsnew(RCSRadar);
	RCSMakeRID(radar_owner, subject);
}
bool Sentrience::radar_assert(const RID& r_rad){
	return radar_owner.owns(r_rad);
}
void Sentrience::radar_set_simulation(const RID& r_rad, const RID& r_simul){
	auto radar = radar_owner.get(r_rad);
	auto simulation = simulation_owner.get(r_simul);
	ERR_FAIL_COND(!radar || !simulation);
	radar->set_simulation(simulation);
}
RID Sentrience::radar_get_simulation(const RID& r_rad){
	GetSimulation(radar_owner, r_rad);
}
void Sentrience::radar_set_profile(const RID& r_rad, const Ref<RCSRadarProfile>& profile){
	auto radar = radar_owner.get(r_rad);
	ERR_FAIL_COND(!radar);
	radar->set_profile(profile);
}
Ref<RCSRadarProfile> Sentrience::radar_get_profile(const RID& r_rad){
	auto radar = radar_owner.get(r_rad);
	ERR_FAIL_COND_V(!radar, Ref<RCSRadarProfile>());
	return radar->get_profile();
}
void Sentrience::radar_request_recheck_on(const RID& r_rad, const RID& r_com){
	auto radar = radar_owner.get(r_rad);
	auto combatant = combatant_owner.get(r_com);
	ERR_FAIL_COND(!radar || !combatant);
	auto simulation = radar->simulation;
	ERR_FAIL_COND(!simulation);
	simulation->radar_request_recheck(new RadarRecheckTicket(radar, combatant));
}
Array Sentrience::radar_get_detected(const RID& r_rad){
	auto radar = radar_owner.get(r_rad);
	ERR_FAIL_COND_V(!radar, Array());
	return radar->get_detected();
}
Array Sentrience::radar_get_locked(const RID& r_rad){
	auto radar = radar_owner.get(r_rad);
	ERR_FAIL_COND_V(!radar, Array());
	return radar->get_locked();
}
