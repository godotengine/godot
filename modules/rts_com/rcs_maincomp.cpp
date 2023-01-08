#include "combat_server.h"

#pragma region Macros

#define MainPoll(what, iter, delta)                   \
	auto target = what[iter];                         \
	if (target) target->poll(delta)

#define RemoveReference(iter, inc, func, pool)        \
	if (iter < pool.size()){                          \
		auto ref = pool[iter];                        \
		if (ref) {                                    \
			ref->func(nullptr);                       \
		}                                             \
		inc++;                                        \
	}
#define BilateralCleanup(vec, opposite)               \
	while (!vec.empty()){                             \
		auto ref = vec[0];                            \
		ref->opposite = nullptr;                      \
		VEC_ERASE(vec, 0)                             \
	}
#define SimulationRecord(recorder, event)             \
	event->simulation = this;                         \
	if (recorder){                                    \
		recorder->push_event(event);                  \
	} else ((void)0)

#if defined(DEBUG_ENABLED) && defined(RCS_MAINCOMP_DEBUG_LOG)
#define CleanerLog(classptr, i) \
print_verbose(String(#classptr) + String(" id.") + String(std::to_string(get_self().get_id()).c_str()) + String(" cleaning up at iter: ") + String(std::to_string(i).c_str()))
#else
#define CleanerLog(classptr, i)
#endif

#ifdef DEBUG_ENABLED
#define ThiccCheck(vec) ((void)0)
	// if (true) ((void)0); else if (vec.size() > 1000) { print_verbose(String("[" FUNCTION_STR "] She\'s a thicc one...: " + itos(vec.size()))); ERR_FAIL(); }

#define FattCheck(vec) ((void)0)
	// if (true) ((void)0); else if (vec.size() > 1000) { print_verbose(String("[" FUNCTION_STR "] She\'s a fatt one...: " + itos(vec.size()))); ERR_FAIL(); }

#define SimAdditionLog(data, what) \
	print_verbose(String("[RCSSimulation:") + itos(get_self().get_id()) + String("] Adding " #what " with id ") + itos(data->get_self().get_id()))

#define SimErasureLog(data, what) \
	print_verbose(String("[RCSSimulation:") + itos(get_self().get_id()) + String("] Removing " #what " with id ") + itos(data->get_self().get_id()))
#else
#define ThiccCheck(vec) ((void)0)
#define FattCheck(vec) ((void)0)
#define SimAdditionLog(data, what) ((void)0)
#define SimErasureLog(data, what) ((void)0)
#endif

VARIANT_ENUM_CAST(RCSEngagement::EngagementScale);
VARIANT_ENUM_CAST(RCSCombatantProfile::CombatantAttribute);
VARIANT_ENUM_CAST(RCSCombatantProfile::CombatantStand);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanMode);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanBase);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanAttributes);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarTargetMode);
VARIANT_ENUM_CAST(RCSUnilateralTeamsBind::TeamRelationship);
VARIANT_ENUM_CAST(RCSUnilateralTeamsBind::InterTeamAttribute);

RCSMemoryAllocation* RCSMemoryAllocation::tracker_ptr = nullptr;

RCSMemoryAllocation::RCSMemoryAllocation(){
	ERR_FAIL_COND(tracker_ptr);
	tracker_ptr = this;
}
RCSMemoryAllocation::~RCSMemoryAllocation(){
	tracker_ptr = nullptr;
}

#pragma endregion

#pragma region EventReport
CombatantEventReport::CombatantEventReport(){}
SquadEventReport::SquadEventReport(){}
TeamEventReport::TeamEventReport(){}
ProjectileEventReport::ProjectileEventReport(){}
RadarEventReport::RadarEventReport(){}
EngagementEventReport::EngagementEventReport(){}

void ProjectileEventReport::set_package_alpha(RCSCombatant *com) {
	package_a = com; pcka_id = com->get_combatant_id();
}
void ProjectileEventReport::set_package_beta(RCSCombatant *com)  {
	package_b = com; pckb_id = com->get_combatant_id();
}

Dictionary ProjectileEventReport::primitive_describe() const {
	Dictionary re;
	re["event_name"] = "ProjectileEventReport";
	auto event_type_uint32_t = (uint32_t)event_type;
	auto pa = !package_a ? RID_TYPE() : package_a->get_self();
	auto pb = !package_b ? RID_TYPE() : package_b->get_self();
	auto id_a = pcka_id;
	auto id_b = pckb_id;
	// auto emitter = !sender ? RID_TYPE() : sender->get_self();
	EventRecord(re, event_type_uint32_t);
	EventRecord(re, pa);
	EventRecord(re, pb);
	EventRecord(re, id_a);
	EventRecord(re, id_b);
	return re;
}
Dictionary RadarEventReport::primitive_describe() const {
	Dictionary re;
	re["event_name"] = "RadarEventReport";
	auto event_type_uint32_t = (uint32_t)event_type;
	auto emitter = !sender ? RID_TYPE() : sender->get_self();
	Variant newly_detected = conclusion->newly_detected;
	Variant newly_locked = conclusion->newly_locked;
	Variant nolonger_detected = conclusion->nolonger_detected;
	Variant nolonger_locked = conclusion->nolonger_locked;
	EventRecord(re, event_type_uint32_t);
	EventRecord(re, emitter);
	EventRecord(re, newly_detected);
	EventRecord(re, newly_locked);
	EventRecord(re, nolonger_detected);
	EventRecord(re, nolonger_locked);
	return re;
}

#pragma endregion

#pragma region World

RCSSingleWorld::RCSSingleWorld(){
	inner_world = rcsnew(UnitedWorld);
}
RCSSingleWorld::~RCSSingleWorld(){
	rcsdel(inner_world);
}
RCSStaticWorld::RCSStaticWorld(){
	static_world = rcsnew(StaticWorldPartition);
	// synchronizer = rcsnew(CellsSynchronizer);
}
RCSStaticWorld::~RCSStaticWorld(){
	rcsdel(static_world);
	// rcsdel(synchronizer);
}
RCSStaticWorld::StaticWorldPartition::StaticWorldPartition(){

}
RCSStaticWorld::StaticWorldPartition::~StaticWorldPartition(){
	deallocate_world();
}
void RCSStaticWorld::StaticWorldPartition::allocate_by_static_size(const double& world_width, const uint32_t& cell_per_row) {
	deallocate_world();
	//-----------------------------------------------------------
	this->world_width = world_width;
	this->cpr = cell_per_row;
	this->cell_width = world_width / cell_per_row;
	this->cell_count = cell_per_row * cell_per_row;
	//-----------------------------------------------------------
	cells_2d_array.resize(this->cell_count);
}
void RCSStaticWorld::StaticWorldPartition::allocate_by_dynamic_size(const double& cell_width, const uint32_t& cell_per_row) {
	deallocate_world();
	//-----------------------------------------------------------
	this->world_width = cell_width * cell_per_row;
	this->cpr = cell_per_row;
	this->cell_width = cell_width;
	this->cell_count = cell_per_row * cell_per_row;
	//-----------------------------------------------------------
	cells_2d_array.resize(this->cell_count);
}

#pragma endregion

#pragma region Recording
RCSRecording::RCSRecording(){
	// start_time_usec = OS::get_singleton()->get_ticks_usec();
	start_time_usec = 0;
}
RCSRecording::~RCSRecording(){
	// purge();
}

void RCSRecording::purge(){
	// while (!reports_holder.empty()){
	// 	auto report = reports_holder.operator[](0);
	// 	memdelete(report);
	// 	VEC_REMOVE(reports_holder, 0);
	// }
	reports_holder.clear();
}

void RCSRecording::push_event(const std::shared_ptr<EventReport>& event){
	if (!running) return;
	auto timestamp = get_timestamp();
	// auto ticket = memnew(EventReportTicket(timestamp, event));
	// auto ticket = std::make_unique<EventReportTicket>(timestamp, event);
	reports_holder.push_back(std::make_shared<EventReportTicket>(timestamp, event));
}

VECTOR<std::weak_ptr<EventReportTicket>> RCSRecording::events_by_simulation(const RCSSimulation* simulation) const {
	VECTOR<std::weak_ptr<EventReportTicket>> re;
	auto size = reports_holder.size();
	// Resize the return-array to the max-possible size for fater writting
	re.resize(size);
	uint32_t usable_size = 0;
	// auto mut_sim = const_cast<RCSSimulation*>(simulation);
	for (uint32_t i = 0; i < size; i++){
		auto event = reports_holder[i];
		if (event->event->get_simulation() == simulation){
			// re[usable_size].swap(std::weak_ptr<EventReportTicket>(event));
			// std::weak_ptr<EventReportTicket>& ptr = re[usable_size];
			// ptr.swap(event);
			re.write[usable_size] = std::weak_ptr<EventReportTicket>(event);
			usable_size += 1;
		}
	}
	// Resize the return array to the usable size
	re.resize(usable_size);
	return re;
}
Dictionary RCSRecording::events_by_simulation_compat(const RCSSimulation* simulation) const{
	auto events = events_by_simulation(simulation);
	Dictionary re;
	for (uint32_t i = 0, s = events.size(); i < s; i++) {
		auto ticket = events[i].lock();
		if (!ticket) continue;
		auto timestamp = ticket->timestamp;
		re[timestamp] = ticket->event->primitive_describe();
	}
	return re;
}

VECTOR<std::weak_ptr<EventReportTicket>> RCSRecording::get_all_events() const {
	VECTOR<std::weak_ptr<EventReportTicket>> re;
	auto size = reports_holder.size();
	re.resize(size);
	for (uint32_t i = 0; i < size; i++){
		auto event = reports_holder[i];
		// re[i].swap(std::weak_ptr<EventReportTicket>(event));
		// std::weak_ptr<EventReportTicket>& ptr = re[i];
		// ptr.swap(event);
		re.set(i, std::weak_ptr<EventReportTicket>(event));
	}
	return re;
}

void RCSRecording::poll(const float& delta){
	RID_RCS::poll(delta);
}
#pragma endregion

#pragma region Simulation

void RCSSimulationProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_edt", "edt"), &RCSSimulationProfile::set_edt);
	ClassDB::bind_method(D_METHOD("get_edt"), &RCSSimulationProfile::get_edt);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "engagement_deactivation_time", PROPERTY_HINT_RANGE, "5,600,0.1"), "set_edt", "get_edt");
}

RCSSimulation::RCSSimulation() {
	profile = Ref<RCSSimulationProfile>(memnew(RCSSimulationProfile));
	recorder = nullptr;
}

RCSSimulation::~RCSSimulation(){
	while (true){
		uint8_t cond = 0;
		if (!combatants.empty()) {
			combatants[0]->set_simulation(nullptr);
			cond += 1;
		}
		if (!squads.empty()) {
			squads[0]->set_simulation(nullptr);
			cond += 1;
		}
		if (!teams.empty()) {
			teams[0]->set_simulation(nullptr);
			cond += 1;
		}
		if (!radars.empty()) {
			radars[0]->set_simulation(nullptr);
			cond += 1;
		}
		if (cond == 0) return;
	}
	// while (!combatants.empty()){
	// 	combatants[0]->set_simulation(nullptr);
	// 	// combatants[0]->simulation = nullptr;
	// 	// VEC_REMOVE(combatants, 0);
	// }
	// while (!squads.empty()){
	// 	squads[0]->set_simulation(nullptr);
	// 	// squads[0]->simulation = nullptr;
	// 	// VEC_REMOVE(squads, 0);
	// }
	// while (!teams.empty()){
	// 	teams[0]->set_simulation(nullptr);
	// 	// teams[0]->simulation = nullptr;
	// 	// VEC_REMOVE(teams, 0);
	// }
	// while (!radars.empty()){
	// 	radars[0]->set_simulation(nullptr);
	// 	// radars[0]->simulation = nullptr;
	// 	// VEC_REMOVE(radars, 0);
	// }
}

Vector<RCSSquad*> RCSSimulation::request_scanable_squads(const RCSSquad* base) const{
	ERR_FAIL_COND_V(!base, Vector<RCSSquad*>());
	Vector<RCSSquad*> re;
	for (uint32_t i = 0, size = squads.size(); i < size; i++){
		auto squad = squads.get(i);
		if (base->is_scannable(squad)) re.push_back(squad);
	}
	return re;
}

Vector<RCSSquad*> RCSSimulation::request_reachable_squads(const RCSSquad* from) const{
	// For now only do this
	// ERR_FAIL_COND_V(!from, Vector<RCSSquad*>());
#if defined(USE_STL_VECTOR)
	Vector<RCSSquad*> re;
	for (uint32_t i = 0, size = squads.size(); i < size; i++){
		re.push_back(squads.get(i));
	}
	return re;
#else
	return squads;
#endif
}

Vector<RCSCombatant*> RCSSimulation::request_reachable_combatants(const RCSSquad* from) const{
// ERR_FAIL_COND_V(!from, Vector<RCSCombatant*>());
#if defined(USE_STL_VECTOR)
	Vector<RCSSquad*> re;
	for (uint32_t i = 0, size = combatants.size(); i < size; i++){
		re.push_back(combatants.get(i));
	}
	return re;
#else
	return combatants;
#endif
}

std::weak_ptr<RCSEngagementInternal> RCSSimulation::request_active_engagement(const RID_TYPE& offending_squad, const RID_TYPE& defending_squad) const{
	for (uint32_t i = 0, size = engagements.size(); i < size; i++){
		auto en = engagements[i];
		if (en->is_engagement_over()) continue;
		// auto o_squads = en->_get_offending_squads();
		// auto d_squads = en->get_deffending_squad();
		// if (VectorHas<RID_TYPE>(en->offending_squads, offending_squad) && VectorHas<RID_TYPE>(en->deffending_squads, defending_squad)){
		// 	return en;
		// }
		if (en->get_offending_squad() == offending_squad && en->get_deffending_squad() == defending_squad){
			return en;
		}
	}
	return std::weak_ptr<RCSEngagementInternal>();
}

VECTOR<std::weak_ptr<RCSEngagementInternal>> RCSSimulation::request_engagements_list(const RID_TYPE& offending_squad, const RID_TYPE& defending_squad) const{
	VECTOR<std::weak_ptr<RCSEngagementInternal>> re;
	for (uint32_t i = 0, size = engagements.size(); i < size; i++){
		auto en = engagements[i];
		// auto o_squads = en->_get_offending_squads();
		// auto d_squads = en->get_deffending_squad();
		// if (VectorHas<RID_TYPE>(en->offending_squads, offending_squad) && VectorHas<RID_TYPE>(en->deffending_squads, defending_squad)){
		// 	re.push_back(std::weak_ptr<RCSEngagementInternal>(en));
		// }
		if (en->get_offending_squad() == offending_squad && en->get_deffending_squad() == defending_squad){
			re.push_back(std::weak_ptr<RCSEngagementInternal>(en));
		}
	}
	return re;
}
Vector<std::weak_ptr<RCSEngagementInternal>> RCSSimulation::request_all_active_engagements() const {
	Vector<std::weak_ptr<RCSEngagementInternal>> re;
	for (uint32_t i = 0, s = engagements.size(); i < s; i++){
		const auto& en = engagements[i];
		if (!en->engaging) continue;
		re.push_back(en);
	}
	return re;
}
Array RCSSimulation::request_all_active_engagements_compat() const {
	Array re;
	for (uint32_t i = 0, s = engagements.size(); i < s; i++){
		const auto& en = engagements[i];
		if (!en->engaging) continue;
		re.push_back(en->spawn_reference());
	}
	return re;
}
Vector<std::weak_ptr<RCSEngagementInternal>> RCSSimulation::request_all_engagements() const {
	Vector<std::weak_ptr<RCSEngagementInternal>> re;
	re.resize(engagements.size());
	for (uint32_t i = 0, s = engagements.size(); i < s; i++){
		re.write[i] = engagements[i];
	}
	return re;
}
Array RCSSimulation::request_all_engagements_compat() const {
	Array re;
	re.resize(engagements.size());
	for (uint32_t i = 0, s = engagements.size(); i < s; i++){
		re[i] = engagements[i]->spawn_reference();
	}
	return re;
}

void RCSSimulation::add_combatant(RCSCombatant* com){
	FattCheck(combatants);
	SimAdditionLog(com, RCSCombatant);
	combatants.push_back(com);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::CombatantAdded;
		event->event_subject = com->get_self();
		simulation_event(event);
	}
}
void RCSSimulation::add_squad(RCSSquad* squad){
	FattCheck(squads);
	SimAdditionLog(squad, RCSSquad);
	squads.push_back(squad);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::SquadAdded;
		event->event_subject = squad->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::add_team(RCSTeam* team){
	FattCheck(teams);
	SimAdditionLog(team, RCSTeam);
	teams.push_back(team);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::TeamAdded;
		event->event_subject = team->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::add_radar(RCSRadar* rad){
	FattCheck(radars);
	SimAdditionLog(rad, RCSRadar);
	radars.push_back(rad);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::RadarAdded;
		event->event_subject = rad->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::remove_combatant(RCSCombatant* com)
{
	// print_verbose(String("Removing Combatant..."));
	ThiccCheck(combatants);
	SimErasureLog(com, RCSCombatant);
	VEC_ERASE(combatants, com)
	// for (uint32_t i = 0, rsize = radars.size(); i < rsize; i++){
	// 	auto radar = radars[i];
	// 	// if (radar) radar->dangling_pointer_cleanup(com->get_self());
	// }
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::CombatantRemoved;
		event->event_subject = com->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::remove_squad(RCSSquad* squad)
{
	ThiccCheck(squads);
	auto idx = squads.find(squad);
	if (idx == -1) return;
	SimErasureLog(squad, RCSSquad);
	while (!squad->participating.empty()){
		auto engagement = squad->participating[0].lock();
		engagement->remove_side(squad);
	}
	squads.remove(idx);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::SquadRemoved;
		event->event_subject = squad->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::remove_team(RCSTeam* team)
{
	ThiccCheck(teams);
	SimErasureLog(team, RCSTeam);
	VEC_ERASE(teams, team)
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::TeamRemoved;
		event->event_subject = team->get_self();
		simulation_event(event);
	}
}

static _FORCE_INLINE_ void remove_radar_from_engagement(const RID_TYPE& radar, const VECTOR<std::weak_ptr<RCSEngagementInternal>>& engagements){
	for (uint32_t i = 0, s = engagements.size(); i < s; i++){
		auto engagement = engagements[i].lock();
		auto& scouting = engagement->get_active_radars();
		auto scout_idx = scouting.find(radar);
		if (scout_idx) scouting.erase(scout_idx);
	}
}

void RCSSimulation::remove_radar(RCSRadar* rad){
	auto idx = radars.find(rad);
	if (idx == -1) return;
	SimErasureLog(rad, RCSRadar);
	// ------------------------------------------------------------------
	//                       Erase all references
	//
	// In theory, all locked vessels are also detected,
	// so we should only scan once
	//
	Vector<RID_TYPE> processed_squads;
	for (auto E = rad->detected.front(); E; E = E->next()){
		auto const_squad = (request_squad_from_rid(E->get()));
		RCSSquad* processing_squad = nullptr;
		if (!const_squad) {
			// Possible bottleneck
			auto const_com = request_combatant_from_rid(E->get());
			if (const_com){
				processing_squad = (const_cast<RCSCombatant*>(const_com))->get_squad();
				if (processed_squads.find(processing_squad->get_self()) != -1)
					processing_squad = nullptr;
			}
		} else processing_squad = const_cast<RCSSquad*>(const_squad);
		if (!processing_squad) continue;
		const auto& engagements = processing_squad->participating;
		remove_radar_from_engagement(rad->get_self(), engagements);
		processed_squads.push_back(processing_squad->get_self());
	}
	// ------------------------------------------------------------------
	radars.remove(idx);
	if (recorder) {
		auto event = std::make_shared<SimulationEventReport>();
		event->event_type = SimulationEventReport::RadarRemoved;
		event->event_subject = rad->get_self();
		simulation_event(event);
	}
}

void RCSSimulation::set_recorder(RCSRecording* rec){
	// No need for conclusion i guess?
	recorder = rec;
}

const RCSCombatant* RCSSimulation::request_combatant_from_rid(const RID_TYPE& rid) const {
	const RCSCombatant *re = nullptr;
	SimRecordSearch(rid, combatants, re);
	return re;
}
const RCSSquad* RCSSimulation::request_squad_from_rid(const RID_TYPE& rid) const {
	const RCSSquad *re = nullptr;
	SimRecordSearch(rid, squads, re);
	return re;
}
const RCSTeam* RCSSimulation::request_team_from_rid(const RID_TYPE& rid) const {
	const RCSTeam *re = nullptr;
	SimRecordSearch(rid, teams, re);
	return re;
}
const RCSRadar* RCSSimulation::request_radar_from_rid(const RID_TYPE& rid) const {
	const RCSRadar *re = nullptr;
	SimRecordSearch(rid, radars, re);
	return re;
}

void RCSSimulation::ihandler_projectile_fired(std::shared_ptr<ProjectileEventReport>& event){
	auto o_com = event->get_sender()->get_host()->get_fired_by();
	ERR_FAIL_COND_MSG(!o_com, "No projectile\'s host was set");
	auto d_com = event->get_package_beta();
	auto old_target = event->get_package_alpha();
	auto o_squad = o_com->get_squad();
	// Target reset, do not proceed further
	if (!d_com) return;
	auto d_squad = d_com->get_squad();
	ERR_FAIL_COND_MSG(!o_squad || !d_squad, "Either combatants have not been assigned to a squad");
	auto active_engagement = request_active_engagement(o_squad->get_self(), d_squad->get_self()).lock();
	ERR_FAIL_COND_MSG(!active_engagement, "Engagement has yet to be initialized. This should not have happened");
	active_engagement->reset_action_timer();
	// auto curr_scale = active_engagement->get_scale();
	// if (curr_scale < RCSEngagement::Skirmish) curr_scale = RCSEngagement::Skirmish;
	active_engagement->total_heat += 1.0;
}
void RCSSimulation::ihandler_radar_scan_concluded(std::shared_ptr<RadarEventReport>& event){
	auto radar = event->sender;
	auto com = radar->assigned_vessel;
	auto squad = com->get_squad();
	auto newly_detected_squad = event->conclusion->newly_detected;
	auto nolonger_detected_squad = event->conclusion->nolonger_detected;
	switch (event->target_mode){
		case RCSRadarProfile::TargetCombatants: {
			Vector<RID_TYPE> added_newly_detected_squad;
			Vector<RID_TYPE> added_nolonger_detected_squad;
			newly_detected_squad = Vector<RID_TYPE>();
			nolonger_detected_squad = Vector<RID_TYPE>();
			for (uint32_t i = 0, s = event->conclusion->newly_detected.size(); i < s; i++){
				const auto& curr_rid = event->conclusion->newly_detected[i];
				auto curr_com_const = request_combatant_from_rid(curr_rid);
				if (!curr_com_const) continue;
				auto curr_com = const_cast<RCSCombatant*>(curr_com_const);
				auto curr_squad = curr_com->get_squad();
				if (!curr_squad) continue;
				auto squad_rid = curr_squad->get_self();
				if (added_newly_detected_squad.find(squad_rid) != -1) continue;
				nolonger_detected_squad.push_back(squad_rid);
				added_newly_detected_squad.push_back(squad_rid);
			}
			for (uint32_t i = 0, s = event->conclusion->nolonger_detected.size(); i < s; i++){
				const auto& curr_rid = event->conclusion->nolonger_detected[i];
				auto curr_com_const = request_combatant_from_rid(curr_rid);
				if (!curr_com_const) continue;
				auto curr_com = const_cast<RCSCombatant*>(curr_com_const);
				auto curr_squad = curr_com->get_squad();
				if (!curr_squad) continue;
				auto squad_rid = curr_squad->get_self();
				if (added_nolonger_detected_squad.find(squad_rid) != -1) continue;
				nolonger_detected_squad.push_back(squad_rid);
				added_nolonger_detected_squad.push_back(squad_rid);
			}
		}
		case RCSRadarProfile::TargetSquadPartial: {
			for (uint32_t i = 0, s = newly_detected_squad.size(); i < s; i++){
				const auto& curr_rid = newly_detected_squad[i];
				auto engagement_idx = request_active_engagement(squad->get_self(), curr_rid);
				auto engagement = engagement_idx.lock();
				if (engagement){
					engagement->reset_action_timer();
					engagement->get_active_radars().push_back(radar->get_self());
				} else {
					auto event = std::make_shared<EngagementEventReport>();
					event->event_type = EngagementEventReport::EngagementStarted;
					event->deffender = curr_rid;
					event->offender = squad->get_self();
					engagement_event(event);
				}
			}
			for (uint32_t i = 0, s = nolonger_detected_squad.size(); i < s; i++){
				const auto& curr_rid = newly_detected_squad[i];
				auto engagement_idx = request_active_engagement(squad->get_self(), curr_rid);
				auto engagement = engagement_idx.lock();
				if (!engagement) continue;
				engagement->get_active_radars().erase(radar->get_self());
			}
			break;
		}
	}
}
void RCSSimulation::simulation_event(std::shared_ptr<SimulationEventReport>& event){
	SimulationRecord(recorder, event);
	// auto ev = event.operator->();
	// ev->simulation = this;
	// if (recorder) recorder->push_event(event);
}
void RCSSimulation::combatant_event(std::shared_ptr<CombatantEventReport>& event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::squad_event(std::shared_ptr<SquadEventReport>& event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::team_event(std::shared_ptr<TeamEventReport>& event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::projectile_event(std::shared_ptr<ProjectileEventReport>& event){
	SimulationRecord(recorder, event);
	switch (event->get_event()){
		  case ProjectileEventReport::Activation: {
			if (!event->get_package_beta()) ihandler_projectile_fired(event);
		} case ProjectileEventReport::Deactivation: {

		} case ProjectileEventReport::Marked: {

		} case ProjectileEventReport::Unmarked: {

		} case ProjectileEventReport::HostSet: {

		} case ProjectileEventReport::TargetSet: {
			if (event->get_sender()->get_state()) ihandler_projectile_fired(event);
		} default: return;
	}
}
void RCSSimulation::engagement_event(std::shared_ptr<EngagementEventReport>& event){
	// Verify that the squad actually exist
	auto my_squad_const = request_squad_from_rid(event->offender);
	auto target_squad_const = request_squad_from_rid(event->deffender);
	ERR_FAIL_COND(!my_squad_const || !target_squad_const);
	auto my_squad = const_cast<RCSSquad*>(my_squad_const);
	auto target_squad = const_cast<RCSSquad*>(target_squad_const);
	switch (event->event_type){
		case EngagementEventReport::EngagementStarted: {
			auto new_engagement = create_engagement().lock();
			new_engagement->offending_squads = event->offender;
			new_engagement->deffending_squads = event->deffender;
			event->opaque_engagement = new_engagement;
			// Add references
			my_squad->add_participating(new_engagement);
			target_squad->add_participating(new_engagement);
			// auto o_team = my_squad->get_team();
			// if (o_team) {
			// 	new_engagement->offending_team = o_team->get_self();
			// 	// o_team->add_participating(new_engagement);
			// }
			// auto d_team = target_squad->get_team();
			// if (d_team) {
			// 	new_engagement->deffending_team = d_team->get_self();
			// 	d_team->add_participating(new_engagement);
			// }
			new_engagement->degrade_to_finished = profile->get_edt();
			new_engagement->scale = RCSEngagement::Stalk;
			event->scale = new_engagement->scale;
			break;
		}
		case EngagementEventReport::EngagementFinished: {
			auto actual_engagement = event->opaque_engagement.lock();
			actual_engagement->engaging = false;
			actual_engagement->engagement_time = actual_engagement->time_elapsed;
			event->scale = actual_engagement->scale;
			event->deffender = actual_engagement->deffending_squads;
			event->offender = actual_engagement->offending_squads;
			event->winner = actual_engagement->winner;
			my_squad->remove_participating(actual_engagement);
			target_squad->remove_participating(actual_engagement);
			break;
		} case EngagementEventReport::EngagementScaleChanged: {

		}
		default: return;
	}
	SimulationRecord(recorder, event);
}
void RCSSimulation::radar_event(std::shared_ptr<RadarEventReport>& event){
	SimulationRecord(recorder, event);
	switch (event->event_type){
		case RadarEventReport::ScanConcluded: {
			auto com = event->sender->assigned_vessel;
			if (com && com->get_squad()) ihandler_radar_scan_concluded(event);
		} default: return;
	}
}

RCSCombatant* RCSSimulation::request_combatant_from_iid(const uint64_t& iid) const{
	auto csize = combatants.size();
	for (uint32_t i = 0; i < csize; i++){
		auto combatant = combatants[i];
		if (!combatant) continue;
		if (combatant->get_iid() == iid) return combatant;
	}
	return nullptr;
}
void RCSSimulation::poll(const float &delta) {
	RID_RCS::poll(delta);
	// Radars run first for rechecks
	auto radars_count = radars.size();
	for (uint32_t k = 0; k < radars_count; k++) {
		MainPoll(radars, k, delta);
	}
	// Radar recheck requests
	for (uint32_t u = 0; u < rrecheck_requests.size(); u++) {
		auto ticket = rrecheck_requests[u];
		if (!ticket->request_sender)
			continue;
		ticket->request_sender->late_check(delta, ticket);
		memdelete(ticket);
	}
	rrecheck_requests.clear();
	// Radar conclude requests
	for (uint32_t q = 0; q < rconclude_requests.size(); q++){
		auto ticket = rconclude_requests[q];
		if (!ticket->request_sender) continue;
		ticket->request_sender->conclude(delta);
		memdelete(ticket);
	}
	rconclude_requests.clear();
	// auto engagements_count = engagements.size();
	for (uint32_t x = 0; x < engagements.size(); x++) {
		auto en = engagements[x];
		en->poll(delta);
#ifdef ENGAGEMENT_AUTO_CLEAN
		if (en->is_engagement_over()) {
			VEC_REMOVE(engagements, x);
			x--;
			// engagements_count--;
		}
#endif
	}
	for (uint32_t j = 0; j < combatants.size(); j++) {
		MainPoll(combatants, j, delta);
	}
	for (uint32_t i = 0; i < squads.size(); i++) {
		MainPoll(squads, i, delta);
	}
}


#pragma endregion

#pragma region Engagement

RCSEngagementInternal::RCSEngagementInternal(){
	// offending_team = nullptr;
	engaging = false;
	scale = RCSEngagement::NA;
}

RCSEngagementInternal::~RCSEngagementInternal(){
	flush_all_references();
	// cut_ties_to_all();
}

void RCSEngagementInternal::poll(const float& delta){
	// RID_RCS::poll(delta);
	time_since_last_action += delta;
	time_elapsed += delta;
	if (engaging && time_since_last_action > degrade_to_finished){
		auto new_event = std::make_shared<EngagementEventReport>();
		new_event->event_type = EngagementEventReport::EngagementFinished;
		new_event->opaque_engagement = self_ref;
		sim->engagement_event(new_event);
	}
	// Decide engagement scale base on heat meter
	auto current_heat = get_heat_meter();
}

Ref<RCSEngagement> RCSEngagementInternal::spawn_reference() const {
	Ref<RCSEngagement> ref = memnew(RCSEngagement);
	referencing.push_back(ref.ptr());
	ref->logger = const_cast<RCSEngagementInternal*>(this);
	return ref;
}

// void RCSEngagementInternal::cut_ties_team(RCSTeam* team){
// 	if (offending_team == team) offending_team = nullptr;
// 	VEC_ERASE(deffending_team, team);
// 	VEC_ERASE(participating, team);
// }
// void RCSEngagementInternal::cut_ties_squad(RCSSquad* squad){
// 	VEC_ERASE(offending_squads, squad);
// 	VEC_ERASE(deffending_squads, squad);
// }

// void RCSEngagementInternal::cut_ties_to_all(){
// 	offending_team = nullptr;
// 	BilateralCleanup(participating,     engagement_loggers);
// 	BilateralCleanup(deffending_team,  engagement_loggers);
// 	BilateralCleanup(offending_squads,  engagement_loggers);
// 	BilateralCleanup(deffending_squads, engagement_loggers);
// }

void RCSEngagementInternal::erase_reference(RCSEngagement* to){
	to->logger = nullptr;
	auto idx = referencing.find(to);
	if (idx) referencing.erase(idx);
}

void RCSEngagementInternal::flush_all_references(){
	for (auto E = referencing.front(); E; E = E->next()){
		E->get()->logger = nullptr;
		referencing.erase(E);
	}
}

Array RCSEngagementInternal::get_involving_teams() const {
	Array re;
	re.push_back(offending_team); re.push_back(deffending_team);
	return re;
}
Array RCSEngagementInternal::get_involving_squads() const {
	Array re;
	re.push_back(offending_squads);
	re.push_back(deffending_squads);
	return re;
}
void RCSEngagementInternal::remove_side(RCSSquad* which){
	if (deffending_squads == which->get_self()){
		winner = offending_squads;
	} else if (offending_squads == which->get_self()){
		winner = deffending_squads;
	} else ERR_FAIL_MSG("Failed to determine which squad to remove");
	// engaging = false;
	auto new_event = std::make_shared<EngagementEventReport>();
	new_event->event_type = EngagementEventReport::EngagementFinished;
	new_event->opaque_engagement = self_ref;
	sim->engagement_event(new_event);
}

RCSEngagement::RCSEngagement(){
	logger = nullptr;
}
RCSEngagement::~RCSEngagement(){
	if (logger) logger->erase_reference(this);
}
void RCSEngagement::_bind_methods(){
	BIND_ENUM_CONSTANT(NA);
	BIND_ENUM_CONSTANT(Stalk);
	BIND_ENUM_CONSTANT(Standoff);
	BIND_ENUM_CONSTANT(Skirmish);
	BIND_ENUM_CONSTANT(Ambush);
	BIND_ENUM_CONSTANT(Assault);
	BIND_ENUM_CONSTANT(Landing);
	// BIND_ENUM_CONSTANT(FullEngagement);
	BIND_ENUM_CONSTANT(Encirclement);
	BIND_ENUM_CONSTANT(Siege);

	ClassDB::bind_method(D_METHOD("is_engagement_happening"), &RCSEngagement::is_engagement_happening);
	ClassDB::bind_method(D_METHOD("is_engagement_over"), &RCSEngagement::is_engagement_over);
	ClassDB::bind_method(D_METHOD("get_scale"), &RCSEngagement::get_scale);

	ClassDB::bind_method(D_METHOD("get_involving_teams"), &RCSEngagement::get_involving_teams);
	ClassDB::bind_method(D_METHOD("get_involving_squads"), &RCSEngagement::get_involving_squads);
	ClassDB::bind_method(D_METHOD("get_offending_team"), &RCSEngagement::get_offending_team);
	ClassDB::bind_method(D_METHOD("get_deffending_team"), &RCSEngagement::get_deffending_team);
	ClassDB::bind_method(D_METHOD("get_offending_squad"), &RCSEngagement::get_offending_squad);
	ClassDB::bind_method(D_METHOD("get_deffending_squad"), &RCSEngagement::get_deffending_squad);
}

#pragma endregion

#pragma region Profiles

void RCSProfile::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_pname", "name"), &RCSProfile::set_pname);
	ClassDB::bind_method(D_METHOD("get_pname"), &RCSProfile::get_pname);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "profile_name"), "set_pname", "get_pname");
}
void RCSSpatialProfile::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_detection_threshold", "new_threshold"), &RCSCombatantProfile::set_detection_threshold);
	ClassDB::bind_method(D_METHOD("get_detection_threshold"), &RCSCombatantProfile::get_detection_threshold);

	ClassDB::bind_method(D_METHOD("set_acquisition_threshold", "new_threshold"), &RCSCombatantProfile::set_acquisition_threshold);
	ClassDB::bind_method(D_METHOD("get_acquisition_threshold"), &RCSCombatantProfile::get_acquisition_threshold);

	ClassDB::bind_method(D_METHOD("set_subvention", "new_subvention"), &RCSCombatantProfile::set_subvention);
	ClassDB::bind_method(D_METHOD("get_subvention"), &RCSCombatantProfile::get_subvention);

	ClassDB::bind_method(D_METHOD("set_phantom_mode", "new_state"), &RCSCombatantProfile::set_phantom_mode);
	ClassDB::bind_method(D_METHOD("get_phantom_mode"), &RCSCombatantProfile::get_phantom_mode);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_threshold", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_detection_threshold", "get_detection_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "acquisition_threshold", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_acquisition_threshold", "get_acquisition_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_subvention", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_subvention", "get_subvention");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_phantom"), "set_phantom_mode", "get_phantom_mode");
}

#pragma endregion

#pragma region Combatant
RCSCombatantProfile::RCSCombatantProfile(){}

void RCSCombatantProfile::_bind_methods(){
	// ClassDB::bind_method(D_METHOD("set_id", "new_id"), &RCSCombatantProfile::set_id);
	// ClassDB::bind_method(D_METHOD("get_id"), &RCSCombatantProfile::get_id);

	ClassDB::bind_method(D_METHOD("set_stand", "new_stand"), &RCSCombatantProfile::set_stand);
	ClassDB::bind_method(D_METHOD("get_stand"), &RCSCombatantProfile::get_stand);

	ClassDB::bind_method(D_METHOD("set_combatant_attributes", "attr"), &RCSCombatantProfile::set_combatant_attributes);
	ClassDB::bind_method(D_METHOD("get_combatant_attributes"), &RCSCombatantProfile::get_combatant_attributes);

	// ClassDB::bind_method(D_METHOD("_set_detection_meter", "new_param"), &RCSCombatantProfile::_set_detection_meter);
	// ClassDB::bind_method(D_METHOD("_get_detection_meter"), &RCSCombatantProfile::_get_detection_meter);

	BIND_ENUM_CONSTANT(NormalCombatant);
	BIND_ENUM_CONSTANT(Projectile);
	BIND_ENUM_CONSTANT(Guided);
	BIND_ENUM_CONSTANT(Undetectable);

	BIND_ENUM_CONSTANT(Movable);
	BIND_ENUM_CONSTANT(Static);
	BIND_ENUM_CONSTANT(Passive);
	BIND_ENUM_CONSTANT(Deffensive);
	BIND_ENUM_CONSTANT(Retaliative);
	BIND_ENUM_CONSTANT(Aggressive);

	// ADD_PROPERTY(PropertyInfo(Variant::INT, "combatant_id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stand"), "set_stand", "get_stand");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "attributes"), "set_combatant_attributes", "get_combatant_attributes");
	// ADD_PROPERTY(PropertyInfo(Variant::REAL, "_detection_meter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_detection_meter", "_get_detection_meter");
}

void RCSProjectileProfile::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_hpl", "hpl"), &RCSProjectileProfile::set_hpl);
	ClassDB::bind_method(D_METHOD("get_hpl"), &RCSProjectileProfile::get_hpl);

	ClassDB::bind_method(D_METHOD("set_hps", "hps"), &RCSProjectileProfile::set_hps);
	ClassDB::bind_method(D_METHOD("get_hps"), &RCSProjectileProfile::get_hps);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "heat_per_launch", PROPERTY_HINT_RANGE, String("0.0,") + itos(MAX_HEAT_PER_INSTANCE) + String(",0.1")), "set_hpl", "get_hpl");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "heat_per_second", PROPERTY_HINT_RANGE, String("0.0,") + itos(MAX_HEAT_PER_INSTANCE) + String(",0.1")), "set_hps", "get_hps");
}

RCSCombatant::RCSCombatant(){
	simulation = nullptr;
	my_squad = nullptr;
	fired_by = nullptr;
	magic = COMBATANT_MAGIC;
	// projectile_bind = new RCSProjectileBind();
	projectile_bind = std::make_unique<RCSProjectileBind>();
}

RCSCombatant::~RCSCombatant(){
	set_simulation(nullptr);
	set_squad(nullptr);
	fired_by = nullptr;
	// Target and host must be manually reset prior to deletion
	projectile_bind->set_target(nullptr);
	while (!punishers.empty()){
#ifdef DEBUG_ENABLED
		auto prev_size = punishers.size();
		punishers.front()->get()->set_target(nullptr);
		auto this_size = punishers.size();
		if (prev_size == this_size) punishers.erase(punishers.front());
		WARN_PRINT("punishers\' size did not change after erasure, this could mean the target was invalid");
#else
		punishers.front()->get()->set_target(nullptr);
#endif
	}
	projectile_bind->set_host(nullptr);
	// rcsdel(projectile_bind);
}

void RCSCombatant::set_projectile_profile(const Ref<RCSProjectileProfile>& new_prof) { projectile_bind->set_profile(new_prof); }
Ref<RCSProjectileProfile> RCSCombatant::get_projectile_profile() const { return projectile_bind->get_profile(); }

// void RCSCombatant::set_cname(const StringName &cname) {
// 	ERR_FAIL_COND(combatant_profile.is_null());
// 	combatant_profile->set_pname(cname);
// }
// StringName RCSCombatant::get_cname() const {
// 	ERR_FAIL_COND_V(combatant_profile.is_null(), StringName());
// 	return combatant_profile->get_pname();
// }
// void RCSCombatant::set_stand(const uint32_t &new_stand) {
// 	ERR_FAIL_COND(combatant_profile.is_null());
// 	combatant_profile->set_stand(new_stand);
// }
// uint32_t RCSCombatant::get_stand() const {
// 	ERR_FAIL_COND_V(combatant_profile.is_null(), 0);
// 	return combatant_profile->get_stand();
// }
Ref<RawRecord> RCSCombatant::serialize() const {
	Ref<RawRecord> rrec = memnew(RawRecord);
	// Ref<RawRecordData> rdata = memnew(RawRecordData);
	// rdata->name = StringName("__main");

	// PUSH_RECORD_PRIMITIVE(rdata, space_transform);
	// PUSH_RECORD_PRIMITIVE(rdata, local_transform);
	// // PUSH_RECORD_PRIMITIVE(rdata, stand);

	// rdata->external_refs.resize(rdata->table.size());
	// rdata->external_refs.fill(0);
	return rrec;
}
bool RCSCombatant::serialize(const Ref<RawRecord> &from) {
	return false;
}

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

bool RCSCombatant::is_engagable(RCSCombatant* bogey) const {
	if (!bogey || !my_squad) return false;
	if (bogey->get_profile().is_valid() && bogey->get_profile()->get_phantom_mode()) return false;
	auto bogey_squad = bogey->get_squad();
	if (bogey_squad == my_squad) return false;
	return my_squad->is_engagable(bogey_squad);
}
bool RCSCombatant::is_scannable(RCSCombatant* bogey) const {
	if (!bogey || !my_squad) return false;
	if (bogey->get_profile().is_valid() && bogey->get_profile()->get_phantom_mode()) return false;
	auto bogey_squad = bogey->get_squad();
	if (bogey_squad == my_squad) return false;
	return my_squad->is_scannable(bogey_squad);
}

void RCSCombatant::poll(const float& delta){
	RID_RCS::poll(delta);
	if (combatant_profile.is_null()) return;
	auto new_meter = detection_meter - (combatant_profile->get_subvention() * delta);
	detection_meter = CLAMP(new_meter, 0.0, COMBATANT_DETECTION_LIMIT);
}

void RCSCombatant::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_combatant(this);
		// simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_combatant(this);
}

#pragma endregion

#pragma region Squad
RCSSquad::RCSSquad(){
	simulation = nullptr;
	my_team = nullptr;
	magic = SQUAD_MAGIC;
	// engagement_loggers = nullptr;
}

RCSSquad::~RCSSquad(){
	set_simulation(nullptr);
	set_team(nullptr);
	while (!combatants.empty()){
		combatants[0]->set_squad(nullptr);
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

bool RCSSquad::is_engagable(RCSSquad *bogey) const {
	if (!bogey || !my_team) return false;
	auto bogey_team = bogey->get_team();
	// if (bogey_team == my_team) return false;
	return my_team->is_engagable(bogey_team);
}
bool RCSSquad::is_scannable(RCSSquad *bogey) const {
	if (!bogey || !my_team) return false;
	auto bogey_team = bogey->get_team();
	// if (bogey_team == my_team) return false;
	return my_team->is_scannable(bogey_team);
}

#pragma endregion

#pragma region Team
RCSUnilateralTeamsBind::RCSUnilateralTeamsBind(){}
RCSUnilateralTeamsBind::~RCSUnilateralTeamsBind(){}

void RCSUnilateralTeamsBind::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_relationship", "rel"), &RCSUnilateralTeamsBind::set_relationship);
	ClassDB::bind_method(D_METHOD("get_relationship"), &RCSUnilateralTeamsBind::get_relationship);

	ClassDB::bind_method(D_METHOD("set_attributes", "attr"), &RCSUnilateralTeamsBind::set_attributes);
	ClassDB::bind_method(D_METHOD("get_attributes"), &RCSUnilateralTeamsBind::get_attributes);

	// ClassDB::bind_method(D_METHOD("set_from_rid", "rid"), &RCSUnilateralTeamsBind::set_from_rid);
	// ClassDB::bind_method(D_METHOD("get_from_rid"), &RCSUnilateralTeamsBind::get_from_rid);

	// ClassDB::bind_method(D_METHOD("set_to_rid", "rid"), &RCSUnilateralTeamsBind::set_to_rid);
	ClassDB::bind_method(D_METHOD("get_to_rid"), &RCSUnilateralTeamsBind::get_to_rid);

	BIND_ENUM_CONSTANT(TeamNeutral);
	BIND_ENUM_CONSTANT(TeamAllies);
	BIND_ENUM_CONSTANT(TeamHostiles);

	BIND_ENUM_CONSTANT(ITA_None);
	BIND_ENUM_CONSTANT(ITA_Engagable);
	BIND_ENUM_CONSTANT(ITA_AutoEngage);
	BIND_ENUM_CONSTANT(ITA_DetectionWarning);
	BIND_ENUM_CONSTANT(ITA_Scanable);
	BIND_ENUM_CONSTANT(ITA_ShareRadar);
	BIND_ENUM_CONSTANT(ITA_ShareLocation);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "relationship", PROPERTY_HINT_ENUM, "Neutral,Allies,Hostiles"), "set_relationship", "get_relationship");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "attributes", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_attributes", "get_attributes");
	// ADD_PROPERTY(PropertyInfo(Variant::_RID, "from"), "set_from_rid", "get_from_rid");
	// ADD_PROPERTY(PropertyInfo(Variant::_RID, "toward"), "set_to_rid", "get_to_rid");
}

RID_TYPE RCSUnilateralTeamsBind::get_to_rid() const{
	return (!toward ? RID_TYPE() : toward->get_self());
}


RCSTeam::RCSTeam(){
	simulation = nullptr;
}

RCSTeam::~RCSTeam(){
	set_simulation(nullptr);
	while (!squads.empty()){
		squads[0]->set_team(nullptr);
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
VECTOR<std::weak_ptr<RCSEngagementInternal>> RCSTeam::get_all_engagements() const{
	VECTOR<std::weak_ptr<RCSEngagementInternal>> re;
	for (uint32_t i = 0, s = squads.size(); i < s; i++){
		const auto& squad = squads[i];
		const auto& sp = squad->get_participating();
		for (uint32_t j = 0, ms = sp.size(); j < ms; j++){
			re.push_back(sp[j]);
		}
	}
	return re;
}
Array RCSTeam::get_all_engagements_compat() const {
	Array re;
	for (uint32_t i = 0, s = squads.size(); i < s; i++){
		const auto& squad = squads[i];
		const auto& sp = squad->get_participating();
		for (uint32_t j = 0, ms = sp.size(); j < ms; j++){
			auto en = sp[j].lock();
			if (!en) continue;
			re.push_back(en->spawn_reference());
		}
	}
	return re;
}
Ref<RCSUnilateralTeamsBind> RCSTeam::add_link(RCSTeam *toward){
	Ref<RCSUnilateralTeamsBind> link = memnew(RCSUnilateralTeamsBind);
	link->toward = toward;
	team_binds.push_back(link);
	return link;
}

bool RCSTeam::remove_link(RCSTeam *to){
	auto size = team_binds.size();
	for (uint32_t i = 0; i < size; i++){
		auto link = team_binds[i];
		if (link.is_null()) continue;
		if (link->toward == to){
			VEC_REMOVE(team_binds, i);
			return true;
		}
	}
	return false;
}

Ref<RCSUnilateralTeamsBind> RCSTeam::get_link_to(RCSTeam *to) const {
	auto size = team_binds.size();
	for (uint32_t i = 0; i < size; i++){
		auto link = team_binds[i];
		if (link.is_null()) continue;
		if (link->toward == to) return link;
	}
	return Ref<RCSUnilateralTeamsBind>();
}
void RCSTeam::purge_all_links() {
	while (!team_binds.empty()) {
		auto link = team_binds[0];
		VEC_REMOVE(team_binds, 0);
		if (!link.is_valid()) continue;
		if (!link->toward) continue;
		link->toward->remove_link(this);
	}
}

#pragma endregion

#pragma region Radar
#define RADAR_PROFILE_DETECT_VMETHOD StringName("_detect")
#define RADAR_PROFILE_LATE_DETECT_VMETHOD StringName("_late_detect")
#define RADAR_PROFILE_ACQUIRE_VMETHOD StringName("_acquire_lock")

RCSRadarProfile::RCSRadarProfile(){

}

void RCSRadarProfile::_bind_methods(){
	ClassDB::bind_method(D_METHOD("set_dcurve", "curve"), &RCSRadarProfile::set_dcurve);
	ClassDB::bind_method(D_METHOD("get_dcurve"), &RCSRadarProfile::get_dcurve);

	ClassDB::bind_method(D_METHOD("set_acurve", "curve"), &RCSRadarProfile::set_acurve);
	ClassDB::bind_method(D_METHOD("get_acurve"), &RCSRadarProfile::get_acurve);

	ClassDB::bind_method(D_METHOD("set_spread", "new_spread"), &RCSRadarProfile::set_spread);
	ClassDB::bind_method(D_METHOD("get_spread"), &RCSRadarProfile::get_spread);

	ClassDB::bind_method(D_METHOD("set_freq", "new_freq"), &RCSRadarProfile::set_freq);
	ClassDB::bind_method(D_METHOD("get_freq"), &RCSRadarProfile::get_freq);

	ClassDB::bind_method(D_METHOD("set_rpd", "new_rpd"), &RCSRadarProfile::set_rpd);
	ClassDB::bind_method(D_METHOD("get_rpd"), &RCSRadarProfile::get_rpd);

	ClassDB::bind_method(D_METHOD("set_method", "new_met"), &RCSRadarProfile::set_method);
	ClassDB::bind_method(D_METHOD("get_method"), &RCSRadarProfile::get_method);

	ClassDB::bind_method(D_METHOD("set_base", "new_base"), &RCSRadarProfile::set_base);
	ClassDB::bind_method(D_METHOD("get_base"), &RCSRadarProfile::get_base);

	ClassDB::bind_method(D_METHOD("set_target_mode", "new_tm"), &RCSRadarProfile::set_target_mode);
	ClassDB::bind_method(D_METHOD("get_target_mode"), &RCSRadarProfile::get_target_mode);

	ClassDB::bind_method(D_METHOD("set_attr", "new_attr"), &RCSRadarProfile::set_attr);
	ClassDB::bind_method(D_METHOD("get_attr"), &RCSRadarProfile::get_attr);

	ClassDB::bind_method(D_METHOD("set_cmask", "new_mask"), &RCSRadarProfile::set_cmask);
	ClassDB::bind_method(D_METHOD("get_cmask"), &RCSRadarProfile::get_cmask);

	ClassDB::bind_method(D_METHOD("set_contribution", "new_con"), &RCSRadarProfile::set_contribution);
	ClassDB::bind_method(D_METHOD("get_contribution"), &RCSRadarProfile::get_contribution);

	BIND_ENUM_CONSTANT(ScanModeSwarm);
	BIND_ENUM_CONSTANT(ScanModeSingle);
	BIND_ENUM_CONSTANT(ScanTransform);
	BIND_ENUM_CONSTANT(ScanDirectSpaceState);
	BIND_ENUM_CONSTANT(ScanCollideWithBodies);
	BIND_ENUM_CONSTANT(ScanCollideWithAreas);
	BIND_ENUM_CONSTANT(TargetCombatants);
	BIND_ENUM_CONSTANT(TargetSquadPartial);
	// BIND_ENUM_CONSTANT(TargetSquadCompleted);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "acquisition_curve", PROPERTY_HINT_RESOURCE_TYPE, "", PROPERTY_USAGE_DEFAULT, "AdvancedCurve"), "set_acurve", "get_acurve");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "detection_curve", PROPERTY_HINT_RESOURCE_TYPE, "", PROPERTY_USAGE_DEFAULT, "AdvancedCurve"), "set_dcurve", "get_dcurve");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_spread", PROPERTY_HINT_RANGE, "0.009,3.141,0.001"), "set_spread", "get_spread");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_frequency", PROPERTY_HINT_RANGE, "0.001,5.0,0.001"), "set_freq", "get_freq");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "scan_contribution", PROPERTY_HINT_RANGE, "0.0001,0.999,0.001"), "set_contribution", "get_contribution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "rays_per_degree", PROPERTY_HINT_RANGE, "0.1,5.0,0.1"), "set_rpd", "get_rpd");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_method", PROPERTY_HINT_ENUM, "ScanModeSwarm,ScanModeSingle"), "set_method", "get_method");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_base", PROPERTY_HINT_ENUM, "ScanTransform,ScanDirectSpaceState"), "set_base", "get_base");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_target", PROPERTY_HINT_ENUM, "TargetCombatants,TargetSquadPartial"), "set_target_mode", "get_target_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_attributes"), "set_attr", "get_attr");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_cmask", "get_cmask");

	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_ACQUIRE_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_LATE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
}

void RCSRadarProfile::ping_target(RadarPingRequest* ping_request){
	auto cache = ping_cache(ping_request);
	auto script = get_script_instance();

	if (ping_request->compute_lock){
		if (script && script->has_method(RADAR_PROFILE_ACQUIRE_VMETHOD)){
			auto script_re = script->call(RADAR_PROFILE_ACQUIRE_VMETHOD,
				ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
			if (script_re.get_type() == Variant::BOOL) ping_request->lock_result = (bool)script_re;
		} else internal_acquire(ping_request, cache);
	}

	// If locked, detected by default

	if (ping_request->lock_result) {
		ping_request->detect_result = true;
	} else if (script && script->has_method(RADAR_PROFILE_DETECT_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_DETECT_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_detect(ping_request, cache);
	memdelete(cache);
}
void RCSRadarProfile::swarm_detect(RadarPingRequest* ping_request){
	auto script = get_script_instance();
	if (script && script->has_method(RADAR_PROFILE_LATE_DETECT_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_LATE_DETECT_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_swarm_detect(ping_request);
}
RCSRadarProfile::RadarPingCache *RCSRadarProfile::ping_cache(RadarPingRequest* ping_request) const{
	auto cache = new RadarPingCache();
	auto bogey_transform = ping_request->to->get_combined_transform();
	cache->from_origin = ping_request->self_transform.get_origin();
	cache->from_forward = -(ping_request->self_transform.get_basis()[2]);
	cache->to_origin = bogey_transform.get_origin();
	cache->distance = cache->from_origin.distance_to(cache->to_origin);
	cache->to_dir = cache->from_origin.direction_to(cache->to_origin);
	cache->bearing = cache->from_forward.angle_to(cache->to_dir);
	return cache;
}

void RCSRadarProfile::internal_detect(RadarPingRequest* ping_request, RadarPingCache* cache){
	// Null safety is handled by RCSRadar
	// Check if bogey is out-of-range
	if (cache->distance > detection_curve->get_range()) return;
	// Check for bogey's bearing
	if (cache->bearing > spread) return;
	const auto& bogey_profile = ping_request->bogey_profile;
	// If there is no profile presented for bogey,
	// do no further test and marked as detected
	if (bogey_profile.is_null()) {
		ping_request->detect_result = true;
		return;
	}
	auto dst_percentage = cache->distance / detection_curve->get_range();
	auto scan_power = detection_curve->interpolate_baked(dst_percentage);
	switch (scan_method){
		case RadarScanMode::ScanModeSingle: {
			// Simple scan check, if radar power overwhelm
			// bogey's threshold, mark as detected
			if (scan_power > bogey_profile->get_detection_threshold())
				ping_request->detect_result = true;
				// RadarPartialSquadCheck(ping_request->to, cache->auto_detect, scan_target);
			break;
		} case RadarScanMode::ScanModeSwarm: {
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
		} default: { ERR_FAIL_MSG("Invalid scan_method"); }
	}
}

void RCSRadarProfile::internal_acquire(RadarPingRequest* ping_request, RadarPingCache* cache){
	if (cache->distance > acquisition_curve->get_range()) return;
	// Check for bogey's bearing
	if (cache->bearing > spread) return;
	const auto& bogey_profile = ping_request->bogey_profile;
	// If there is no profile presented for bogey,
	// do no further test and marked as detected
	if (bogey_profile.is_null()) {
		ping_request->lock_result = true;
		return;
	}
	// There should only one method for lock acquisition
	auto dst_percentage = cache->distance / acquisition_curve->get_range();
	auto scan_power = acquisition_curve->interpolate_baked(dst_percentage);
	if (scan_power > bogey_profile->get_acquisition_threshold()){
		ping_request->lock_result = true;
	}
}

void RCSRadarProfile::internal_swarm_detect(RadarPingRequest* ping_request){
	auto bogey_curr_dmeter = ping_request->to->_get_detection_meter();
	if (bogey_curr_dmeter >= 1.0)
		ping_request->detect_result = true;
}

void RCSRadar::fetch_space_state() {
	auto scene_tree = SceneTree::get_singleton();
	ERR_FAIL_COND(!scene_tree);
	auto root = scene_tree->get_root();
	if (!root)
		return;
	auto world = root->find_world();
	if (world.is_null())
		return;
	space_state = world->get_direct_space_state();
}

RCSRadar::RCSRadar(){
	simulation = nullptr;
	assigned_vessel = nullptr;
	space_state = nullptr;
	// async_handler = rcsnew(AsyncOperator(this));
	// if (!async_handler->setup_success) rcsdel(async_handler);
}
RCSRadar::~RCSRadar(){
	// if (async_handler) rcsdel(async_handler);
	set_simulation(nullptr);
}
void RCSRadar::set_simulation(RCSSimulation* sim){
	if (simulation){
		simulation->remove_radar(this);
		simulation = nullptr;
	}
	simulation = sim;
	if (simulation) simulation->add_radar(this);
}
void RCSRadar::set_vessel(RCSCombatant *new_vessel) {
	assigned_vessel = new_vessel;
}

// TargetSquadPartial is expired
// No real effective way to implement

void RCSRadar::ping_base_transform(const float &delta) {
	// auto combatants = simulation->radar_request_scannable(this);
	auto scanable = screening_result.get_scanable();
	auto c_size = scanable.size();
	for (uint32_t i = 0; i < c_size; i++) {
		auto com = scanable[i];
		if (!com /* || !assigned_vessel->is_scannable(com)*/)
			continue;
		// ---------------------------------------------------
		bool engagable = false;
		if (com->get_magic() == COMBATANT_MAGIC) engagable = assigned_vessel->is_engagable((RCSCombatant*)com);
		else if (com->get_magic() == SQUAD_MAGIC){
			auto my_squad = assigned_vessel->get_squad();
			if (my_squad) engagable = my_squad->is_engagable((RCSSquad*)com);
		}
		RadarPingRequest req(assigned_vessel, com, engagable);
		req.self_transform = assigned_vessel->get_combined_transform();
		req.bogey_profile = com->get_spatial_profile();
		rprofile->ping_target(&req);

		if (req.late_recheck) {
			simulation->demand_radar_recheck(new RadarRecheckTicket(this, com));
		} else if (req.detect_result) {
			// detected[com->get_self()] = com;
			target_detected(com);
		}
		if (req.lock_result) {
			// locked[com->get_self()] = com;
			target_locked(com);
		}
	}
}
void RCSRadar::ping_base_direct_space_state(const float &delta) {
	if (space_state == nullptr) {
		fetch_space_state();
		ERR_FAIL_COND(space_state == nullptr);
	}
	// Rays per degree = ray count / degree count
	// Degrees * PI / 180 = Radians
	// Rays per radian = ray count / (degrees * PI / 180) = (180 * ray count) / (PI * degrees count)
	double rays_per_degree = rprofile->get_rpd();
	double rays_per_radian = (rays_per_degree * 180.0) / Math_PI;
	double radians_per_ray = 1.0 / rays_per_radian;
	double ray_spread = rprofile->get_spread();
	double max_distance = rprofile->get_dcurve()->get_range();
	Transform vessel_transform = assigned_vessel->get_combined_transform();
	Vector3 VEC3_UP = Vector3(0.0, 1.0, 0.0);
	Vector3 vessel_origin = vessel_transform.get_origin();
	Vector3 vessel_forward = vessel_transform.get_basis()[2];
	uint32_t radar_attr = rprofile->get_attr();
	uint32_t radar_cmask = rprofile->get_cmask();
	bool collide_with_bodies = radar_attr & RCSRadarProfile::ScanCollideWithBodies;
	bool collide_with_areas = radar_attr & RCSRadarProfile::ScanCollideWithAreas;
	Set<RID_TYPE> exclude;
	for (double angle_offset = -ray_spread; angle_offset < ray_spread; ray_spread += radians_per_ray) {
		Vector3 current_ray = vessel_forward.rotated(VEC3_UP, angle_offset);
		Vector3 scan_origin = vessel_origin + (current_ray * max_distance);
		PhysicsDirectSpaceState::RayResult ray_res;
		auto intersect_result = space_state->intersect_ray(vessel_origin, scan_origin, ray_res,
				exclude, radar_cmask, collide_with_bodies, collide_with_areas);
		if (!intersect_result)
			continue;
		auto obj_id = ray_res.collider_id;
		auto combatant = simulation->request_combatant_from_iid(obj_id);
		// If found object is not combatant,
		// it might be an obstacle, therefore, don't exclude it
		// This method might skip over some combatant,
		// but with enough ray density, it wouldn't be a problem
		if (!combatant)
			continue;
		// FIX: exclude the RID_TYPE regardless of the assertion result
		// if it make it all the way to this point ||
		//                                        \./
		if (!assigned_vessel->is_scannable(combatant)) {
		} else {
			RadarPingRequest req(assigned_vessel, combatant, assigned_vessel->is_engagable(combatant));
			req.self_transform = assigned_vessel->get_combined_transform();
			rprofile->ping_target(&req);

			if (req.late_recheck) {
				simulation->demand_radar_recheck(new RadarRecheckTicket(this, combatant));
			} else if (req.detect_result) {
				// detected[combatant->get_self()] = combatant;
				target_detected(combatant);
			}
			if (req.lock_result) {
				// locked[combatant->get_self()] = combatant;
				target_locked(combatant);
			}
		}
		exclude.insert(ray_res.rid);
	}
}

RCSRadar::TargetScreeningResult RCSRadar::target_screening() const {
	TargetScreeningResult re;
	ERR_FAIL_COND_V(!simulation || !assigned_vessel || rprofile.is_null(), re);
	auto target_mode = rprofile->get_target_mode();
	switch (target_mode){
		case RCSRadarProfile::TargetCombatants: {
			re.scanable_combatants = simulation->request_reachable_combatants(assigned_vessel->get_squad());
			re.selected_scanable.resize(re.scanable_combatants.size());
			for (uint32_t i = 0, s = re.scanable_combatants.size(); i < s; i++){
				re.selected_scanable.write[i] = re.scanable_combatants[i];
			}
		} case RCSRadarProfile::TargetSquadPartial: {
			re.scanable_squads = simulation->request_scanable_squads(assigned_vessel->get_squad());
			re.selected_scanable.resize(re.scanable_squads.size());
			for (uint32_t i = 0, s = re.scanable_squads.size(); i < s; i++){
				re.selected_scanable.write[i] = re.scanable_squads[i];
			}
		}
	}
	return re;
}

void RCSRadar::target_detected(RCSSpatial* com){
	auto rid = com->get_self();
	auto prev_index = nolonger_detected.find(rid);
	if (prev_index == nullptr) {
		newly_detected.push_back(rid);
	} else {
		// If previously detected rid is still being detected during this call
		// then erase it from the list, so the radar can conclude which
		// combatants "vanished" later on.
		nolonger_detected.erase(prev_index);
	}
	detected.push_back(rid);
}

void RCSRadar::target_locked(RCSSpatial* com){
	auto rid = com->get_self();
	auto prev_index = nolonger_locked.find(rid);
	if (prev_index == nullptr) {
		newly_detected.push_back(rid);
	} else {
		// Same story
		nolonger_locked.erase(prev_index);
	}
	locked.push_back(rid);
}

void RCSRadar::poll(const float &delta) {
	// Do stuff before unlocking
	// async_handler->unlock_op();
	RID_RCS::poll(delta);
	if (rprofile.is_null() || !simulation || !assigned_vessel)
		return;
	if (rprofile->get_acurve().is_null() || rprofile->get_dcurve().is_null())
		return;
	timer += delta;
	if (timer < (1.0 / rprofile->get_freq()))
		return;
	screening_result = target_screening();
	//-------------------------------------------------
	timer = 0.0;
	nolonger_detected = detected;
	nolonger_locked = locked;
	// detected_squads.clear();
	detected.clear();
	locked.clear();
	newly_detected.clear();
	newly_locked.clear();
	//--------------------------------------------------
	// Combatants pool size should stay the same during radars' iteration

	ping_base_transform(delta);
	// simulation->demand_radar_conclude(new RadarConcludeTicket(this));
	conclude(delta);
}

void RCSRadar::late_check(const float &delta, RadarRecheckTicket *recheck) {
	if (rprofile.is_null() || !simulation || !assigned_vessel)
		return;
	if (!recheck->bogey)
		return;
	RadarPingRequest req(assigned_vessel, recheck->bogey);
	req.self_transform = assigned_vessel->get_combined_transform();
	rprofile->swarm_detect(&req);
	if (req.detect_result) {
		// detected[recheck->bogey->get_self()] = recheck->bogey;
		target_detected(recheck->bogey);
	}
}
void RCSRadar::conclude(const float& delta){
	// nolonger_detected = prev_detected;
	// nolonger_locked = prev_locked;
	if (nolonger_detected.empty() && nolonger_locked.empty() &&
		newly_detected.empty() && newly_locked.empty()) return;
	send_conclusion();
}
void RCSRadar::send_conclusion(){
	std::shared_ptr<RadarEventReport> ticket = std::make_shared<RadarEventReport>();
	ticket->sender = this;
	ticket->event_type = RadarEventReport::ScanConcluded;
	ticket->conclusion = std::make_unique<RadarEventReport::ScanConclusion>();
	// for (auto E = detected.front(); E; E = E->next()){
	// 	ticket->conclusion->add_detected(E->get());
	// }
	// for (auto E = locked.front(); E; E = E->next()){
	// 	ticket->conclusion->add_locked(E->get());
	// }
	for (auto E = newly_detected.front(); E; E = E->next()){
		ticket->conclusion->add_newly_detected(E->get());
	}
	for (auto E = newly_locked.front(); E; E = E->next()){
		ticket->conclusion->add_newly_locked(E->get());
	}
	for (auto E = nolonger_detected.front(); E; E = E->next()){
		ticket->conclusion->add_nolonger_detected(E->get());
	}
	for (auto E = nolonger_locked.front(); E; E = E->next()){
		ticket->conclusion->add_nolonger_locked(E->get());
	}
	ticket->target_mode = (uint32_t)get_profile()->get_target_mode();
	simulation->radar_event(ticket);
	
}


#pragma endregion
