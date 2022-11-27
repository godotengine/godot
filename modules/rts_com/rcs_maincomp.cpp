#include "rcs_maincomp.h"

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

VARIANT_ENUM_CAST(RCSEngagement::EngagementScale);
VARIANT_ENUM_CAST(RCSCombatantProfile::CombatantAttribute);
VARIANT_ENUM_CAST(RCSCombatantProfile::CombatantStand);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanMode);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanBase);
VARIANT_ENUM_CAST(RCSRadarProfile::RadarScanttributes);
VARIANT_ENUM_CAST(RCSUnilateralTeamsBind::TeamRelationship);
VARIANT_ENUM_CAST(RCSUnilateralTeamsBind::InterTeamAttribute);

static RCSMemoryAllocation RCSMemoryAllocationPtr;
RCSMemoryAllocation* RCSMemoryAllocation::tracker_ptr = nullptr;

RCSMemoryAllocation::RCSMemoryAllocation(){
	ERR_FAIL_COND(tracker_ptr);
	tracker_ptr = this;
}
RCSMemoryAllocation::~RCSMemoryAllocation(){
	tracker_ptr = nullptr;
}

void *operator new(size_t size){
	RCSMemoryAllocation::tracker_ptr->allocated += size;
	return malloc(size);
}

void operator delete(void* memory, size_t size){
	RCSMemoryAllocation::tracker_ptr->deallocated += size;
	free(memory);
}

#pragma endregion

#pragma region EventReport
CombatantEventReport::CombatantEventReport(){}
SquadEventReport::SquadEventReport(){}
TeamEventReport::TeamEventReport(){}
RadarEventReport::RadarEventReport(){}
#pragma endregion

#pragma region Recording
RCSRecording::RCSRecording(){
}
RCSRecording::~RCSRecording(){
	// purge();
}

void RCSRecording::purge(){
	while (!reports_holder.empty()){
		auto report = reports_holder.operator[](0);
		memdelete(report);
		VEC_REMOVE(reports_holder, 0);
	}
}

void RCSRecording::push_event(const Ref<EventReport>& event){
	auto timestamp = OS::get_singleton()->get_ticks_usec();
	auto ticket = rcsnew(EventReportTicket(timestamp, event));
	reports_holder.push_back(ticket);
}

VECTOR<EventReportTicket*> *RCSRecording::events_by_simulation(RCSSimulation* simulation){
	auto re = memnew(VECTOR<EventReportTicket*>());
	auto size = reports_holder.size();
	for (uint32_t i = 0; i < size; i++){
		auto event = reports_holder[i];
		if (event->event->get_simulation() == simulation){
			re->push_back(event);
		}
	}
	return re;
}

void RCSRecording::poll(const float& delta){
	RID_RCS::poll(delta);
}
#pragma endregion

#pragma region Simulation
RCSSimulation::RCSSimulation(){
	recorder = nullptr;
}
RCSSimulation::~RCSSimulation(){
	uint32_t i = 0;
	while (true){
		uint16_t cond = 0;
		CleanerLog(RCSSimulation, i);
		RemoveReference(i, cond, set_simulation, combatants);
		RemoveReference(i, cond, set_simulation, squads);
		RemoveReference(i, cond, set_simulation, teams);
		RemoveReference(i, cond, set_simulation, radars);
		if (cond == 0) break;
		i++;
	}
	// if (recorder) recorder->remove_simulation(this);
}

#define ThiccCheck(vec) \
	if (vec.size() > 1000) { print_verbose(String("[" FUNCTION_STR "] She\'s a thicc one...: " + itos(vec.size()))); ERR_FAIL(); }

#define FattCheck(vec) \
	if (vec.size() > 1000) { print_verbose(String("[" FUNCTION_STR "] She\'s a fatt one...: " + itos(vec.size()))); ERR_FAIL(); }

void RCSSimulation::add_combatant(RCSCombatant* com){
	FattCheck(combatants);
	combatants.push_back(com);
}
void RCSSimulation::add_squad(RCSSquad* squad){
	FattCheck(squads);
	squads.push_back(squad);
}

void RCSSimulation::add_team(RCSTeam* team){
	FattCheck(teams);
	teams.push_back(team);
}

void RCSSimulation::add_radar(RCSRadar* rad){
	FattCheck(combatants);
	radars.push_back(rad);
}

void RCSSimulation::remove_combatant(RCSCombatant* com)
{
	print_verbose(String("Removing Combatant..."));
	ThiccCheck(combatants);
	VEC_ERASE(combatants, com)
}

void RCSSimulation::remove_squad(RCSSquad* squad)
{
	ThiccCheck(squads);
	VEC_ERASE(squads, squad)
}

void RCSSimulation::remove_team(RCSTeam* team)
{
	ThiccCheck(teams);
	VEC_ERASE(teams, team)
}

void RCSSimulation::remove_radar(RCSRadar* rad)
	VEC_ERASE(radars, rad)

void RCSSimulation::set_recorder(RCSRecording* rec){
	recorder = rec;
}

void RCSSimulation::combatant_event(Ref<CombatantEventReport> event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::squad_event(Ref<SquadEventReport> event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::team_event(Ref<TeamEventReport> event){
	SimulationRecord(recorder, event);
}
void RCSSimulation::radar_event(Ref<RadarEventReport> event){
	SimulationRecord(recorder, event);
}

RCSCombatant* RCSSimulation::get_combatant_from_iid(const uint64_t& iid) const{
	auto csize = combatants.size();
	for (uint32_t i = 0; i < csize; i++){
		auto combatant = combatants[i];
		if (!combatant) continue;
		if (combatant->get_iid() == iid) return combatant;
	}
	return nullptr;
}

void RCSSimulation::radar_request_recheck(RadarRecheckTicket* ticket){
	rrecheck_requests.push_back(ticket);
}

void RCSSimulation::poll(const float& delta){
	RID_RCS::poll(delta);
	// Radars run first for rechecks
	auto radars_count = radars.size();
	for (uint32_t k = 0; k < radars_count; k++){
		MainPoll(radars, k, delta);
	}
	for (uint32_t u = 0; u < rrecheck_requests.size(); u++){
		auto ticket = rrecheck_requests[u];
		if (!ticket->request_sender) continue;
		ticket->request_sender->late_check(delta, ticket);
		memdelete(ticket);
	}
	rrecheck_requests.clear();
	for (uint32_t j = 0; j < combatants.size(); j++){
		MainPoll(combatants, j, delta);
	}
	for (uint32_t i = 0; i < squads.size(); i++){
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

Ref<RCSEngagement> RCSEngagementInternal::spawn_reference(){
	Ref<RCSEngagement> ref = rcsnew(RCSEngagement);
	referencing.push_back(ref.ptr());
	ref->logger = this;
	return ref;
}

// void RCSEngagementInternal::cut_ties_team(RCSTeam* team){
// 	if (offending_team == team) offending_team = nullptr;
// 	VEC_ERASE(deffending_teams, team);
// 	VEC_ERASE(participating, team);
// }
// void RCSEngagementInternal::cut_ties_squad(RCSSquad* squad){
// 	VEC_ERASE(offending_squads, squad);
// 	VEC_ERASE(deffending_squads, squad);
// }

// void RCSEngagementInternal::cut_ties_to_all(){
// 	offending_team = nullptr;
// 	BilateralCleanup(participating,     engagement_loggers);
// 	BilateralCleanup(deffending_teams,  engagement_loggers);
// 	BilateralCleanup(offending_squads,  engagement_loggers);
// 	BilateralCleanup(deffending_squads, engagement_loggers);
// }

void RCSEngagementInternal::erase_reference(RCSEngagement* to){
	to->logger = nullptr;
	VEC_ERASE(referencing, to);
}

void RCSEngagementInternal::flush_all_references(){
	BilateralCleanup(referencing, logger);
}

bool RCSEngagementInternal::is_engagement_happening() const {
	return engaging;
}
bool RCSEngagementInternal::is_engagement_over() const {
	return !engaging;
}
uint32_t RCSEngagementInternal::get_scale() const {
	return scale;
}
Array RCSEngagementInternal::get_involving_teams() const {
	Array re;
	VEC2GDARRAY(participating, re);
	return re;
}
Array RCSEngagementInternal::get_involving_squads() const {
	Array re;
	VEC2GDARRAY(offending_squads, re);
	VEC2GDARRAY(deffending_squads, re);
	return re;
}
RID_TYPE RCSEngagementInternal::get_offending_team() const {
	// if (!offending_team) return RID_TYPE();
	// return offending_team->get_self();
	return offending_team;
}
Array RCSEngagementInternal::get_deffending_teams() const {
	Array re;
	VEC2GDARRAY(deffending_teams, re);
	return re;
}
Array RCSEngagementInternal::get_offending_squads() const {
	Array re;
	VEC2GDARRAY(offending_squads, re);
	return re;
}
Array RCSEngagementInternal::get_deffending_squads() const {
	Array re;
	VEC2GDARRAY(deffending_squads, re);
	return re;
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
	BIND_ENUM_CONSTANT(FullEngagement);
	BIND_ENUM_CONSTANT(Encirclement);
	BIND_ENUM_CONSTANT(Siege);

	ClassDB::bind_method(D_METHOD("is_engagement_happening"), &RCSEngagement::is_engagement_happening);
	ClassDB::bind_method(D_METHOD("is_engagement_over"), &RCSEngagement::is_engagement_over);
	ClassDB::bind_method(D_METHOD("get_scale"), &RCSEngagement::get_scale);

	ClassDB::bind_method(D_METHOD("get_involving_teams"), &RCSEngagement::get_involving_teams);
	ClassDB::bind_method(D_METHOD("get_involving_squads"), &RCSEngagement::get_involving_squads);
	ClassDB::bind_method(D_METHOD("get_offending_team"), &RCSEngagement::get_offending_team);
	ClassDB::bind_method(D_METHOD("get_deffending_teams"), &RCSEngagement::get_deffending_teams);
	ClassDB::bind_method(D_METHOD("get_offending_squads"), &RCSEngagement::get_offending_squads);
	ClassDB::bind_method(D_METHOD("get_deffending_squads"), &RCSEngagement::get_deffending_squads);
}

#pragma endregion

#pragma region Combatant
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

	ClassDB::bind_method(D_METHOD("set_combatant_attributes", "attr"), &RCSCombatantProfile::set_combatant_attributes);
	ClassDB::bind_method(D_METHOD("get_combatant_attributes"), &RCSCombatantProfile::get_combatant_attributes);

	ClassDB::bind_method(D_METHOD("set_acquisition_threshold", "new_threshold"), &RCSCombatantProfile::set_acquisition_threshold);
	ClassDB::bind_method(D_METHOD("get_acquisition_threshold"), &RCSCombatantProfile::get_acquisition_threshold);

	ClassDB::bind_method(D_METHOD("set_subvention", "new_subvention"), &RCSCombatantProfile::set_subvention);
	ClassDB::bind_method(D_METHOD("get_subvention"), &RCSCombatantProfile::get_subvention);

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

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "combatant_name"), "set_pname", "get_pname");
	// ADD_PROPERTY(PropertyInfo(Variant::INT, "combatant_id"), "set_id", "get_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stand"), "set_stand", "get_stand");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "attributes"), "set_combatant_attributes", "get_combatant_attributes");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_threshold", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_detection_threshold", "get_detection_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "acquisition_threshold", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_acquisition_threshold", "get_acquisition_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detection_subvention", PROPERTY_HINT_RANGE, "0.001,0.980,0.001"), "set_subvention", "get_subvention");
	// ADD_PROPERTY(PropertyInfo(Variant::REAL, "_detection_meter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_detection_meter", "_get_detection_meter");
}

RCSCombatant::RCSCombatant(){
	simulation = nullptr;
	my_squad = nullptr;
}

RCSCombatant::~RCSCombatant(){
	CleanerLog(RCSCombatant, 0);
	set_simulation(nullptr);
	CleanerLog(RCSCombatant, 1);
	set_squad(nullptr);
	CleanerLog(RCSCombatant, 2);
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

bool RCSCombatant::is_engagable(RCSCombatant* bogey){
	if (!bogey || !my_squad) return false;
	auto bogey_squad = bogey->get_squad();
	if (bogey_squad == my_squad) return false;
	return my_squad->is_engagable(bogey_squad);
}

void RCSCombatant::poll(const float& delta){
	RID_RCS::poll(delta);
	if (combatant_profile.is_null()) return;
	auto new_meter = detection_meter - (combatant_profile->get_subvention() * delta);
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
	// Ref<RawRecordData> rdata = memnew(RawRecordData);
	// rdata->name = StringName("__main");

	// PUSH_RECORD_PRIMITIVE(rdata, space_transform);
	// PUSH_RECORD_PRIMITIVE(rdata, local_transform);
	// // PUSH_RECORD_PRIMITIVE(rdata, stand);

	// rdata->external_refs.resize(rdata->table.size());
	// rdata->external_refs.fill(0);
	return rrec;
}
bool RCSCombatant::serialize(const Ref<RawRecord>& from){
	return false;
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
	// engagement_loggers = nullptr;
}

RCSSquad::~RCSSquad(){
	set_simulation(nullptr);
	set_team(nullptr);
	// if (engagement_loggers) engagement_loggers->cut_ties_squad(this);
	uint32_t i = 0;
	while (true) {
		CleanerLog(RCSSquad, i);
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

bool RCSSquad::is_engagable(RCSSquad *bogey){
	if (!bogey || !my_team) return false;
	auto bogey_team = bogey->get_team();
	// if (bogey_team == my_team) return false;
	return my_team->is_engagable(bogey_team);
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

	uint32_t i = 0;
	while (true) {
		uint16_t cond = 0;
		CleanerLog(RCSTeam, i);
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

bool RCSTeam::is_engagable(RCSTeam *bogey) const{
	if (!bogey || bogey == this) return false;
	auto relation = get_link_to(bogey);
	if (relation.is_null()) return false;
	return (relation->get_attributes() & RCSUnilateralTeamsBind::ITA_Engagable);
}

Array RCSTeam::get_engagements_ref() const {
	Array re;
	for (uint32_t i = 0, size = engagement_loggers.size(); i < size; i++){
		re.push_back(Variant(engagement_loggers[i]->spawn_reference()));
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

void RCSTeam::purge_all_links(){
	while (!team_binds.empty()){
		auto link = team_binds[0];
		link->toward->remove_link(this);
		VEC_REMOVE(team_binds, 0);
	}
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

	ClassDB::bind_method(D_METHOD("set_pname", "name"), &RCSRadarProfile::set_pname);
	ClassDB::bind_method(D_METHOD("get_pname"), &RCSRadarProfile::get_pname);

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

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "acquisition_curve", PROPERTY_HINT_RESOURCE_TYPE, "", PROPERTY_USAGE_DEFAULT, "AdvancedCurve"), "set_acurve", "get_acurve");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "detection_curve", PROPERTY_HINT_RESOURCE_TYPE, "", PROPERTY_USAGE_DEFAULT, "AdvancedCurve"), "set_dcurve", "get_dcurve");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "profile_name"), "set_pname", "get_pname");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_spread", PROPERTY_HINT_RANGE, "0.009,3.141,0.001"), "set_spread", "get_spread");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radar_frequency", PROPERTY_HINT_RANGE, "0.001,5.0,0.001"), "set_freq", "get_freq");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "scan_contribution", PROPERTY_HINT_RANGE, "0.0001,0.999,0.001"), "set_contribution", "get_contribution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "rays_per_degree", PROPERTY_HINT_RANGE, "0.1,5.0,0.1"), "set_rpd", "get_rpd");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_method", PROPERTY_HINT_ENUM, "ScanModeSwarm,ScanModeSingle"), "set_method", "get_method");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_base", PROPERTY_HINT_ENUM, "ScanTransform,ScanDirectSpaceState"), "set_base", "get_base");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scan_attributes"), "set_attr", "get_attr");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_cmask", "get_cmask");

	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_ACQUIRE_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, RADAR_PROFILE_LATE_DETECT_VMETHOD, PropertyInfo(Variant::_RID, "from_id"), PropertyInfo(Variant::_RID, "to_id"), PropertyInfo(Variant::TRANSFORM, "self_transform")));
}

void RCSRadarProfile::ping_target(RadarPingRequest* ping_request){
	auto cache = ping_cache(ping_request);
	auto script = get_script_instance();
	if (script && script->has_method(RADAR_PROFILE_DETECT_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_DETECT_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->detect_result = (bool)script_re;
	} else internal_detect(ping_request, cache);
	if (script && script->has_method(RADAR_PROFILE_ACQUIRE_VMETHOD)){
		auto script_re = script->call(RADAR_PROFILE_ACQUIRE_VMETHOD,
			ping_request->from->get_self(), ping_request->to->get_self(), ping_request->self_transform);
		if (script_re.get_type() == Variant::BOOL) ping_request->lock_result = (bool)script_re;
	} else internal_acquire(ping_request, cache);
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
RadarPingCache *RCSRadarProfile::ping_cache(RadarPingRequest* ping_request) const{
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
	auto bogey_profile = ping_request->to->get_profile();
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
		} default: { ERR_FAIL_MSG("Invalid scan method"); }
	}
}

void RCSRadarProfile::internal_acquire(RadarPingRequest* ping_request, RadarPingCache* cache){
	if (cache->distance > acquisition_curve->get_range()) return;
	// Check for bogey's bearing
	if (cache->bearing > spread) return;
	auto bogey_profile = ping_request->to->get_profile();
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

RCSRadar::RCSRadar(){
	simulation = nullptr;
	assigned_vessel = nullptr;
	space_state = nullptr;
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

void RCSRadar::fetch_space_state(){
	auto scene_tree = SceneTree::get_singleton();
	ERR_FAIL_COND(!scene_tree);
	auto root = scene_tree->get_root();
	if (!root) return;
	auto world = root->find_world();
	if (world.is_null()) return;
	space_state = world->get_direct_space_state();
}

void RCSRadar::ping_base_transform(const float& delta){
	auto combatants = simulation->get_combatants();
	auto c_size = combatants->size();
	for (uint32_t i = 0; i < c_size; i++){
		auto com = combatants->operator[](i);
		if (!com || !assigned_vessel->is_engagable(com)) continue;
		// ---------------------------------------------------
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
void RCSRadar::ping_base_direct_space_state(const float& delta){
	if (space_state == nullptr){
		fetch_space_state();
		ERR_FAIL_COND(space_state == nullptr);
	}
	// Rays per degree = ray count / degree count
	// Degrees * PI / 180 = Radians
	// Rays per radian = ray count / (degrees * PI / 180) = (180 * ray count) / (PI * degrees count)
	double rays_per_degree = rprofile->get_rpd();
	double rays_per_radian = (rays_per_degree * 180.0) / PI;
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
	bool collide_with_areas  = radar_attr & RCSRadarProfile::ScanCollideWithAreas;
	Set<RID_TYPE> exclude;
	for (double angle_offset = -ray_spread; angle_offset < ray_spread; ray_spread += radians_per_ray){
		Vector3 current_ray = vessel_forward.rotated(VEC3_UP, angle_offset);
		Vector3 scan_origin = vessel_origin + (current_ray * max_distance);
		PhysicsDirectSpaceState::RayResult ray_res;
		auto intersect_result = space_state->intersect_ray(vessel_origin, scan_origin, ray_res,
			exclude, radar_cmask, collide_with_bodies, collide_with_areas);
		if (!intersect_result) continue;
		auto obj_id = ray_res.collider_id;
		auto combatant = simulation->get_combatant_from_iid(obj_id);
		// If found object is not combatant,
		// it might be an obstacle, therefore, don't exclude it
		// This method might skip over some combatant,
		// but with enough ray density, it wouldn't be a problem
		if (!combatant) continue;
		// FIX: exclude the RID_TYPE regardless of the assertion result
		// if it make it all the way to this point ||
		//                                        \./
		if (!assigned_vessel->is_engagable(combatant)){

		} else {
			RadarPingRequest req(assigned_vessel, combatant);
			req.self_transform = assigned_vessel->get_combined_transform();
			rprofile->ping_target(&req);

			if (req.late_recheck){
				simulation->radar_request_recheck(new RadarRecheckTicket(this, combatant));
			} else if (req.detect_result) {
				detected_combatants.push_back(combatant);
				detected_rids.push_back(combatant->get_self());
			}
			if (req.lock_result){
				locked_combatants.push_back(combatant);
				locked_rids.push_back(combatant->get_self());
			}
		}
		exclude.insert(ray_res.rid);
	}
}

void RCSRadar::poll(const float& delta){
	RID_RCS::poll(delta);
	if (rprofile.is_null() || !simulation || !assigned_vessel) return;
	if (rprofile->get_acurve().is_null()  || rprofile->get_dcurve().is_null()) return;
	timer += delta;
	if (timer < ( 1.0 / rprofile->get_freq())) return;

	//-------------------------------------------------
	timer = 0.0;
	detected_combatants.clear();
	detected_rids.clear();
	locked_combatants.clear();
	locked_rids.clear();
	//--------------------------------------------------
	// Combatants pool size should stay the same during radars' iteration
	switch (rprofile->get_base()){
		case RCSRadarProfile::ScanTransform:
			ping_base_transform(delta); break;
		case RCSRadarProfile::ScanDirectSpaceState:
			ping_base_direct_space_state(delta); break;
		default:
			break;
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
#pragma endregion
