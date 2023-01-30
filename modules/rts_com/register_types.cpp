#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "combat_server.h"

static Sentrience* RTSCombatServerPtr = nullptr;
static RCSMemoryAllocation* RCSMemoryAllocationPtr = nullptr;

Tuple<int, char, float, double, int, char, float, double> test_tuple_B(){
	return Tuple<int, char, float, double, int, char, float, double>
				(1, 2, 3, 4.4, 5, 6, 7, 8.8);
}
Tuple<int, char, float, double> test_tuple_C(){
	return Tuple<int, char, float, double>
				(1, 2, 3, 4.4);
}

void test_tuple_A(){
	auto tuple = test_tuple_B();

	ERR_FAIL_COND(tuple.size() 						!= 8);

	ERR_FAIL_COND(tuple.get_A() 					!= 1);
	ERR_FAIL_COND(tuple.get_B() 					!= 2);
	ERR_FAIL_COND(tuple.get_C() 					!= 3);
	ERR_FAIL_COND(tuple.get_D() 					!= 4.4);
	ERR_FAIL_COND(tuple.get_E() 					!= 5);
	ERR_FAIL_COND(tuple.get_F() 					!= 6);
	ERR_FAIL_COND(tuple.get_G() 					!= 7);
	ERR_FAIL_COND(tuple.get_H() 					!= 8.8);
	ERR_FAIL_COND(*(const int*)(tuple[0])   		!= 1);
	ERR_FAIL_COND(*(const char*)(tuple[1])   		!= 2);
	ERR_FAIL_COND(*(const float*)(tuple[2])  		!= 3);
	ERR_FAIL_COND(*(const double*)(tuple[3])  		!= 4.4);
	ERR_FAIL_COND(*(const int*)(tuple[4])   		!= 5);
	ERR_FAIL_COND(*(const char*)(tuple[5])   		!= 6);
	ERR_FAIL_COND(*(const float*)(tuple[6])   		!= 7);
	ERR_FAIL_COND(*(const double*)(tuple[7])   		!= 8.8);

	auto another = test_tuple_C();
	CRASH_COND(another.size() != 4);
}

void register_rts_com_types(){
#ifndef USE_THREAD_SAFE_API
	ClassDB::register_class<SentrienceContext>();
#endif
	ClassDB::register_class<Sentrience>();
	ClassDB::register_class<RCSChip>();
	ClassDB::register_class<RCSEngagement>();
	ClassDB::register_class<RCSProfile>();
	ClassDB::register_class<RCSSimulationProfile>();
	ClassDB::register_class<RCSRadarProfile>();
	ClassDB::register_class<RCSSpatialProfile>();
	ClassDB::register_class<RCSCombatantProfile>();
	ClassDB::register_class<RCSSquadProfile>();
	ClassDB::register_class<RCSProjectileProfile>();
	ClassDB::register_class<RCSUnilateralTeamsBind>();
	RCSMemoryAllocationPtr = memnew(RCSMemoryAllocation);
	RTSCombatServerPtr = memnew(Sentrience);
	Sentrience::set_primary_instance(RTSCombatServerPtr);
	Engine::get_singleton()->add_singleton(Engine::Singleton("Sentrience", Sentrience::get_singleton()));

	// test_tuple_A();
}
void unregister_rts_com_types(){
	memdelete(RTSCombatServerPtr);
	memdelete(RCSMemoryAllocationPtr);
	Sentrience::set_primary_instance(nullptr);
}
