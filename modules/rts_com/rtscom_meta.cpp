#include "rtscom_meta.h"

#define RCSC_VMETHOD StringName("_iterate")
#define RCSC_BOOT_VMETHOD StringName("_boot")

RCSChip::RCSChip() {}
RCSChip::~RCSChip(){}

void RCSChip::_bind_methods(){
	ClassDB::bind_method(D_METHOD("get_host"), &RCSChip::get_host);
	BIND_VMETHOD(MethodInfo(Variant::NIL, RCSC_VMETHOD, PropertyInfo(Variant::REAL, "delta")));
	BIND_VMETHOD(MethodInfo(Variant::NIL, RCSC_BOOT_VMETHOD));
}

void RCSChip::callback(const float& delta){
	internal_callback(delta);
	auto script = get_script_instance();
	if (script) script->call(RCSC_VMETHOD, Variant(delta));
}

void RCSChip::set_host(const RID& r_host){
	host = r_host;
	auto script = get_script_instance();
	if (script) script->call(RCSC_BOOT_VMETHOD);
}
