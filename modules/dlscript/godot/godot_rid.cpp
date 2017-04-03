#include "godot_rid.h"

#include "object.h"
#include "resource.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rid_api_anchor() {
}

void GDAPI godot_rid_new(godot_rid *p_rid, godot_object *p_from) {

	Resource *res_from = ((Object *)p_from)->cast_to<Resource>();

	RID *rid = (RID *)p_rid;
	memnew_placement(rid, RID);

	if (res_from) {
		*rid = RID(res_from->get_rid());
	}
}

uint32_t GDAPI godot_rid_get_rid(const godot_rid *p_rid) {
	RID *rid = (RID *)p_rid;
	return rid->get_id();
}

void GDAPI godot_rid_destroy(godot_rid *p_rid) {
	((RID *)p_rid)->~RID();
}

#ifdef __cplusplus
}
#endif
