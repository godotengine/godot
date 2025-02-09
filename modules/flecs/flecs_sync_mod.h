#ifndef FLECS_SYNC_MOD_H
#define FLECS_SYNC_MOD_H

#include "flecs_mod.h"

class FlecsSyncMod : public FlecsMod {
	GDCLASS(FlecsSyncMod, FlecsMod);

public:
	ModuleSyncDirection get_sync_direction() const {return sync_direction;}
	void set_sync_direction(ModuleSyncDirection p_sync_direction) {sync_direction = p_sync_direction;}

protected:
	static void _bind_methods();

protected:
	ModuleSyncDirection sync_direction = ModuleSyncDirection::NONE;
};

#endif // FLECS_SYNC_MOD_H