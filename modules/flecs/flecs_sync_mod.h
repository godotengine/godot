#ifndef FLECS_SYNC_MOD_H
#define FLECS_SYNC_MOD_H

#include "flecs_mod.h"

class FlecsSyncMod : public FlecsMod {
	GDCLASS(FlecsSyncMod, FlecsMod);

public:

	FlecsMod::ModuleSyncDirection get_sync_direction() const;
	void set_sync_direction(FlecsMod::ModuleSyncDirection p_sync_direction);


protected:
	static void _bind_methods();

protected:

	FlecsMod::ModuleSyncDirection sync_direction = FlecsMod::ModuleSyncDirection::NONE;

};

#endif // FLECS_SYNC_MOD_H