#ifndef RTSCOM_META_H
#define RTSCOM_META_H

#include "core/reference.h"
#include "core/string_name.h"
#include "core/script_language.h"

enum CombatantStand {
	CS_NA		= 0,
	Movable		= 1,
	Immovable	= 2,
	Passive		= 4,
	Defensive	= 8,
	Aggressive	= 16,
	CS_MAX,
};

enum CombatantStatus {
	UNINITIALIZED,
	STANDBY,
	ACTIVE,
	DESTROYED,
};

class RID_RCS;

class RCSChip : public Reference{
	GDCLASS(RCSChip, Reference);
private:
	RID host;

	void callback(const float& delta);
protected:
	virtual void internal_callback(const float& delta) {};
	virtual void internal_init() {};
	static void _bind_methods();
public:
	RCSChip();
	~RCSChip();

	friend class RID_RCS;

	void set_host(const RID& r_host);
	_FORCE_INLINE_ RID get_host() const { return host; }
};

#endif
