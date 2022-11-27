#ifndef RID_RTSCOM_H
#define RID_RTSCOM_H

#include "modules/record/record.h"

#include "core/rid.h"
#include "core/string_name.h"
#include "core/hash_map.h"
#include "rtscom_meta.h"

#define PUSH_RECORD_PRIMITIVE(m_record, m_val)                                        \
	m_record->table.push_back(Pair<StringName, Variant>(StringName(#m_val), m_val))    \

class Sentrience;

#ifdef  USE_SAFE_RID_COUNT
class RID_RCS : public RID_Data {
protected:
	Ref<RCSChip> chip;
	RID_TYPE self;
	Sentrience *combat_server;
public:
	virtual Ref<RawRecord> serialize() const { return Ref<RawRecord>(); }
	virtual bool serialize(const Ref<RawRecord>& from) { return false; }
	virtual void deserialize(const RawRecord& rec) {}

	friend class Sentrience;

	_FORCE_INLINE_ void set_self(const RID_TYPE &p_self) { self = p_self; }
	_FORCE_INLINE_ RID_TYPE get_self() const { return self; }

	_FORCE_INLINE_ void _set_combat_server(Sentrience *p_combatServer) { combat_server = p_combatServer; }
	_FORCE_INLINE_ Sentrience *_get_combat_server() const { return combat_server; }

	virtual void capture_event(RID_RCS* from, void* event = nullptr){}

	virtual void poll(const float& delta) {
		if (chip.is_valid()) 
			chip->callback(delta);
		}

	virtual void set_chip(const Ref<RCSChip>& new_chip) { chip = new_chip; if (chip.is_valid()) chip->set_host(self); }
	virtual Ref<RCSChip> get_chip() const { return chip; }
};
#else

class RID_RCS : public RID_Data {
protected:
	Ref<RCSChip> chip;
	RID_TYPE self;
	Sentrience *combat_server;
public:
	virtual Ref<RawRecord> serialize() const { return Ref<RawRecord>(); }
	virtual bool serialize(const Ref<RawRecord>& from) { return false; }
	virtual void deserialize(const RawRecord& rec) {}

	friend class Sentrience;

	_FORCE_INLINE_ void set_self(const RID_TYPE &p_self) { self = p_self; }
	_FORCE_INLINE_ RID_TYPE get_self() const { return self; }

	_FORCE_INLINE_ void _set_combat_server(Sentrience *p_combatServer) { combat_server = p_combatServer; }
	_FORCE_INLINE_ Sentrience *_get_combat_server() const { return combat_server; }

	virtual void capture_event(RID_RCS* from, void* event = nullptr){}

	virtual void poll(const float& delta) {
		if (chip.is_valid()) 
			chip->callback(delta);
		}

	virtual void set_chip(const Ref<RCSChip>& new_chip) { chip = new_chip; if (chip.is_valid()) chip->set_host(self); }
	virtual Ref<RCSChip> get_chip() const { return chip; }
};
#endif

#endif
