#ifndef BT_STRING_NAMES_H
#define BT_STRING_NAMES_H

#include "string_db.h"

class BtStringNames {

friend void register_behavior_tree_types();
friend void unregister_behavior_tree_types();

	static BtStringNames* singleton;

	static void create() { singleton = memnew(BtStringNames); }
	static void free() { memdelete( singleton); singleton=NULL; }

	BtStringNames();

public:
	_FORCE_INLINE_ static BtStringNames* get_singleton() { return singleton; }

	StringName _continue;
	StringName _prepare;
	StringName _self_update;
	StringName _child_update;
	StringName _update;
	StringName _pre_update;
	StringName _post_update;
	StringName _abort;

};

#endif
