#pragma once

#include "core/string/ustring.h"
#include "core/variant/dictionary.h"

class Node;

namespace SlimeAI {
class SceneTransaction;

// Opt-in editor integration probe. A parent harness controls the real file
// lock; this code uses the same transaction and save path as the dock.
class SaveFailureProbe {
	String probe_dir;
	String expected_scene;
	String operation_id;
	String previous_disk_hash;
	Dictionary original_proposal;
	int initial_child_count = 0;
	int stage = 0;

	void write_result(const String &p_name, const Dictionary &p_value) const;

public:
	SaveFailureProbe();
	bool enabled() const { return !probe_dir.is_empty(); }
	void tick(SceneTransaction &p_transaction, Node *p_root, bool p_editor_unsaved);
};
} // namespace SlimeAI
