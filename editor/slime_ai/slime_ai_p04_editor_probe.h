#pragma once

#include "core/string/ustring.h"
#include "core/variant/dictionary.h"

class Node;

namespace SlimeAI {
class RunController;
class SceneTransaction;
class ServiceClient;

// Opt-in offline acceptance in a real EditorNode process on a copied fixture.
class P04EditorProbe {
	String probe_dir;
	String expected_scene;
	String original_hash;
	int original_children = 0;
	int stage = 0;
	void write_result(const String &p_name, const Dictionary &p_value) const;

public:
	P04EditorProbe();
	bool enabled() const { return !probe_dir.is_empty(); }
	void tick(ServiceClient &p_service, RunController &p_run, SceneTransaction &p_transaction, Node *p_root, bool p_editor_unsaved);
};
} // namespace SlimeAI
