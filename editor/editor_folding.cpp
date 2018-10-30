#include "editor_folding.h"

#include "core/os/file_access.h"
#include "editor_settings.h"

PoolVector<String> EditorFolding::_get_unfolds(const Object *p_object) {

	PoolVector<String> sections;
	sections.resize(p_object->editor_get_section_folding().size());
	if (sections.size()) {
		PoolVector<String>::Write w = sections.write();
		int idx = 0;
		for (const Set<String>::Element *E = p_object->editor_get_section_folding().front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	return sections;
}

void EditorFolding::save_resource_folding(const RES &p_resource, const String &p_path) {
	Ref<ConfigFile> config;
	config.instance();
	PoolVector<String> unfolds = _get_unfolds(p_resource.ptr());
	config->set_value("folding", "sections_unfolded", unfolds);

	String path = EditorSettings::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorSettings::get_singleton()->get_project_settings_dir().plus_file(file);
	config->save(file);
}

void EditorFolding::_set_unfolds(Object *p_object, const PoolVector<String> &p_unfolds) {

	int uc = p_unfolds.size();
	PoolVector<String>::Read r = p_unfolds.read();
	p_object->editor_clear_section_folding();
	for (int i = 0; i < uc; i++) {
		p_object->editor_set_section_unfold(r[i], true);
	}
}

void EditorFolding::load_resource_folding(RES p_resource, const String &p_path) {

	Ref<ConfigFile> config;
	config.instance();

	String path = EditorSettings::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorSettings::get_singleton()->get_project_settings_dir().plus_file(file);

	if (config->load(file) != OK) {
		return;
	}

	PoolVector<String> unfolds;

	if (config->has_section_key("folding", "sections_unfolded")) {
		unfolds = config->get_value("folding", "sections_unfolded");
	}
	_set_unfolds(p_resource.ptr(), unfolds);
}

void EditorFolding::_fill_folds(const Node *p_root, const Node *p_node, Array &p_folds, Array &resource_folds, Set<RES> &resources) {
	if (p_root != p_node) {
		if (!p_node->get_owner()) {
			return; //not owned, bye
		}
		if (p_node->get_owner() != p_root && !p_root->is_editable_instance(p_node)) {
			return;
		}
	}

	PoolVector<String> unfolds = _get_unfolds(p_node);

	if (unfolds.size()) {
		p_folds.push_back(p_root->get_path_to(p_node));
		p_folds.push_back(unfolds);
	}

	List<PropertyInfo> plist;
	p_node->get_property_list(&plist);
	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {
		if (E->get().type == Variant::OBJECT) {
			RES res = p_node->get(E->get().name);
			if (res.is_valid() && !resources.has(res) && res->get_path() != String() && !res->get_path().is_resource_file()) {

				PoolVector<String> res_unfolds = _get_unfolds(res.ptr());
				resource_folds.push_back(res->get_path());
				resource_folds.push_back(res_unfolds);
				resources.insert(res);
			}
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_fill_folds(p_root, p_node->get_child(i), p_folds, resource_folds, resources);
	}
}
void EditorFolding::save_scene_folding(const Node *p_scene, const String &p_path) {

	Ref<ConfigFile> config;
	config.instance();

	Array unfolds, res_unfolds;
	Set<RES> resources;
	_fill_folds(p_scene, p_scene, unfolds, res_unfolds, resources);

	config->set_value("folding", "node_unfolds", unfolds);
	config->set_value("folding", "resource_unfolds", res_unfolds);

	String path = EditorSettings::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorSettings::get_singleton()->get_project_settings_dir().plus_file(file);
	config->save(file);
}
void EditorFolding::load_scene_folding(Node *p_scene, const String &p_path) {

	Ref<ConfigFile> config;
	config.instance();

	String path = EditorSettings::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorSettings::get_singleton()->get_project_settings_dir().plus_file(file);

	if (config->load(file) != OK) {
		return;
	}

	Array unfolds;
	if (config->has_section_key("folding", "node_unfolds")) {
		unfolds = config->get_value("folding", "node_unfolds");
	}
	Array res_unfolds;
	if (config->has_section_key("folding", "resource_unfolds")) {
		res_unfolds = config->get_value("folding", "resource_unfolds");
	}

	ERR_FAIL_COND(unfolds.size() & 1);
	ERR_FAIL_COND(res_unfolds.size() & 1);

	for (int i = 0; i < unfolds.size(); i += 2) {
		NodePath path = unfolds[i];
		PoolVector<String> un = unfolds[i + 1];
		Node *node = p_scene->get_node(path);
		if (!node) {
			continue;
		}
		_set_unfolds(node, un);
	}

	for (int i = 0; i < res_unfolds.size(); i += 2) {
		String path = res_unfolds[i];
		RES res;
		if (ResourceCache::has(path)) {
			res = RES(ResourceCache::get(path));
		}
		if (res.is_null()) {
			continue;
		}

		PoolVector<String> unfolds = res_unfolds[i + 1];
		_set_unfolds(res.ptr(), unfolds);
	}
}

bool EditorFolding::has_folding_data(const String &p_path) {
	String path = EditorSettings::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorSettings::get_singleton()->get_project_settings_dir().plus_file(file);
	return FileAccess::exists(file);
}

EditorFolding::EditorFolding() {
}
