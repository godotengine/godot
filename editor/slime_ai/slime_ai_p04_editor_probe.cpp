#include "slime_ai_p04_editor_probe.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "editor/editor_interface.h"
#include "editor/slime_ai/slime_ai_run_controller.h"
#include "editor/slime_ai/slime_ai_scene_transaction.h"
#include "editor/slime_ai/slime_ai_service_client.h"
#include "editor/slime_ai/slime_ai_code_reader.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/main/node.h"
#include "scene/gui/text_edit.h"

namespace SlimeAI {

P04EditorProbe::P04EditorProbe() {
	if (OS::get_singleton()->get_environment("SLIME_AI_TEST_P04_EDITOR_PROBE") != "1") {
		return;
	}
	probe_dir = OS::get_singleton()->get_environment("SLIME_AI_P04_PROBE_DIR");
	expected_scene = OS::get_singleton()->get_environment("SLIME_AI_P04_PROBE_SCENE");
	if (probe_dir.is_empty() || expected_scene.is_empty()) {
		probe_dir.clear();
	}
}

void P04EditorProbe::write_result(const String &p_name, const Dictionary &p_value) const {
	Ref<FileAccess> file = FileAccess::open(probe_dir.path_join(p_name), FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_string(JSON::stringify(p_value, "  "));
		file->flush();
	}
}

void P04EditorProbe::tick(ServiceClient &p_service, RunController &p_run, SceneTransaction &p_transaction, Node *p_root, bool p_editor_unsaved) {
	if (!enabled() || !p_root || p_root->get_scene_file_path() != expected_scene || stage == 9) {
		return;
	}
	if (stage == 0) {
		ScriptEditor *script_editor = ScriptEditor::get_singleton();
		Dictionary read;
		if (!script_editor) {
			read["status"] = "not_run_no_script_editor_in_headless_process";
		} else {
			Ref<Resource> script_resource = ResourceLoader::load("res://read_probe.gd");
			if (script_resource.is_null()) {
				read["status"] = "not_run_script_resource_unavailable";
			} else {
				EditorSettings *editor_settings = EditorSettings::get_singleton();
				const Variant previous_external_editor = editor_settings ? editor_settings->get_setting("text_editor/external/use_external_editor") : Variant(false);
				if (editor_settings) {
					editor_settings->set_setting("text_editor/external/use_external_editor", false);
					script_editor->notification(EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED);
				}
				const bool opened = script_editor->edit(script_resource, false);
				if (editor_settings) {
					editor_settings->set_setting("text_editor/external/use_external_editor", previous_external_editor);
					script_editor->notification(EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED);
				}
				if (!opened) {
					read["status"] = "not_run_script_editor_edit_rejected";
				}
				if (opened) {
				ScriptEditorBase *script_tab = script_editor->get_resource_editor(script_resource);
				TextEdit *text_editor = script_tab ? Object::cast_to<TextEdit>(script_tab->get_base_editor()) : nullptr;
				if (!text_editor) {
					read["status"] = "not_run_script_text_editor_unavailable";
				} else {
					const String original_text = text_editor->get_text();
					text_editor->set_text(original_text.replace("saved value", "unsaved value"));
					read = CodeReader::read("read_probe.gd", 1, 10);
					read["script_editor_unsaved"] = script_tab->is_unsaved();
					text_editor->set_text(original_text);
					script_tab->tag_saved_version();
				}
				}
			}
		}
		write_result("read.json", read);
		original_hash = FileAccess::get_sha256(expected_scene);
		original_children = p_root->get_child_count();
		const String script = OS::get_singleton()->get_executable_path().get_base_dir().get_base_dir().path_join("tools/slime_ai/agent_service/src/main.ts");
		if (!p_service.start(script)) {
			Dictionary failed;
			failed["status"] = "service_unavailable";
			failed["error"] = p_service.get_last_error();
			write_result("error.json", failed);
			stage = 9;
			return;
		}
		stage = 1;
	}
	Vector<Dictionary> frames;
	p_service.poll(frames);
	for (const Dictionary &frame : frames) {
		if (String(frame.get("protocol_version", "")) == "1.1") {
			const Dictionary handled = p_run.on_frame(p_root, p_editor_unsaved, frame);
			if (handled.has("error")) {
				write_result("last_event_error.json", handled);
			}
		}
	}
	if (stage == 1 && p_service.is_ready()) {
		const Dictionary started = p_run.start(p_root, p_editor_unsaved, "fake", "", "discuss", "Describe this selected scene without changes", false);
		if (String(started.get("status", "")) != "waiting_for_provider") {
			write_result("error.json", started);
			stage = 9;
			return;
		}
		stage = 2;
		return;
	}
	if (stage == 2 && String(p_run.status(p_root).get("status", "")) == "completed") {
		Dictionary discuss;
		discuss["run"] = p_run.status(p_root);
		discuss["no_edit"] = p_root->get_child_count() == original_children && FileAccess::get_sha256(expected_scene) == original_hash;
		write_result("discuss.json", discuss);
		const Dictionary started = p_run.start(p_root, p_editor_unsaved, "fake", "", "propose", "Preview one native marker", false);
		if (String(started.get("status", "")) != "waiting_for_provider") {
			write_result("error.json", started);
			stage = 9;
			return;
		}
		stage = 3;
		return;
	}
	if (stage == 3 && String(p_run.status(p_root).get("status", "")) == "completed") {
		stage = 4; // Save can reenter editor notifications; never run Execute twice.
		Dictionary propose;
		propose["run"] = p_run.status(p_root);
		propose["preview_id"] = p_run.get_preview_id();
		propose["no_edit"] = p_root->get_child_count() == original_children && FileAccess::get_sha256(expected_scene) == original_hash;
		propose["apply_denied"] = p_run.apply(p_root, p_editor_unsaved);
		write_result("propose.json", propose);
		if (p_run.get_preview_id().is_empty()) {
			stage = 9;
			return;
		}
		const Dictionary transition = p_run.transition_to_execute(p_root, p_editor_unsaved);
		const Dictionary grant = p_run.grant();
		const Dictionary applied = p_run.apply(p_root, p_editor_unsaved);
		const Dictionary undo = p_transaction.undo(p_root);
		const int children_after_undo = p_root->get_child_count();
		const Dictionary redo = p_transaction.redo(p_root);
		const Error save_error = EditorInterface::get_singleton()->save_scene();
		Dictionary execute;
		execute["transition"] = transition;
		execute["grant"] = grant;
		execute["apply"] = applied;
		execute["undo"] = undo;
		execute["children_after_undo"] = children_after_undo;
		execute["redo"] = redo;
		execute["save_error"] = save_error;
		execute["disk_before"] = original_hash;
		execute["disk_after"] = FileAccess::get_sha256(expected_scene);
		execute["children_after_redo"] = p_root->get_child_count();
		execute["native_operation_status"] = p_transaction.status(p_root, p_run.get_operation_id());
		execute["gameplay_verification"] = "not_run";
		write_result("execute.json", execute);
		stage = 9;
	}
}

} // namespace SlimeAI
