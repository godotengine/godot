#include "resource_saver_zig.h"
#include "../elf/script_elf.h"
#include "../elf/script_language_elf.h"
#include "../register_types.h"
#include "../sandbox_project_settings.h"
#include "script_zig.h"
#include <godot_cpp/classes/editor_file_system.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_settings.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

static Ref<ResourceFormatSaverZig> zig_saver;
static constexpr bool VERBOSE_CMD = false;

void ResourceFormatSaverZig::init() {
	zig_saver.instantiate();
	ResourceSaver::get_singleton()->add_resource_format_saver(zig_saver);
}

void ResourceFormatSaverZig::deinit() {
	ResourceSaver::get_singleton()->remove_resource_format_saver(zig_saver);
	zig_saver.unref();
}

Error ResourceFormatSaverZig::_save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	ZigScript *script = Object::cast_to<ZigScript>(p_resource.ptr());
	if (script != nullptr) {
		Ref<FileAccess> handle = FileAccess::open(p_path, FileAccess::ModeFlags::WRITE);
		if (handle.is_valid()) {
			handle->store_string(script->_get_source_code());
			handle->close();
			// Get the absolute path without the file name
			String path = handle->get_path().get_base_dir().replace("res://", "") + "/";
			String inpname = path + "*.zig";
			String foldername = Docker::GetFolderName(handle->get_path().get_base_dir());
			String outname = path + foldername + String(".elf");

			// Lazily start the docker container
			ZigScript::DockerContainerStart();

			auto builder = [inpname = std::move(inpname), outname = std::move(outname)] {
				// Invoke docker to compile the file
				Array output;
				ZigScript::DockerContainerExecute({ "/usr/api/build.sh", "-o", outname, inpname }, output);
				if (!output.is_empty() && !output[0].operator String().is_empty()) {
					for (int i = 0; i < output.size(); i++) {
						String line = output[i].operator String();
						if constexpr (VERBOSE_CMD)
							ERR_PRINT(line);
						WARN_PRINT(line);
					}
				}
			};
			builder();
			// EditorInterface::get_singleton()->get_editor_settings()->set("text_editor/behavior/files/auto_reload_scripts_on_external_change", true);
			EditorInterface::get_singleton()->get_resource_filesystem()->scan();
			TypedArray<Script> open_scripts = EditorInterface::get_singleton()->get_script_editor()->get_open_scripts();
			for (int i = 0; i < open_scripts.size(); i++) {
				ELFScript *elf_script = Object::cast_to<ELFScript>(open_scripts[i]);
				if (elf_script) {
					elf_script->reload(false);
					elf_script->emit_changed();
				}
			}
			return Error::OK;
		} else {
			return Error::ERR_FILE_CANT_OPEN;
		}
	}
	return Error::ERR_SCRIPT_FAILED;
}
Error ResourceFormatSaverZig::_set_uid(const String &p_path, int64_t p_uid) {
	return Error::OK;
}
bool ResourceFormatSaverZig::_recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<ZigScript>(p_resource.ptr()) != nullptr;
}
PackedStringArray ResourceFormatSaverZig::_get_recognized_extensions(const Ref<Resource> &p_resource) const {
	PackedStringArray array;
	if (Object::cast_to<ZigScript>(p_resource.ptr()) == nullptr)
		return array;
	array.push_back("zig");
	return array;
}
bool ResourceFormatSaverZig::_recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return Object::cast_to<ZigScript>(p_resource.ptr()) != nullptr;
}
