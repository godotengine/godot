/*************************************************************************/
/*  gdnative_library_editor_plugin.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifdef TOOLS_ENABLED
#include "gdnative_library_editor_plugin.h"
#include "gdnative.h"

#include "editor/editor_scale.h"

void GDNativeLibraryEditor::edit(Ref<GDNativeLibrary> p_library) {
	library = p_library;
	Ref<ConfigFile> config = p_library->get_config_file();

	for (KeyValue<String, NativePlatformConfig> &E : platforms) {
		for (List<String>::Element *it = E.value.entries.front(); it; it = it->next()) {
			String target = E.key + "." + it->get();
			TargetConfig ecfg;
			ecfg.library = config->get_value("entry", target, "");
			ecfg.dependencies = config->get_value("dependencies", target, Array());
			entry_configs[target] = ecfg;
		}
	}

	_update_tree();
}

void GDNativeLibraryEditor::_bind_methods() {
}

void GDNativeLibraryEditor::_update_tree() {
	tree->clear();
	TreeItem *root = tree->create_item();

	PopupMenu *filter_list = filter->get_popup();
	String text = "";
	for (int i = 0; i < filter_list->get_item_count(); i++) {
		if (!filter_list->is_item_checked(i)) {
			continue;
		}
		Map<String, NativePlatformConfig>::Element *E = platforms.find(filter_list->get_item_metadata(i));
		if (!text.is_empty()) {
			text += ", ";
		}
		text += E->get().name;

		TreeItem *platform = tree->create_item(root);
		platform->set_text(0, E->get().name);
		platform->set_metadata(0, E->get().library_extension);

		platform->set_custom_bg_color(0, get_theme_color(SNAME("prop_category"), SNAME("Editor")));
		platform->set_custom_bg_color(1, get_theme_color(SNAME("prop_category"), SNAME("Editor")));
		platform->set_custom_bg_color(2, get_theme_color(SNAME("prop_category"), SNAME("Editor")));
		platform->set_selectable(0, false);
		platform->set_expand_right(0, true);

		for (List<String>::Element *it = E->value().entries.front(); it; it = it->next()) {
			String target = E->key() + "." + it->get();
			TreeItem *bit = tree->create_item(platform);

			bit->set_text(0, it->get());
			bit->set_metadata(0, target);
			bit->set_selectable(0, false);
			bit->set_custom_bg_color(0, get_theme_color(SNAME("prop_subsection"), SNAME("Editor")));

			bit->add_button(1, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")), BUTTON_SELECT_LIBRARY, false, TTR("Select the dynamic library for this entry"));
			String file = entry_configs[target].library;
			if (!file.is_empty()) {
				bit->add_button(1, get_theme_icon(SNAME("Clear"), SNAME("EditorIcons")), BUTTON_CLEAR_LIBRARY, false, TTR("Clear"));
			}
			bit->set_text(1, file);

			bit->add_button(2, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")), BUTTON_SELECT_DEPENDENCES, false, TTR("Select dependencies of the library for this entry"));
			Array files = entry_configs[target].dependencies;
			if (files.size()) {
				bit->add_button(2, get_theme_icon(SNAME("Clear"), SNAME("EditorIcons")), BUTTON_CLEAR_DEPENDENCES, false, TTR("Clear"));
			}
			bit->set_text(2, Variant(files));

			bit->add_button(3, get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")), BUTTON_MOVE_UP, false, TTR("Move Up"));
			bit->add_button(3, get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")), BUTTON_MOVE_DOWN, false, TTR("Move Down"));
			bit->add_button(3, get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")), BUTTON_ERASE_ENTRY, false, TTR("Remove current entry"));
		}

		TreeItem *new_arch = tree->create_item(platform);
		new_arch->set_text(0, TTR("Double click to create a new entry"));
		new_arch->set_text_alignment(0, HORIZONTAL_ALIGNMENT_CENTER);
		new_arch->set_custom_color(0, get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		new_arch->set_expand_right(0, true);
		new_arch->set_metadata(1, E->key());

		platform->set_collapsed(collapsed_items.find(E->get().name) != nullptr);
	}
	filter->set_text(text);
}

void GDNativeLibraryEditor::_on_item_button(Object *item, int column, int id) {
	String target = Object::cast_to<TreeItem>(item)->get_metadata(0);
	String platform = target.substr(0, target.find("."));
	String entry = target.substr(platform.length() + 1, target.length());
	String section = (id == BUTTON_SELECT_DEPENDENCES || id == BUTTON_CLEAR_DEPENDENCES) ? "dependencies" : "entry";

	if (id == BUTTON_SELECT_LIBRARY || id == BUTTON_SELECT_DEPENDENCES) {
		TreeItem *treeItem = Object::cast_to<TreeItem>(item)->get_parent();
		EditorFileDialog::FileMode mode = EditorFileDialog::FILE_MODE_OPEN_FILE;
		if (id == BUTTON_SELECT_DEPENDENCES) {
			mode = EditorFileDialog::FILE_MODE_OPEN_FILES;
		} else if (treeItem->get_text(0) == "iOS" || treeItem->get_text(0) == "macOS") {
			mode = EditorFileDialog::FILE_MODE_OPEN_ANY;
		}

		file_dialog->set_meta("target", target);
		file_dialog->set_meta("section", section);
		file_dialog->clear_filters();

		String filter_string = treeItem->get_metadata(0);
		Vector<String> filters = filter_string.split(",", false, 0);
		for (int i = 0; i < filters.size(); i++) {
			file_dialog->add_filter(filters[i]);
		}

		file_dialog->set_file_mode(mode);
		file_dialog->popup_file_dialog();

	} else if (id == BUTTON_CLEAR_LIBRARY) {
		_set_target_value(section, target, "");
	} else if (id == BUTTON_CLEAR_DEPENDENCES) {
		_set_target_value(section, target, Array());
	} else if (id == BUTTON_ERASE_ENTRY) {
		_erase_entry(platform, entry);
	} else if (id == BUTTON_MOVE_UP || id == BUTTON_MOVE_DOWN) {
		_move_entry(platform, entry, id);
	}
}

void GDNativeLibraryEditor::_on_library_selected(const String &file) {
	_set_target_value(file_dialog->get_meta("section"), file_dialog->get_meta("target"), file);
}

void GDNativeLibraryEditor::_on_dependencies_selected(const PackedStringArray &files) {
	_set_target_value(file_dialog->get_meta("section"), file_dialog->get_meta("target"), files);
}

void GDNativeLibraryEditor::_on_filter_selected(int index) {
	PopupMenu *filter_list = filter->get_popup();
	filter_list->set_item_checked(index, !filter_list->is_item_checked(index));
	_update_tree();
}

void GDNativeLibraryEditor::_on_item_collapsed(Object *p_item) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	String name = item->get_text(0);

	if (item->is_collapsed()) {
		collapsed_items.insert(name);
	} else if (Set<String>::Element *e = collapsed_items.find(name)) {
		collapsed_items.erase(e);
	}
}

void GDNativeLibraryEditor::_on_item_activated() {
	TreeItem *item = tree->get_selected();
	if (item && tree->get_selected_column() == 0 && item->get_metadata(0).get_type() == Variant::NIL) {
		new_architecture_dialog->set_meta("platform", item->get_metadata(1));
		new_architecture_dialog->popup_centered();
	}
}

void GDNativeLibraryEditor::_on_create_new_entry() {
	String platform = new_architecture_dialog->get_meta("platform");
	String entry = new_architecture_input->get_text().strip_edges();
	if (!entry.is_empty()) {
		platforms[platform].entries.push_back(entry);
		_update_tree();
	}
}

void GDNativeLibraryEditor::_set_target_value(const String &section, const String &target, Variant file) {
	if (section == "entry") {
		entry_configs[target].library = file;
	} else if (section == "dependencies") {
		entry_configs[target].dependencies = file;
	}
	_translate_to_config_file();
	_update_tree();
}

void GDNativeLibraryEditor::_erase_entry(const String &platform, const String &entry) {
	if (platforms.has(platform)) {
		if (List<String>::Element *E = platforms[platform].entries.find(entry)) {
			String target = platform + "." + entry;

			platforms[platform].entries.erase(E);
			_set_target_value("entry", target, "");
			_set_target_value("dependencies", target, Array());
			_translate_to_config_file();
			_update_tree();
		}
	}
}

void GDNativeLibraryEditor::_move_entry(const String &platform, const String &entry, int dir) {
	if (List<String>::Element *E = platforms[platform].entries.find(entry)) {
		if (E->prev() && dir == BUTTON_MOVE_UP) {
			platforms[platform].entries.insert_before(E->prev(), E->get());
			platforms[platform].entries.erase(E);
		} else if (E->next() && dir == BUTTON_MOVE_DOWN) {
			platforms[platform].entries.insert_after(E->next(), E->get());
			platforms[platform].entries.erase(E);
		}
		_translate_to_config_file();
		_update_tree();
	}
}

void GDNativeLibraryEditor::_translate_to_config_file() {
	if (!library.is_null()) {
		Ref<ConfigFile> config = library->get_config_file();
		config->erase_section("entry");
		config->erase_section("dependencies");

		for (KeyValue<String, NativePlatformConfig> &E : platforms) {
			for (List<String>::Element *it = E.value.entries.front(); it; it = it->next()) {
				String target = E.key + "." + it->get();
				if (entry_configs[target].library.is_empty() && entry_configs[target].dependencies.is_empty()) {
					continue;
				}

				config->set_value("entry", target, entry_configs[target].library);
				config->set_value("dependencies", target, entry_configs[target].dependencies);
			}
		}

		library->notify_property_list_changed();
	}
}

GDNativeLibraryEditor::GDNativeLibraryEditor() {
	{ // Define platforms
		NativePlatformConfig platform_windows;
		platform_windows.name = "Windows";
		platform_windows.entries.push_back("64");
		platform_windows.entries.push_back("32");
		platform_windows.library_extension = "*.dll";
		platforms["Windows"] = platform_windows;

		NativePlatformConfig platform_linux;
		platform_linux.name = "Linux/X11";
		platform_linux.entries.push_back("64");
		platform_linux.entries.push_back("32");
		platform_linux.library_extension = "*.so";
		platforms["X11"] = platform_linux;

		NativePlatformConfig platform_osx;
		platform_osx.name = "macOS";
		platform_osx.entries.push_back("64");
		platform_osx.library_extension = "*.framework; Framework, *.dylib; Dynamic Library";
		platforms["macOS"] = platform_osx;

		NativePlatformConfig platform_haiku;
		platform_haiku.name = "Haiku";
		platform_haiku.entries.push_back("64");
		platform_haiku.entries.push_back("32");
		platform_haiku.library_extension = "*.so";
		platforms["Haiku"] = platform_haiku;

		NativePlatformConfig platform_uwp;
		platform_uwp.name = "UWP";
		platform_uwp.entries.push_back("arm");
		platform_uwp.entries.push_back("32");
		platform_uwp.entries.push_back("64");
		platform_uwp.library_extension = "*.dll";
		platforms["UWP"] = platform_uwp;

		NativePlatformConfig platform_android;
		platform_android.name = "Android";
		platform_android.entries.push_back("armeabi-v7a");
		platform_android.entries.push_back("arm64-v8a");
		platform_android.entries.push_back("x86");
		platform_android.entries.push_back("x86_64");
		platform_android.library_extension = "*.so";
		platforms["Android"] = platform_android;

		NativePlatformConfig platform_html5;
		platform_html5.name = "HTML5";
		platform_html5.entries.push_back("wasm32");
		platform_html5.library_extension = "*.wasm";
		platforms["HTML5"] = platform_html5;

		NativePlatformConfig platform_ios;
		platform_ios.name = "iOS";
		platform_ios.entries.push_back("armv7");
		platform_ios.entries.push_back("arm64");
		platform_ios.entries.push_back("x86_64");
		// iOS can use both Static and Dynamic libraries.
		// Frameworks is actually a folder with files.
		platform_ios.library_extension = "*.framework; Framework, *.xcframework; Binary Framework, *.a; Static Library, *.dylib; Dynamic Library";
		platforms["iOS"] = platform_ios;
	}

	VBoxContainer *container = memnew(VBoxContainer);
	add_child(container);
	container->set_anchors_and_offsets_preset(PRESET_WIDE);

	HBoxContainer *hbox = memnew(HBoxContainer);
	container->add_child(hbox);
	Label *label = memnew(Label);
	label->set_text(TTR("Platform:"));
	hbox->add_child(label);
	filter = memnew(MenuButton);
	filter->set_h_size_flags(SIZE_EXPAND_FILL);
	filter->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	hbox->add_child(filter);
	PopupMenu *filter_list = filter->get_popup();
	filter_list->set_hide_on_checkable_item_selection(false);

	int idx = 0;
	for (const KeyValue<String, NativePlatformConfig> &E : platforms) {
		filter_list->add_check_item(E.value.name, idx);
		filter_list->set_item_metadata(idx, E.key);
		filter_list->set_item_checked(idx, true);
		idx += 1;
	}
	filter_list->connect("index_pressed", callable_mp(this, &GDNativeLibraryEditor::_on_filter_selected));

	tree = memnew(Tree);
	container->add_child(tree);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->set_hide_root(true);
	tree->set_column_titles_visible(true);
	tree->set_columns(4);
	tree->set_column_expand(0, false);
	tree->set_column_custom_minimum_width(0, int(200 * EDSCALE));
	tree->set_column_title(0, TTR("Platform"));
	tree->set_column_title(1, TTR("Dynamic Library"));
	tree->set_column_title(2, TTR("Dependencies"));
	tree->set_column_expand(3, false);
	tree->set_column_custom_minimum_width(3, int(110 * EDSCALE));
	tree->connect("button_pressed", callable_mp(this, &GDNativeLibraryEditor::_on_item_button));
	tree->connect("item_collapsed", callable_mp(this, &GDNativeLibraryEditor::_on_item_collapsed));
	tree->connect("item_activated", callable_mp(this, &GDNativeLibraryEditor::_on_item_activated));

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	//file_dialog->set_resizable(true);
	add_child(file_dialog);
	file_dialog->connect("file_selected", callable_mp(this, &GDNativeLibraryEditor::_on_library_selected));
	file_dialog->connect("dir_selected", callable_mp(this, &GDNativeLibraryEditor::_on_library_selected));
	file_dialog->connect("files_selected", callable_mp(this, &GDNativeLibraryEditor::_on_dependencies_selected));

	new_architecture_dialog = memnew(ConfirmationDialog);
	add_child(new_architecture_dialog);
	new_architecture_dialog->set_title(TTR("Add an architecture entry"));
	new_architecture_input = memnew(LineEdit);
	new_architecture_dialog->add_child(new_architecture_input);
	//	new_architecture_dialog->set_custom_minimum_size(Vector2(300, 80) * EDSCALE);
	new_architecture_input->set_anchors_and_offsets_preset(PRESET_HCENTER_WIDE, PRESET_MODE_MINSIZE, 5 * EDSCALE);
	new_architecture_dialog->get_ok_button()->connect("pressed", callable_mp(this, &GDNativeLibraryEditor::_on_create_new_entry));
}

void GDNativeLibraryEditorPlugin::edit(Object *p_node) {
	Ref<GDNativeLibrary> new_library = Object::cast_to<GDNativeLibrary>(p_node);
	if (new_library.is_valid()) {
		library_editor->edit(new_library);
	}
}

bool GDNativeLibraryEditorPlugin::handles(Object *p_node) const {
	return p_node->is_class("GDNativeLibrary");
}

void GDNativeLibraryEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		EditorNode::get_singleton()->make_bottom_panel_item_visible(library_editor);

	} else {
		if (library_editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
		button->hide();
	}
}

GDNativeLibraryEditorPlugin::GDNativeLibraryEditorPlugin(EditorNode *p_node) {
	library_editor = memnew(GDNativeLibraryEditor);
	library_editor->set_custom_minimum_size(Size2(0, 250 * EDSCALE));
	button = p_node->add_bottom_panel_item(TTR("GDNativeLibrary"), library_editor);
	button->hide();
}

#endif
