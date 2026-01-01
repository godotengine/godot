/**************************************************************************/
/*  claude_ai_editor_plugin.cpp                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "claude_ai_editor_plugin.h"

#include "ai_studio_main_ui.h"
#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/docks/dock_constants.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "editor/editor_node.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/box_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/rich_text_label.h"
#include "core/io/config_file.h"
#include "core/io/resource_loader.h"
#include "core/io/file_access.h"
#include "servers/text/text_server.h"
#include "modules/gdscript/gdscript.h"
#include "editor/file_system/editor_file_system.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/check_box.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/margin_container.h"
#include "core/os/os.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"

class ClaudeAIDock : public EditorDock {
	GDCLASS(ClaudeAIDock, EditorDock)

private:
	VBoxContainer *main_container = nullptr;
	TextEdit *prompt_input = nullptr;
	RichTextLabel *response_output = nullptr;
	ScrollContainer *conversation_container = nullptr;
	VBoxContainer *conversation_messages = nullptr;
	Button *clear_conversation_button = nullptr;
	Button *send_button = nullptr;
	Button *save_button = nullptr;
	Button *write_to_codebase_button = nullptr;
	CheckBox *auto_save_checkbox = nullptr;
	LineEdit *api_key_input = nullptr;
	Label *status_label = nullptr;
	Node *api_handler = nullptr;
	bool codebase_aware = true;
	bool auto_save_enabled = true; // Default to auto-save enabled
	String last_response_text = ""; // Store last response for auto-save
	
	// SaaS UI Components
	HBoxContainer *auth_container = nullptr;
	Label *user_email_label = nullptr;
	Button *login_button = nullptr;
	Button *logout_button = nullptr;
	VBoxContainer *subscription_container = nullptr;
	Label *tier_label = nullptr;
	ProgressBar *usage_bar = nullptr;
	Label *usage_label = nullptr;
	Button *upgrade_button = nullptr;
	bool is_authenticated = false;
	bool use_saas_mode = true; // Default to SaaS mode

	void _send_request();
	void _save_code();
	void _write_to_codebase();
	void _on_file_selected(const String &path);
	void _on_request_complete(const String &response_text);
	void _on_request_error(const String &error_message);
	void _on_auto_save_toggled(bool pressed);
	void _setup_project_files();
	bool _check_project_setup();
	void _reload_api_handler();
	void _verify_api_handler_methods(); // Verify API handler methods are available
	void _retry_write_after_reload(const String &response_text, bool is_auto); // Retry write after reload
	void _write_to_codebase_auto(); // Automatic write (Cursor-like)
	void _write_to_codebase_internal(bool is_auto); // Internal write method
	void _setup_saas_ui(); // Setup SaaS UI components
	void _show_login_dialog(); // Show login/register dialog
	void _on_login_clicked(); // Handle login button
	void _on_logout_clicked(); // Handle logout button
	void _on_auth_status_changed(bool authenticated); // Handle auth status change
	void _on_usage_updated(Dictionary usage_data); // Handle usage updates
	void _update_subscription_display(); // Update subscription UI
	void _check_saas_mode(); // Check if SaaS mode should be used
	void _check_auth_status(); // Check authentication status
	void _on_upgrade_clicked(); // Handle upgrade button click
	void _on_clear_conversation(); // Clear conversation history
	void _on_conversation_updated(); // Handle conversation update
	void _on_ai_question(const String &question); // Handle AI question
	void _add_message_to_conversation(const String &role, const String &content); // Add message to chat
	void _scroll_conversation_to_bottom(); // Scroll conversation to bottom
	void _add_welcome_message(); // Add welcome message
	String _format_message_content(const String &content); // Format message content with BBCode
	void _on_save_code_from_message(const String &content); // Save code from message
	void _on_copy_code_from_message(const String &content); // Copy code from message

protected:
	static void _bind_methods();

public:
	ClaudeAIDock();
};

void ClaudeAIDock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_send_request"), &ClaudeAIDock::_send_request);
	ClassDB::bind_method(D_METHOD("_save_code"), &ClaudeAIDock::_save_code);
	ClassDB::bind_method(D_METHOD("_write_to_codebase"), &ClaudeAIDock::_write_to_codebase);
	ClassDB::bind_method(D_METHOD("_on_file_selected", "path"), &ClaudeAIDock::_on_file_selected);
	ClassDB::bind_method(D_METHOD("_on_request_complete", "response_text"), &ClaudeAIDock::_on_request_complete);
	ClassDB::bind_method(D_METHOD("_on_request_error", "error_message"), &ClaudeAIDock::_on_request_error);
	ClassDB::bind_method(D_METHOD("_on_auto_save_toggled", "pressed"), &ClaudeAIDock::_on_auto_save_toggled);
	ClassDB::bind_method(D_METHOD("_write_to_codebase_auto"), &ClaudeAIDock::_write_to_codebase_auto);
	ClassDB::bind_method(D_METHOD("_on_login_clicked"), &ClaudeAIDock::_on_login_clicked);
	ClassDB::bind_method(D_METHOD("_on_logout_clicked"), &ClaudeAIDock::_on_logout_clicked);
	ClassDB::bind_method(D_METHOD("_on_auth_status_changed", "authenticated"), &ClaudeAIDock::_on_auth_status_changed);
	ClassDB::bind_method(D_METHOD("_on_usage_updated", "usage_data"), &ClaudeAIDock::_on_usage_updated);
	ClassDB::bind_method(D_METHOD("_on_upgrade_clicked"), &ClaudeAIDock::_on_upgrade_clicked);
	ClassDB::bind_method(D_METHOD("_on_clear_conversation"), &ClaudeAIDock::_on_clear_conversation);
	ClassDB::bind_method(D_METHOD("_on_conversation_updated"), &ClaudeAIDock::_on_conversation_updated);
	ClassDB::bind_method(D_METHOD("_on_ai_question", "question"), &ClaudeAIDock::_on_ai_question);
	ClassDB::bind_method(D_METHOD("_on_save_code_from_message", "content"), &ClaudeAIDock::_on_save_code_from_message);
	ClassDB::bind_method(D_METHOD("_on_copy_code_from_message", "content"), &ClaudeAIDock::_on_copy_code_from_message);
	ClassDB::bind_method(D_METHOD("_reload_api_handler"), &ClaudeAIDock::_reload_api_handler);
	ClassDB::bind_method(D_METHOD("_verify_api_handler_methods"), &ClaudeAIDock::_verify_api_handler_methods);
	ClassDB::bind_method(D_METHOD("_retry_write_after_reload", "response_text", "is_auto"), &ClaudeAIDock::_retry_write_after_reload);
}

ClaudeAIDock::ClaudeAIDock() {
	set_name(TTR("DotAI Assistant"));
	set_icon_name("Script");
	set_default_slot(DockConstants::DOCK_SLOT_RIGHT_UL);
	set_available_layouts(EditorDock::DOCK_LAYOUT_ALL);
	set_global(false);
	set_transient(false); // Make it persistent as a core feature
	set_custom_minimum_size(Size2(450, 400) * EDSCALE);

	main_container = memnew(VBoxContainer);
	main_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(main_container);

	// Check SaaS mode preference - DISABLED FOR TESTING
	_check_saas_mode();
	use_saas_mode = false; // Force disable SaaS mode for testing (no login required)
	
	// Skip SaaS UI setup for testing - always use direct API mode
	if (false) { // Disabled for testing
		_setup_saas_ui();
	} else {
		// Direct API mode - show API key input
		HBoxContainer *api_key_container = memnew(HBoxContainer);
		Label *api_key_label = memnew(Label);
		api_key_label->set_text(TTR("API Key:"));
		api_key_label->set_custom_minimum_size(Size2(80 * EDSCALE, 0));
		api_key_container->add_child(api_key_label);

		api_key_input = memnew(LineEdit);
		api_key_input->set_placeholder(TTR("Enter your Claude API key (required)"));
		api_key_input->set_secret(true);
		api_key_input->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		api_key_container->add_child(api_key_input);
		main_container->add_child(api_key_container);
	}

	// Status Label (minimal, hidden by default)
	status_label = memnew(Label);
	status_label->set_text("");
	status_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	status_label->set_visible(false);
	main_container->add_child(status_label);
	
	// Auto-save is always enabled
	auto_save_enabled = true;

	// Cursor-like UI: Conversation area (takes most space)
	conversation_container = memnew(ScrollContainer);
	conversation_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	conversation_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_container->add_child(conversation_container);
	
	// Messages container
	conversation_messages = memnew(VBoxContainer);
	conversation_messages->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	conversation_messages->add_theme_constant_override("separation", 16 * EDSCALE);
	conversation_container->add_child(conversation_messages);
	
	// Add welcome message
	_add_welcome_message();
	
	// Input area at bottom (Cursor style)
	HSeparator *input_separator = memnew(HSeparator);
	main_container->add_child(input_separator);
	
	MarginContainer *input_container = memnew(MarginContainer);
	input_container->add_theme_constant_override("margin_left", 16 * EDSCALE);
	input_container->add_theme_constant_override("margin_right", 16 * EDSCALE);
	input_container->add_theme_constant_override("margin_top", 12 * EDSCALE);
	input_container->add_theme_constant_override("margin_bottom", 12 * EDSCALE);
	
	VBoxContainer *input_vbox = memnew(VBoxContainer);
	input_container->add_child(input_vbox);
	
	// Input field
	prompt_input = memnew(TextEdit);
	prompt_input->set_placeholder(TTR("Ask anything... (Ctrl+Enter to send)"));
	prompt_input->set_custom_minimum_size(Size2(0, 80 * EDSCALE));
	prompt_input->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	prompt_input->set_deselect_on_focus_loss_enabled(false);
	input_vbox->add_child(prompt_input);
	
	// Buttons
	HBoxContainer *input_button_row = memnew(HBoxContainer);
	input_button_row->set_alignment(BoxContainer::ALIGNMENT_END);
	
	clear_conversation_button = memnew(Button);
	clear_conversation_button->set_text(TTR("Clear"));
	clear_conversation_button->set_flat(true);
	clear_conversation_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_clear_conversation));
	input_button_row->add_child(clear_conversation_button);
	
	input_button_row->add_spacer();
	
	send_button = memnew(Button);
	send_button->set_text(TTR("Send"));
	send_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_send_request));
	input_button_row->add_child(send_button);
	
	input_vbox->add_child(input_button_row);
	main_container->add_child(input_container);
	
	// Hidden response output
	response_output = memnew(RichTextLabel);
	response_output->set_visible(false);
	main_container->add_child(response_output);

	// Create API handler - script will be loaded when needed
	api_handler = memnew(Node);
	add_child(api_handler);
	
	// Try to load script from module directory (for development) or project (for distribution)
	// First try project path, then module path
	String script_path = "res://addons/claude_ai/claude_api_handler.gd";
	Ref<Script> script = ResourceLoader::load(script_path);
	if (!script.is_valid()) {
		// Try alternative path in project root
		script_path = "res://claude_api_handler.gd";
		script = ResourceLoader::load(script_path);
	}
	
	if (script.is_valid()) {
		print_line("DotAI C++: Loading API handler script: ", script_path);
		api_handler->set_script(script);
		
		// Wait a frame for script to initialize before verifying methods
		call_deferred("_verify_api_handler_methods");
		
		// Connect signals from the script
		if (api_handler->has_signal("request_complete")) {
			api_handler->connect("request_complete", callable_mp(this, &ClaudeAIDock::_on_request_complete));
		}
		if (api_handler->has_signal("request_error")) {
			api_handler->connect("request_error", callable_mp(this, &ClaudeAIDock::_on_request_error));
		}
		if (api_handler->has_signal("conversation_updated")) {
			api_handler->connect("conversation_updated", callable_mp(this, &ClaudeAIDock::_on_conversation_updated));
		}
		if (api_handler->has_signal("ai_question")) {
			api_handler->connect("ai_question", callable_mp(this, &ClaudeAIDock::_on_ai_question));
		}
		print_line("DotAI C++: API handler script loaded and signals connected");
	} else {
		print_line("DotAI C++: WARNING - API handler script not found at: ", script_path);
		// Script not found - try to set up project files
		call_deferred("_setup_project_files");
	}
	
	// Check if project needs setup
	call_deferred("_check_project_setup");
}

void ClaudeAIDock::_send_request() {
	// Authentication check DISABLED FOR TESTING
	// if (use_saas_mode && !is_authenticated) {
	// 	if (status_label) {
	// 		status_label->set_text(TTR("Please sign in to use AI features"));
	// 	}
	// 	_show_login_dialog();
	// 	return;
	// }
	
	// Check if API key is provided (required)
	if (!use_saas_mode && (!api_key_input || api_key_input->get_text().is_empty())) {
		if (status_label) {
			status_label->set_text(TTR("Error: API key is required. Please enter your Claude API key."));
			status_label->set_visible(true);
		}
		return;
	}

	if (!prompt_input || prompt_input->get_text().is_empty()) {
		if (status_label) {
			status_label->set_text(TTR("Error: Please enter a prompt"));
		}
		return;
	}

	if (api_handler && api_handler->has_method("send_request")) {
		// Cursor-like: Show that we're working on the complete feature
		if (status_label) {
			status_label->set_text(TTR("Generating code and creating files..."));
		}
		if (send_button) {
			send_button->set_disabled(true);
		}
		if (write_to_codebase_button) {
			write_to_codebase_button->set_disabled(true);
		}

		Dictionary params;
		// API key is required - get it from the input field
		if (!use_saas_mode && api_key_input) {
			params["api_key"] = api_key_input->get_text();
		}
		params["prompt"] = prompt_input->get_text();
		params["include_codebase"] = true; // Enable codebase awareness
		params["is_conversation"] = true; // Enable conversation mode

		// Add user message to conversation UI (Cursor style)
		String user_message = prompt_input->get_text();
		if (!user_message.is_empty()) {
			_add_message_to_conversation("user", user_message);
			
			// Clear prompt input
			prompt_input->set_text("");
		}

		Variant result = api_handler->call("send_request", params);
		// The script will handle the response via signals, which will auto-save files
	} else {
		if (status_label) {
			status_label->set_text(TTR("Error: API handler script not found. Please ensure claude_api_handler.gd exists."));
		}
	}
}

void ClaudeAIDock::_on_request_complete(const String &response_text) {
	// Store response text for auto-save
	last_response_text = response_text;
	
	// Add AI response to conversation
	_add_message_to_conversation("assistant", response_text);
	
	// Re-enable send button
	if (send_button) {
		send_button->set_disabled(false);
	}
	if (clear_conversation_button) {
		clear_conversation_button->set_disabled(false);
	}
	
	// Cursor-like: Automatically write files immediately after generation
	if (auto_save_enabled && !response_text.is_empty()) {
		// Check if response contains code (file markers, code blocks, or GDScript keywords)
		// More aggressive detection - check for any code-like content
		bool has_code = response_text.contains("# File:") || 
		                response_text.contains("File:") ||
		                response_text.contains("```gdscript") || 
		                response_text.contains("```gd") ||
		                response_text.contains("```") ||
		                response_text.contains("extends ") || 
		                response_text.contains("class_name") ||
		                response_text.contains("@tool") ||
		                response_text.contains("@export") ||
		                response_text.contains("func ") ||
		                response_text.contains("var ") ||
		                response_text.contains("const ") ||
		                response_text.contains("signal ") ||
		                response_text.contains("[gd_scene") ||
		                response_text.contains("[ext_resource") ||
		                response_text.contains("[node");
		
		// Always try to write if code is detected OR if response is substantial
		// Let FileWriter decide if there's actually code to save
		if (has_code || response_text.length() > 100) {
			// Automatically write files - Cursor-like seamless experience
			if (status_label) {
				status_label->set_text(TTR("Writing files to project..."));
				status_label->set_visible(true);
			}
			call_deferred("_write_to_codebase_auto");
		} else {
			// Very short response, likely just text
			if (status_label) {
				status_label->set_text(TTR("✓ Response received (no code detected)"));
				status_label->set_visible(true);
			}
		}
	} else {
		if (status_label) {
			status_label->set_text(TTR("✓ Response received"));
			status_label->set_visible(true);
		}
	}
}

void ClaudeAIDock::_on_request_error(const String &error_message) {
	// Add error message to conversation (Cursor style)
	_add_message_to_conversation("assistant", "❌ Error: " + error_message);
	
	// Re-enable send button
	if (send_button) {
		send_button->set_disabled(false);
	}
	
	if (status_label) {
		status_label->set_text(TTR("✗ Error: ") + error_message);
		status_label->set_visible(true);
	}
}

void ClaudeAIDock::_write_to_codebase_auto() {
	// Cursor-like automatic write - seamless experience
	_write_to_codebase_internal(true);
}

void ClaudeAIDock::_write_to_codebase() {
	// Manual write (from button click)
	_write_to_codebase_internal(false);
}

void ClaudeAIDock::_write_to_codebase_internal(bool is_auto) {
	// Use stored response text instead of reading from hidden response_output
	String response_text = last_response_text;
	
	// Debug output
	print_line(vformat("DotAI C++: _write_to_codebase_internal called (auto: %s)", is_auto ? "true" : "false"));
	print_line(vformat("DotAI C++: Response text length: %d", response_text.length()));
	
	if (response_text.is_empty()) {
		print_line("DotAI C++: ERROR - Response text is empty!");
		if (status_label) {
			status_label->set_text(TTR("Error: No response text available"));
			status_label->set_visible(true);
		}
		return;
	}
	
	// Always try to write - let FileWriter decide if there's code
	print_line("DotAI C++: Proceeding with file write...");

	if (is_auto && status_label) {
		status_label->set_text(TTR("Writing files to project..."));
	}

	// Use file writer to parse and write files
	if (!api_handler) {
		print_line("DotAI C++: ERROR - api_handler is null!");
		if (status_label) {
			status_label->set_text(TTR("Error: API handler not initialized"));
			status_label->set_visible(true);
		}
		return;
	}
	
	// Check if script is loaded
	Ref<Script> script = api_handler->get_script();
	if (!script.is_valid()) {
		print_line("DotAI C++: ERROR - API handler script not loaded!");
		if (status_label) {
			status_label->set_text(TTR("Error: API handler script not loaded. Please restart the editor."));
			status_label->set_visible(true);
		}
		return;
	}
	
	// Check if method exists
	if (!api_handler->has_method("write_files_to_codebase")) {
		print_line("DotAI C++: ERROR - write_files_to_codebase method not found!");
		if (script.is_valid()) {
			print_line("DotAI C++: API handler script: ", script->get_path());
		}
		// Note: get_method_list() requires an argument in Godot 4
		// print_line("DotAI C++: Available methods: ", api_handler->get_method_list(false));
		
		// Try to reload the script
		call_deferred("_reload_api_handler");
		
		if (status_label) {
			status_label->set_text(TTR("Error: Method not found. Reloading script..."));
			status_label->set_visible(true);
		}
		return;
	}
	
	print_line("DotAI C++: Calling write_files_to_codebase method...");
	Dictionary params;
	params["response_text"] = response_text;
	
	Variant result_variant = api_handler->call("write_files_to_codebase", params);
	
	if (result_variant.get_type() == Variant::DICTIONARY) {
			Dictionary result = result_variant;
			
			if (result.has("success") && result["success"]) {
				Array files_written = result.get("files_written", Array());
				Array files_created = result.get("files_created", Array());
				Array files_modified = result.get("files_modified", Array());
				Array messages = result.get("messages", Array());
				
				// Build detailed message
				String message;
				if (is_auto) {
					// Cursor-like: Brief, friendly message
					message = TTR("✓ ") + String::num_int64(files_written.size()) + TTR(" file(s) created");
					if (files_created.size() > 0 && files_modified.size() > 0) {
						message += TTR(" (") + String::num_int64(files_created.size()) + TTR(" new, ") + 
						           String::num_int64(files_modified.size()) + TTR(" updated)");
					}
				} else {
					// Manual: More detailed
					message = TTR("✓ Successfully processed ") + String::num_int64(files_written.size()) + TTR(" file(s)");
					if (files_created.size() > 0) {
						message += " (" + String::num_int64(files_created.size()) + TTR(" created");
						if (files_modified.size() > 0) {
							message += ", " + String::num_int64(files_modified.size()) + TTR(" modified");
						}
						message += ")";
					}
				}
				
				// Show file list in response output (for reference)
				if (response_output && messages.size() > 0) {
					String details = TTR("Files written:\n\n");
					for (int i = 0; i < messages.size(); i++) {
						details += String(messages[i]) + "\n";
					}
					// Keep original code visible below
					details += "\n" + TTR("--- Generated Code ---\n\n") + response_text;
					response_output->set_text(details);
				}
				
				if (status_label) {
					status_label->set_text(message);
				}
				
				// Refresh file system
				EditorFileSystem::get_singleton()->scan_changes();
			} else {
				String error = result.get("error", TTR("Unknown error"));
				Array messages = result.get("messages", Array());
				Array files_failed = result.get("files_failed", Array());
				
				String error_msg = TTR("Error: ") + error;
				if (files_failed.size() > 0) {
					error_msg += "\n" + TTR("Failed files: ");
					for (int i = 0; i < files_failed.size(); i++) {
						if (i > 0) error_msg += ", ";
						error_msg += String(files_failed[i]);
					}
				}
				
				if (status_label) {
					status_label->set_text(error_msg);
				}
				
				// Show error messages in response output
				if (response_output && messages.size() > 0) {
					String details = TTR("Errors occurred:\n\n");
					for (int i = 0; i < messages.size(); i++) {
						details += String(messages[i]) + "\n";
					}
					response_output->set_text(details);
				}
			}
		} else {
			// Result is not a dictionary
			print_line("DotAI C++: ERROR - write_files_to_codebase did not return a dictionary.");
			if (status_label) {
				status_label->set_text(TTR("Error: Invalid response from write_files_to_codebase"));
				status_label->set_visible(true);
			}
		}
}

void ClaudeAIDock::_on_auto_save_toggled(bool pressed) {
	auto_save_enabled = pressed;
	
	// Save preference
	Ref<ConfigFile> config;
	config.instantiate();
		String config_path = "user://dotai.cfg";
	Error err = config->load(config_path);
	if (err == OK) {
		config->set_value("settings", "auto_save", pressed);
		config->save(config_path);
	} else {
		// Create new config
		config->set_value("settings", "auto_save", pressed);
		config->save(config_path);
	}
}

bool ClaudeAIDock::_check_project_setup() {
	String project_path = ProjectSettings::get_singleton()->get_resource_path();
	if (project_path.is_empty()) {
		return false; // No project loaded
	}
	
	String addons_path = project_path.path_join("addons/claude_ai");
	
	// Check if required files exist
	Ref<FileAccess> test_api = FileAccess::open("res://addons/claude_ai/claude_api_handler.gd", FileAccess::READ);
	Ref<FileAccess> test_scanner = FileAccess::open("res://addons/claude_ai/codebase_scanner.gd", FileAccess::READ);
	Ref<FileAccess> test_writer = FileAccess::open("res://addons/claude_ai/file_writer.gd", FileAccess::READ);
	bool has_api_handler = test_api.is_valid();
	bool has_scanner = test_scanner.is_valid();
	bool has_writer = test_writer.is_valid();
	if (test_api.is_valid()) test_api->close();
	if (test_scanner.is_valid()) test_scanner->close();
	if (test_writer.is_valid()) test_writer->close();
	
	if (!has_api_handler || !has_scanner || !has_writer) {
		// Project needs setup
		_setup_project_files();
		return false;
	}
	
	return true;
}

void ClaudeAIDock::_setup_project_files() {
	String project_path = ProjectSettings::get_singleton()->get_resource_path();
	if (project_path.is_empty()) {
		return; // No project loaded
	}
	
	// Create addons directory if it doesn't exist
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (dir.is_valid() && dir->change_dir(project_path) == OK) {
		if (!dir->dir_exists("addons")) {
			Error err = dir->make_dir("addons");
			if (err != OK) {
				if (status_label) {
					status_label->set_text(TTR("Warning: Could not create addons directory"));
				}
				return;
			}
		}
		dir->change_dir("addons");
		if (!dir->dir_exists("claude_ai")) {
			Error err = dir->make_dir("claude_ai");
			if (err != OK) {
				if (status_label) {
					status_label->set_text(TTR("Warning: Could not create claude_ai directory"));
				}
				return;
			}
		}
	}
	
	// Try to load files from module directory (embedded in executable)
	// Or use ResourceLoader to get them from res://
	String module_base = "res://modules/claude_ai/";
	
	// List of files to copy
	PackedStringArray files_to_copy;
	files_to_copy.append("claude_api_handler.gd");
	files_to_copy.append("codebase_scanner.gd");
	files_to_copy.append("file_writer.gd");
	files_to_copy.append("conversation_manager.gd");
	
	bool files_copied = false;
	for (int i = 0; i < files_to_copy.size(); i++) {
		String filename = files_to_copy[i];
		String dest_path = "res://addons/claude_ai/" + filename;
		
		// Only copy if destination doesn't exist
		Ref<FileAccess> test_file = FileAccess::open(dest_path, FileAccess::READ);
		if (!test_file.is_valid()) {
			// Try to load from module path first
			String source_path = module_base + filename;
			Ref<FileAccess> source_file = FileAccess::open(source_path, FileAccess::READ);
			
			if (!source_file.is_valid()) {
				// Try alternative path
				source_path = "res://" + filename;
				source_file = FileAccess::open(source_path, FileAccess::READ);
			}
			
			if (source_file.is_valid()) {
				String content = source_file->get_as_text();
				source_file->close();
				
				Ref<FileAccess> dest_file = FileAccess::open(dest_path, FileAccess::WRITE);
				if (dest_file.is_valid()) {
					dest_file->store_string(content);
					dest_file->close();
					files_copied = true;
				}
			}
		}
	}
	
	if (files_copied) {
		// Refresh file system
		EditorFileSystem::get_singleton()->scan_changes();
		
		// Try to reload the script after a short delay
		call_deferred("_reload_api_handler");
	}
}

void ClaudeAIDock::_reload_api_handler() {
	if (!api_handler) {
		print_line("DotAI C++: Cannot reload - api_handler is null");
		return;
	}
	
	print_line("DotAI C++: Reloading API handler script...");
	String script_path = use_saas_mode ? "res://addons/claude_ai/saas_api_handler.gd" : "res://addons/claude_ai/claude_api_handler.gd";
	Ref<Script> script = ResourceLoader::load(script_path);
	if (!script.is_valid()) {
		script_path = use_saas_mode ? "res://saas_api_handler.gd" : "res://claude_api_handler.gd";
		script = ResourceLoader::load(script_path);
	}
	
	if (script.is_valid()) {
		print_line("DotAI C++: Script loaded: ", script_path);
		api_handler->set_script(script);
		
		// Wait a frame for script to initialize, then verify methods
		call_deferred("_verify_api_handler_methods");
		
		// Reconnect all signals
		if (api_handler->has_signal("request_complete")) {
			api_handler->connect("request_complete", callable_mp(this, &ClaudeAIDock::_on_request_complete));
		}
		if (api_handler->has_signal("request_error")) {
			api_handler->connect("request_error", callable_mp(this, &ClaudeAIDock::_on_request_error));
		}
		if (api_handler->has_signal("conversation_updated")) {
			api_handler->connect("conversation_updated", callable_mp(this, &ClaudeAIDock::_on_conversation_updated));
		}
		if (api_handler->has_signal("ai_question")) {
			api_handler->connect("ai_question", callable_mp(this, &ClaudeAIDock::_on_ai_question));
		}
		if (use_saas_mode) {
			if (api_handler->has_signal("auth_status_changed")) {
				api_handler->connect("auth_status_changed", callable_mp(this, &ClaudeAIDock::_on_auth_status_changed));
			}
			if (api_handler->has_signal("usage_updated")) {
				api_handler->connect("usage_updated", callable_mp(this, &ClaudeAIDock::_on_usage_updated));
			}
		}
		print_line("DotAI C++: API handler script reloaded successfully");
	} else {
		print_line("DotAI C++: ERROR - Failed to load script from: ", script_path);
		if (status_label) {
			status_label->set_text(TTR("Error: Failed to load API handler script. Check Output panel for details."));
			status_label->set_visible(true);
		}
	}
}

void ClaudeAIDock::_verify_api_handler_methods() {
	if (!api_handler) {
		return;
	}
	
	print_line("DotAI C++: Verifying API handler methods...");
	bool has_write_method = api_handler->has_method("write_files_to_codebase");
	bool has_send_method = api_handler->has_method("send_request");
	
	print_line("DotAI C++: write_files_to_codebase: ", has_write_method ? "AVAILABLE" : "NOT FOUND");
	print_line("DotAI C++: send_request: ", has_send_method ? "AVAILABLE" : "NOT FOUND");
	
	if (!has_write_method) {
		print_line("DotAI C++: WARNING - write_files_to_codebase method still not available after reload!");
		Ref<Script> script = api_handler->get_script();
		if (script.is_valid()) {
			print_line("DotAI C++: Script path: ", script->get_path());
		} else {
			print_line("DotAI C++: Script is not valid!");
		}
	}
}

void ClaudeAIDock::_retry_write_after_reload(const String &response_text, bool is_auto) {
	// Store the response text for retry
	last_response_text = response_text;
	
	// Wait a moment for script to initialize, then try again
	if (api_handler && api_handler->has_method("write_files_to_codebase")) {
		print_line("DotAI C++: Retry successful - method now available");
		_write_to_codebase_internal(is_auto);
	} else {
		print_line("DotAI C++: Retry failed - method still not available");
		if (status_label) {
			status_label->set_text(TTR("Error: Unable to write files. Please check Output panel for details."));
			status_label->set_visible(true);
		}
	}
}

void ClaudeAIDock::_check_saas_mode() {
	Ref<ConfigFile> config;
	config.instantiate();
		String config_path = "user://dotai.cfg";
	if (config->load(config_path) == OK) {
		use_saas_mode = config->get_value("settings", "use_saas_mode", true);
	} else {
		use_saas_mode = true; // Default to SaaS mode
	}
}

void ClaudeAIDock::_setup_saas_ui() {
	// Authentication container
	auth_container = memnew(HBoxContainer);
	auth_container->set_alignment(BoxContainer::ALIGNMENT_END);
	
	user_email_label = memnew(Label);
	user_email_label->set_text(TTR("Not signed in"));
	user_email_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	auth_container->add_child(user_email_label);
	
	login_button = memnew(Button);
	login_button->set_text(TTR("Sign In"));
	login_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_login_clicked));
	auth_container->add_child(login_button);
	
	logout_button = memnew(Button);
	logout_button->set_text(TTR("Sign Out"));
	logout_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_logout_clicked));
	logout_button->set_visible(false);
	auth_container->add_child(logout_button);
	
	main_container->add_child(auth_container);
	
	// Subscription container
	subscription_container = memnew(VBoxContainer);
	
	tier_label = memnew(Label);
	tier_label->set_text(TTR("Free Plan"));
	subscription_container->add_child(tier_label);
	
	usage_bar = memnew(ProgressBar);
	usage_bar->set_max(100);
	usage_bar->set_value(0);
	usage_bar->set_custom_minimum_size(Size2(0, 20 * EDSCALE));
	subscription_container->add_child(usage_bar);
	
	usage_label = memnew(Label);
	usage_label->set_text(TTR("Usage: 0 / 100 requests"));
	usage_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	subscription_container->add_child(usage_label);
	
	upgrade_button = memnew(Button);
	upgrade_button->set_text(TTR("Upgrade to Pro"));
	upgrade_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_upgrade_clicked));
	upgrade_button->set_visible(false);
	subscription_container->add_child(upgrade_button);
	
	main_container->add_child(subscription_container);
}

void ClaudeAIDock::_on_login_clicked() {
	_show_login_dialog();
}

void ClaudeAIDock::_on_logout_clicked() {
	if (api_handler && api_handler->has_method("logout")) {
		api_handler->call("logout");
	}
	is_authenticated = false;
	_on_auth_status_changed(false);
}

void ClaudeAIDock::_on_auth_status_changed(bool authenticated) {
	is_authenticated = authenticated;
	
	if (auth_container) {
		if (authenticated) {
			if (login_button) login_button->set_visible(false);
			if (logout_button) logout_button->set_visible(true);
			if (user_email_label && api_handler && api_handler->has_method("get_user_email")) {
				String email = api_handler->call("get_user_email");
				user_email_label->set_text(email);
			}
		} else {
			if (login_button) login_button->set_visible(true);
			if (logout_button) logout_button->set_visible(false);
			if (user_email_label) user_email_label->set_text(TTR("Not signed in"));
		}
	}
	
	// Auth check DISABLED FOR TESTING - always enable send button
	if (send_button) {
		send_button->set_disabled(false); // Always enabled for testing (no login required)
	}
	
	// Update subscription display
	_update_subscription_display();
}

void ClaudeAIDock::_on_usage_updated(Dictionary usage_data) {
	_update_subscription_display();
}

void ClaudeAIDock::_update_subscription_display() {
	if (!subscription_container) return;
	
	if (api_handler && api_handler->has_method("get_usage")) {
		Dictionary usage = api_handler->call("get_usage");
		// Update UI with usage data
		// This would be called after API handler fetches usage
	}
	
	// For now, show placeholder
	if (usage_label) {
		usage_label->set_text(TTR("Usage: Loading..."));
	}
}

void ClaudeAIDock::_show_login_dialog() {
	// Create a simple dialog for login/register
	// In a full implementation, this would be a proper dialog window
	if (status_label) {
		status_label->set_text(TTR("Login dialog - Enter email and password in API handler"));
	}
	
	// For now, show instructions
	// Full implementation would create AcceptDialog with login form
}

void ClaudeAIDock::_check_auth_status() {
	if (api_handler && api_handler->has_method("check_auth_status")) {
		api_handler->call("check_auth_status");
	}
}

void ClaudeAIDock::_on_upgrade_clicked() {
	// Open upgrade URL or show upgrade dialog
	if (status_label) {
		status_label->set_text(TTR("Upgrade to Pro - Visit https://godot-ai-studio.com/upgrade"));
	}
}

void ClaudeAIDock::_save_code() {
	if (!response_output) {
		return;
	}

	String code = response_output->get_text();
	if (code.is_empty()) {
		if (status_label) {
			status_label->set_text(TTR("Error: No code to save"));
		}
		return;
	}

	// Use EditorFileSystem to get current directory
	FileDialog *file_dialog = memnew(FileDialog);
	file_dialog->set_file_mode(FileDialog::FILE_MODE_SAVE_FILE);
	file_dialog->set_access(FileDialog::ACCESS_RESOURCES);
	file_dialog->add_filter("*.gd", "GDScript");
	file_dialog->set_current_dir("res://");
	file_dialog->set_current_file("generated_script.gd");
	file_dialog->set_title(TTR("Save Generated Code"));
	
	file_dialog->connect("file_selected", callable_mp(this, &ClaudeAIDock::_on_file_selected));
	
	add_child(file_dialog);
	file_dialog->popup_centered_ratio(0.5);
}

void ClaudeAIDock::_on_file_selected(const String &path) {
	if (!response_output) {
		return;
	}

	String code = response_output->get_text();
	
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
	if (file.is_valid()) {
		file->store_string(code);
		file->close();
		
		if (status_label) {
			status_label->set_text(TTR("Code saved successfully to: ") + path);
		}
		
		// Refresh the file system
		EditorFileSystem::get_singleton()->update_file(path);
		EditorFileSystem::get_singleton()->scan_changes();
	} else {
		if (status_label) {
			status_label->set_text(TTR("Error: Failed to save file"));
		}
	}
}

void ClaudeAIDock::_add_message_to_conversation(const String &role, const String &content) {
	if (!conversation_messages) {
		return;
	}
	
	// Cursor-like message layout: Full-width container with padding
	MarginContainer *message_margin = memnew(MarginContainer);
	message_margin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	message_margin->add_theme_constant_override("margin_left", 16 * EDSCALE);
	message_margin->add_theme_constant_override("margin_right", 16 * EDSCALE);
	message_margin->add_theme_constant_override("margin_top", 12 * EDSCALE);
	message_margin->add_theme_constant_override("margin_bottom", 12 * EDSCALE);
	
	HBoxContainer *message_row = memnew(HBoxContainer);
	message_row->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	
	// Avatar/icon (Cursor style - small icon on left)
	Label *avatar_label = memnew(Label);
	if (role == "user") {
		avatar_label->set_text("👤");
	} else {
		avatar_label->set_text("🤖");
	}
	avatar_label->set_custom_minimum_size(Size2(32 * EDSCALE, 32 * EDSCALE));
	avatar_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	avatar_label->set_vertical_alignment(VERTICAL_ALIGNMENT_TOP);
	message_row->add_child(avatar_label);
	
	// Message content container
	VBoxContainer *message_content = memnew(VBoxContainer);
	message_content->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	message_content->add_theme_constant_override("separation", 4 * EDSCALE);
	
	// Role label (subtle, Cursor style)
	Label *role_label = memnew(Label);
		role_label->set_text(role == "user" ? "You" : "DotAI");
	role_label->add_theme_font_size_override("font_size", 12);
	role_label->set_modulate(Color(0.6, 0.6, 0.6)); // Subtle gray
	message_content->add_child(role_label);
	
	// Message text - RichTextLabel for better formatting
	RichTextLabel *message_label = memnew(RichTextLabel);
	message_label->set_use_bbcode(true); // Enable BBCode for formatting
	message_label->set_text(_format_message_content(content));
	message_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	message_label->set_fit_content(true);
	message_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	message_label->set_selection_enabled(true);
	message_label->set_context_menu_enabled(true);
	
	// Store original content for saving
	message_label->set_meta("original_content", content);
	message_label->set_meta("role", role);
	
	message_content->add_child(message_label);
	
	// Add "Save Code" button if message contains code (for AI messages)
	if (role == "assistant" && (content.contains("# File:") || content.contains("```") || content.contains("extends ") || content.contains("func "))) {
		HBoxContainer *action_buttons = memnew(HBoxContainer);
		action_buttons->add_theme_constant_override("separation", 4 * EDSCALE);
		
		Button *save_code_button = memnew(Button);
		save_code_button->set_text(TTR("Save Code"));
		save_code_button->set_flat(true);
		save_code_button->set_custom_minimum_size(Size2(0, 24 * EDSCALE));
		save_code_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_save_code_from_message).bind(content));
		action_buttons->add_child(save_code_button);
		
		Button *copy_code_button = memnew(Button);
		copy_code_button->set_text(TTR("Copy"));
		copy_code_button->set_flat(true);
		copy_code_button->set_custom_minimum_size(Size2(0, 24 * EDSCALE));
		copy_code_button->connect(SceneStringName(pressed), callable_mp(this, &ClaudeAIDock::_on_copy_code_from_message).bind(content));
		action_buttons->add_child(copy_code_button);
		
		action_buttons->add_spacer();
		message_content->add_child(action_buttons);
	}
	
	message_row->add_child(message_content);
	message_margin->add_child(message_row);
	conversation_messages->add_child(message_margin);
	
	// Scroll to bottom
	call_deferred("_scroll_conversation_to_bottom");
}

void ClaudeAIDock::_add_welcome_message() {
	if (!conversation_messages) {
		return;
	}
	
	// Welcome message (Cursor style)
	MarginContainer *welcome_margin = memnew(MarginContainer);
	welcome_margin->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	welcome_margin->add_theme_constant_override("margin_left", 16 * EDSCALE);
	welcome_margin->add_theme_constant_override("margin_right", 16 * EDSCALE);
	welcome_margin->add_theme_constant_override("margin_top", 24 * EDSCALE);
	welcome_margin->add_theme_constant_override("margin_bottom", 12 * EDSCALE);
	
	VBoxContainer *welcome_content = memnew(VBoxContainer);
	welcome_content->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	welcome_content->add_theme_constant_override("separation", 8 * EDSCALE);
	
	Label *welcome_title = memnew(Label);
	welcome_title->set_text("DotAI");
	welcome_title->add_theme_font_size_override("font_size", 18);
	welcome_content->add_child(welcome_title);
	
	Label *welcome_text = memnew(Label);
	welcome_text->set_text("I'm DotAI, your AI-powered game development assistant. I can help you build complete games, write production-ready code, debug issues, and answer questions about your project.\n\nTry asking:\n• \"Create a 2D platformer with player movement and enemies\"\n• \"How does this code work?\"\n• \"Add a health system to my game\"");
	welcome_text->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	welcome_text->set_modulate(Color(0.7, 0.7, 0.7));
	welcome_content->add_child(welcome_text);
	
	welcome_margin->add_child(welcome_content);
	conversation_messages->add_child(welcome_margin);
}

String ClaudeAIDock::_format_message_content(const String &content) {
	// Format message content with BBCode for Cursor-like appearance
	String formatted = content;
	
	// Format file markers (bold)
	formatted = formatted.replace("# File:", "[b]File:[/b]");
	formatted = formatted.replace("File:", "[b]File:[/b]");
	
	// Format code blocks - wrap in [code] tags for syntax highlighting
	// Simple approach: find code blocks and format them
	int code_start = formatted.find("```");
	while (code_start != -1) {
		int code_end = formatted.find("```", code_start + 3);
		if (code_end != -1) {
			String before = formatted.substr(0, code_start);
			String code_block = formatted.substr(code_start + 3, code_end - code_start - 3);
			String after = formatted.substr(code_end + 3);
			
			// Remove language identifier if present
			int newline_pos = code_block.find("\n");
			if (newline_pos != -1) {
				String lang = code_block.substr(0, newline_pos).strip_edges();
				if (lang.length() < 20) { // Reasonable language name length
					code_block = code_block.substr(newline_pos + 1);
				}
			}
			
			formatted = before + "[code]" + code_block.strip_edges() + "[/code]" + after;
			code_start = formatted.find("```", code_start + 7);
		} else {
			break;
		}
	}
	
	// Format inline code
	int inline_start = formatted.find("`");
	while (inline_start != -1) {
		int inline_end = formatted.find("`", inline_start + 1);
		if (inline_end != -1 && inline_end - inline_start < 100) { // Reasonable inline code length
			String before = formatted.substr(0, inline_start);
			String code = formatted.substr(inline_start + 1, inline_end - inline_start - 1);
			String after = formatted.substr(inline_end + 1);
			
			formatted = before + "[code]" + code + "[/code]" + after;
			inline_start = formatted.find("`", inline_start + 7);
		} else {
			break;
		}
	}
	
	return formatted;
}

void ClaudeAIDock::_scroll_conversation_to_bottom() {
	if (conversation_container && conversation_messages) {
		int child_count = conversation_messages->get_child_count();
		if (child_count > 0) {
			Control *last_child = Object::cast_to<Control>(conversation_messages->get_child(child_count - 1));
			if (last_child) {
				conversation_container->ensure_control_visible(last_child);
			}
		}
	}
}

void ClaudeAIDock::_on_clear_conversation() {
	if (conversation_messages) {
		// Remove all message children
		for (int i = conversation_messages->get_child_count() - 1; i >= 0; i--) {
			conversation_messages->get_child(i)->queue_free();
		}
	}
	
	// Clear conversation in API handler
	if (api_handler && api_handler->has_method("clear_conversation")) {
		api_handler->call("clear_conversation");
	}
	
	// Add welcome message back
	_add_welcome_message();
	
	if (status_label) {
		status_label->set_text(TTR("Conversation cleared"));
	}
}

void ClaudeAIDock::_on_conversation_updated() {
	// Refresh conversation display if needed
	// This could reload from API handler's conversation history
	if (api_handler && api_handler->has_method("get_conversation_history")) {
		Variant history = api_handler->call("get_conversation_history");
		// Could rebuild UI from history if needed
	}
}

void ClaudeAIDock::_on_ai_question(const String &question) {
	// Highlight that AI is asking a question
	if (status_label) {
		status_label->set_text(TTR("AI Question: ") + question);
	}
	
	// Could show a notification or highlight in UI
}

void ClaudeAIDock::_on_save_code_from_message(const String &content) {
	// Extract code from message and save it
	if (content.contains("# File:")) {
		// Use existing write to codebase functionality
		if (api_handler && api_handler->has_method("write_files_to_codebase")) {
			Dictionary params;
			params["response_text"] = content;
			Variant result = api_handler->call("write_files_to_codebase", params);
			
			if (status_label) {
				status_label->set_text(TTR("✓ Code saved to project"));
				status_label->set_visible(true);
			}
		} else {
			// Fallback: show file dialog
			_save_code();
		}
	} else {
		// No file markers - show save dialog
		_save_code();
	}
}

void ClaudeAIDock::_on_copy_code_from_message(const String &content) {
	// Extract code blocks from content
	String code_to_copy = "";
	
	if (content.contains("# File:")) {
		// Extract all file content
		code_to_copy = content;
	} else {
		// Extract code blocks
		int code_start = content.find("```");
		if (code_start != -1) {
			int code_end = content.find("```", code_start + 3);
			if (code_end != -1) {
				String code_block = content.substr(code_start + 3, code_end - code_start - 3);
				// Remove language identifier
				int newline_pos = code_block.find("\n");
				if (newline_pos != -1) {
					String lang = code_block.substr(0, newline_pos).strip_edges();
					if (lang.length() < 20) {
						code_block = code_block.substr(newline_pos + 1);
					}
				}
				code_to_copy = code_block.strip_edges();
			}
		} else {
			code_to_copy = content;
		}
	}
	
	// Copy to clipboard
	if (!code_to_copy.is_empty()) {
		DisplayServer::get_singleton()->clipboard_set(code_to_copy);
		if (status_label) {
			status_label->set_text(TTR("✓ Code copied to clipboard"));
			status_label->set_visible(true);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////

String ClaudeAIEditorPlugin::get_plugin_name() const {
	return APPLICATION_NAME_DISPLAY;
}

ClaudeAIEditorPlugin::ClaudeAIEditorPlugin() {
	// Create the main AI Studio dock - this is the core interface
	ClaudeAIDock *dock = memnew(ClaudeAIDock);
	claude_ai_dock = dock;
	EditorDockManager::get_singleton()->add_dock(claude_ai_dock);
	
	// Create main UI controller
	main_ui = memnew(AIStudioMainUI);
	add_child(main_ui);
}


ClaudeAIEditorPlugin::~ClaudeAIEditorPlugin() {
	if (claude_ai_dock) {
		EditorDockManager::get_singleton()->remove_dock(claude_ai_dock);
		claude_ai_dock->queue_free();
	}
}

