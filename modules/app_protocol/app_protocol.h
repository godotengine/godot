/*************************************************************************/
/*  app_protocol.h                                                       */
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

#ifndef APP_PROTOCOL_H
#define APP_PROTOCOL_H

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "thirdparty/ipc/ipc.h"

class ProtocolPlatformImplementation {
public:
	ProtocolPlatformImplementation(){};
	virtual ~ProtocolPlatformImplementation(){};
	virtual bool validate_protocol(const String &p_protocol);
	virtual Error register_protocol_handler(const String &p_protocol) = 0;
};

#if !defined(WINDOWS_ENABLED) && !defined(OSX_ENABLED)
class LinuxDesktopProtocol : public ProtocolPlatformImplementation {
public:
	virtual Error register_protocol_handler(const String &p_protocol) {
		ERR_FAIL_COND_V(!validate_protocol(p_protocol), ERR_INVALID_PARAMETER);
		OS *os = OS::get_singleton();

		// Generate the desktop entry.
		const String scheme_handler = "x-scheme-handler/" + p_protocol;
		const String name = "\nName=" + p_protocol.to_upper() + " Protocol Handler";
#ifdef TOOLS_ENABLED
		// tools enabled / source folders must call explicit path to application
		const String exec = "\nExec=" + os->get_executable_path() + " --path \"" + ProjectSettings::get_singleton()->get_resource_path() + "\" --uri \"%u\"";
#else
		// non tools we assume its an exported game and runs from the current folder - we should test this assumption
		const String exec = "\nExec=" + os->get_executable_path() + " --uri \"%u\"";
#endif
		const String mime = "\nMimeType=" + scheme_handler + ";\n";
		// Example file:
		// [Desktop Entry]
		// Type=Application
		// Name=MYPROTOCOL Protocol Handler
		// Exec=/path/to/godot --uri="%u"
		// MimeType=x-scheme-handler/myprotocol;
		const String desktop_entry = "[Desktop Entry]\nType=Application" + name + exec + mime;
		// Write the desktop entry to a file.
		const String file_name = p_protocol + "-protocol-handler.desktop";
		{
			const String path = os->get_environment("HOME") + "/.local/share/applications/" + file_name;
			Error err;
			Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE, &err);
			if (err) {
				return err;
			}
			file->store_string(desktop_entry);
		}
		// Register this new file with xdg-mime.
		List<String> args;
		args.push_back("default");
		args.push_back(file_name);
		args.push_back(scheme_handler);
		return os->execute("xdg-mime", args);
	}
};
#endif

#ifdef WINDOWS_ENABLED
class WindowsDesktopProtocol : public ProtocolPlatformImplementation {
public:
	virtual Error register_protocol_handler(const String &p_protocol) {
		ERR_FAIL_COND_V(!validate_protocol(p_protocol), ERR_INVALID_PARAMETER);

		const String ExecPath = OS::get_singleton()->get_executable_path().replace("/", "\\");
#ifdef TOOLS_ENABLED
		const String open_command = ExecPath + " --path \"" + ProjectSettings::get_singleton()->get_resource_path() + "\" --uri \"%1\"";
#else
		const String open_command = ExecPath + " --uri \"%1\"";
#endif

		// Create the subkey of HKEY_CLASSES_ROOT if it does not exist.
		HKEY hkey;
		String key_path = "SOFTWARE\\Classes\\" + p_protocol;
		LONG open_res = RegCreateKeyEx(HKEY_CURRENT_USER, key_path.utf8().get_data(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &hkey, nullptr);
		if (open_res != ERROR_SUCCESS) {
			return FAILED;
		}

		// Set empty protocol header (Windows 11 defaults to using the root name of this node)
		LONG set_res2 = RegSetValueEx(hkey, "URL Protocol", 0, REG_SZ, (BYTE *)"", 0);
		// Close the key.
		LONG close_res = RegCloseKey(hkey);

		// Check if all operations were successful.
		if (set_res2 != ERROR_SUCCESS || close_res != ERROR_SUCCESS) {
			return FAILED;
		}

		// Set the shell/open/command subkey.
		String shell_path = "SOFTWARE\\Classes\\" + p_protocol + "\\shell\\open\\command";
		open_res = RegCreateKeyEx(HKEY_CURRENT_USER, shell_path.utf8().get_data(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &hkey, nullptr);
		if (open_res != ERROR_SUCCESS) {
			return FAILED;
		}
		LONG set_res3 = RegSetValueEx(hkey, nullptr, 0, REG_SZ, (BYTE *)open_command.utf8().get_data(), open_command.utf8().length() + 1);
		// Close the key.
		LONG close_res2 = RegCloseKey(hkey);

		// Check if all operations were successful.
		if (set_res3 != ERROR_SUCCESS || close_res2 != ERROR_SUCCESS) {
			return FAILED;
		}

		return OK;
	}
};
#endif

#ifdef OSX_ENABLED
class ApplePlatform : public ProtocolPlatformImplementation {
public:
	virtual Error register_protocol_handler(const String &p_protocol) {
		return OK;
	}
};
#endif // OSX only.

// TODO: make this swap depending on the compiled platform.
// Here I made the assumption that the compiled platform is what uses this
// In the case you run the editor its the CurrentPlatformDefiniton
// In the case you are using an export template it can also be the CurrentPlatformDefiniton since
// The events are happening at export time, and baked into the application
// If this assumption needs to change, totally open to this.

#if defined(WINDOWS_ENABLED) && defined(OSX_ENABLED)
#error "sanity check failed"
#endif

#if defined(WINDOWS_ENABLED)
using CurrentPlatformDefiniton = WindowsDesktopProtocol;
#elif defined(OSX_ENABLED)
using CurrentPlatformDefiniton = ApplePlatform;
#else
using CurrentPlatformDefiniton = LinuxDesktopProtocol; /* Apple is the same across all devices, so specified as 'apple' generically here, nice job apple :) */
#endif

class AppProtocol : public Object {
	GDCLASS(AppProtocol, Object);

protected:
	static AppProtocol *singleton;
	IPCServer *Server = nullptr; // only active when this is enabled, and will be authoritive over this socket until its shutdown.
	static void _bind_methods();

public:
	AppProtocol();
	~AppProtocol();
	static void initialize();
	static void finalize();
	static AppProtocol *get_singleton();
	static bool is_server_running_locally();
	// this object is compile time, so we always keep the same class.
	CurrentPlatformDefiniton CompiledPlatform;
	void register_project_settings();
	static bool is_server_already_running();
	void poll_server();
	static void on_server_get_message(const char *p_str, int strlen);
	static void on_os_get_arguments(const List<String> &args);
};

#endif // APP_PROTOCOL_H
