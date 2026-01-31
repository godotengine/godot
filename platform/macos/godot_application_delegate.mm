/**************************************************************************/
/*  godot_application_delegate.mm                                         */
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

#import "godot_application_delegate.h"

#import "display_server_macos.h"
#import "godot_menu_item.h"
#import "key_mapping_macos.h"
#import "native_menu_macos.h"
#import "os_macos.h"

#import "core/os/main_loop.h"
#import "main/main.h"

#import <Carbon/Carbon.h>

@interface GodotApplicationDelegate ()
- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context;
- (void)accessibilityDisplayOptionsChange:(NSNotification *)notification;
@end

@implementation GodotApplicationDelegate {
	bool high_contrast;
	bool reduce_motion;
	bool reduce_transparency;
	bool voice_over;
	OS_MacOS_NSApp *os_mac;
}

- (GodotApplicationDelegate *)initWithOS:(OS_MacOS_NSApp *)os {
	self = [super init];
	if (self) {
		os_mac = os;
	}

	[[NSWorkspace sharedWorkspace] addObserver:self forKeyPath:@"voiceOverEnabled" options:(NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld) context:(void *)godot_ac_ctx];
	[[[NSWorkspace sharedWorkspace] notificationCenter] addObserver:self selector:@selector(accessibilityDisplayOptionsChange:) name:NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification object:nil];
	high_contrast = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldIncreaseContrast];
	reduce_motion = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceMotion];
	reduce_transparency = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceTransparency];
	voice_over = [[NSWorkspace sharedWorkspace] isVoiceOverEnabled];

	[[NSDistributedNotificationCenter defaultCenter] addObserver:self selector:@selector(system_theme_changed:) name:@"AppleInterfaceThemeChangedNotification" object:nil];
	[[NSDistributedNotificationCenter defaultCenter] addObserver:self selector:@selector(system_theme_changed:) name:@"AppleColorPreferencesChangedNotification" object:nil];

	return self;
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
	return YES;
}

- (NSArray<NSString *> *)localizedTitlesForItem:(id)item {
	NSArray *item_name = @[ item[1] ];
	return item_name;
}

- (void)searchForItemsWithSearchString:(NSString *)searchString resultLimit:(NSInteger)resultLimit matchedItemHandler:(void (^)(NSArray *items))handleMatchedItems {
	NSMutableArray *found_items = [[NSMutableArray alloc] init];

	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds && ds->_help_get_search_callback().is_valid()) {
		Callable cb = ds->_help_get_search_callback();

		Variant ret;
		Variant search_string = String::utf8([searchString UTF8String]);
		Variant result_limit = (uint64_t)resultLimit;
		Callable::CallError ce;
		const Variant *args[2] = { &search_string, &result_limit };

		cb.callp(args, 2, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat(RTR("Failed to execute help search callback: %s."), Variant::get_callable_error_text(cb, args, 2, ce)));
		}
		Dictionary results = ret;
		for (const Variant *E = results.next(); E; E = results.next(E)) {
			const String &key = *E;
			const String &value = results[*E];
			if (key.length() > 0 && value.length() > 0) {
				NSArray *item = @[ [NSString stringWithUTF8String:key.utf8().get_data()], [NSString stringWithUTF8String:value.utf8().get_data()] ];
				[found_items addObject:item];
			}
		}
	}

	handleMatchedItems(found_items);
}

- (void)performActionForItem:(id)item {
	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds && ds->_help_get_action_callback().is_valid()) {
		Callable cb = ds->_help_get_action_callback();

		Variant ret;
		Variant item_string = String::utf8([item[0] UTF8String]);
		Callable::CallError ce;
		const Variant *args[1] = { &item_string };

		cb.callp(args, 1, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat(RTR("Failed to execute help action callback: %s."), Variant::get_callable_error_text(cb, args, 1, ce)));
		}
	}
}

- (void)system_theme_changed:(NSNotification *)notification {
	DisplayServerMacOSBase *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds) {
		ds->emit_system_theme_changed();
	}
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
	os_mac->start_main();
}

static const char *godot_ac_ctx = "gd_accessibility_observer_ctx";

- (void)dealloc {
	[[NSDistributedNotificationCenter defaultCenter] removeObserver:self name:@"AppleInterfaceThemeChangedNotification" object:nil];
	[[NSDistributedNotificationCenter defaultCenter] removeObserver:self name:@"AppleColorPreferencesChangedNotification" object:nil];
	[[NSWorkspace sharedWorkspace] removeObserver:self forKeyPath:@"voiceOverEnabled" context:(void *)godot_ac_ctx];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context {
	if (context == (void *)godot_ac_ctx) {
		voice_over = [[NSWorkspace sharedWorkspace] isVoiceOverEnabled];
	} else {
		[super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
	}
}

- (void)accessibilityDisplayOptionsChange:(NSNotification *)notification {
	high_contrast = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldIncreaseContrast];
	reduce_motion = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceMotion];
	reduce_transparency = [[NSWorkspace sharedWorkspace] accessibilityDisplayShouldReduceTransparency];
}

- (bool)getHighContrast {
	return high_contrast;
}

- (bool)getReduceMotion {
	return reduce_motion;
}

- (bool)getReduceTransparency {
	return reduce_transparency;
}

- (bool)getVoiceOver {
	return voice_over;
}

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
	List<String> args;
	for (NSURL *url in urls) {
		if ([url isFileURL]) {
			args.push_back(String::utf8([url.path UTF8String]));
		} else {
			args.push_back(vformat("--uri=\"%s\"", String::utf8([url.absoluteString UTF8String])));
		}
	}
	if (!args.is_empty()) {
		if (os_mac->get_main_loop()) {
			// Application is already running, open a new instance with the URL/files as command line arguments.
			os_mac->create_instance(args);
		} else if (os_mac->get_cmd_argc() == 0) {
			// Application is just started, add to the list of command line arguments and continue.
			os_mac->set_cmdline_platform_args(args);
		}
	}
}

- (void)application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames {
	List<String> args;
	for (NSString *filename in filenames) {
		NSURL *url = [NSURL URLWithString:filename];
		args.push_back(String::utf8([url.path UTF8String]));
	}
	if (!args.is_empty()) {
		if (os_mac->get_main_loop()) {
			// Application is already running, open a new instance with the URL/files as command line arguments.
			os_mac->create_instance(args);
		} else if (os_mac->get_cmd_argc() == 0) {
			// Application is just started, add to the list of command line arguments and continue.
			os_mac->set_cmdline_platform_args(args);
		}
	}
}

- (void)applicationDidResignActive:(NSNotification *)notification {
	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds) {
		ds->mouse_process_popups(true);
	}
	if (os_mac->get_main_loop()) {
		os_mac->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
	}
}

static const CGKeyCode modifiers[8] = {
	kVK_Command,
	kVK_RightCommand,
	kVK_Shift,
	kVK_RightShift,
	kVK_Option,
	kVK_RightOption,
	kVK_Control,
	kVK_RightControl,
};

// The list of modifier flags we care about for raising pressed events when the application becomes active.
constexpr static NSEventModifierFlags FLAGS = NSEventModifierFlagCommand | NSEventModifierFlagShift | NSEventModifierFlagOption | NSEventModifierFlagControl;

- (void)applicationDidBecomeActive:(NSNotification *)notification {
	if (os_mac->get_main_loop()) {
		os_mac->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
	}
	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (!ds) {
		return;
	}
	Input *input = Input::get_singleton();
	if (!input) {
		return;
	}

	// Poll the modifier keys and submit pressed events if they are down when the application becomes active.
	int mod = NSEvent.modifierFlags;
	if ((mod & FLAGS) == 0) {
		// No flags we care about.
		return;
	}

	DisplayServer::WindowID window_id = ds->get_focused_window();
	NSEventModifierFlags flags = static_cast<NSEventModifierFlags>(mod);

	for (const CGKeyCode key : modifiers) {
		bool is_down = CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, key);
		if (likely(!is_down)) {
			continue;
		}
		Ref<InputEventKey> ke;
		ke.instantiate();

		ke->set_window_id(window_id);
		ke->set_echo(false);
		ke->set_pressed(true);
		ds->get_key_modifier_state(flags, ke);
		ke->set_keycode(KeyMappingMacOS::remap_key(key, mod, false));
		ke->set_physical_keycode(KeyMappingMacOS::translate_key(key));
		ke->set_key_label(KeyMappingMacOS::remap_key(key, mod, true));
		ke->set_location(KeyMappingMacOS::translate_location(key));
		input->parse_input_event(ke);
	}
}

- (BOOL)validateMenuItem:(NSMenuItem *)item {
	if (item) {
		GodotMenuItem *value = [item representedObject];
		if (value) {
			return value->enabled;
		}
	}
	return YES;
}

- (void)globalMenuCallback:(id)sender {
	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds) {
		return ds->menu_callback(sender);
	}
}

- (NSMenu *)applicationDockMenu:(NSApplication *)sender {
	if (NativeMenu::get_singleton()) {
		NativeMenuMacOS *nmenu = (NativeMenuMacOS *)NativeMenu::get_singleton();
		return nmenu->_get_dock_menu();
	} else {
		return nullptr;
	}
}

- (void)applicationWillTerminate:(NSNotification *)notification {
	os_mac->cleanup();
	exit(os_mac->get_exit_code());
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
	if (os_mac->os_should_terminate()) {
		return NSTerminateNow;
	}

	DisplayServerMacOS *ds = Object::cast_to<DisplayServerMacOS>(DisplayServer::get_singleton());
	if (ds && ds->has_window(DisplayServerMacOS::MAIN_WINDOW_ID)) {
		ds->send_window_event(ds->get_window(DisplayServerMacOS::MAIN_WINDOW_ID), DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
	}

	return NSTerminateCancel;
}

- (void)showAbout:(id)sender {
	if (os_mac->get_main_loop()) {
		os_mac->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_ABOUT);
	}
}

@end
