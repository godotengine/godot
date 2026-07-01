// string_marshal.js
// Helper for passing .NET strings into the Godot/Emscripten module.
// Strings are encoded as UTF-8 on the Emscripten stack and must be used
// immediately by the target call.
export class StringMarshaller {
	constructor(TheGodotModule) {
		// Emscripten Module for the active Godot runtime.
		this.TheGodotModule = TheGodotModule;
	}

	marshal(value) {
		// Convert the incoming value to a string and place it as UTF-8 on the Emscripten stack. The returned pointer is only valid for the duration of the current native call.
		return this.TheGodotModule.stringToUTF8OnStack(String(value));
	}
}
