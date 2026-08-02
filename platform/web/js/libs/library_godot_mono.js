/**
 * Godot Emscripten library for .NET 8 WASM Runtime Integration.
 * 
 * This file binds Godot's C++ Web platform to the .NET WASM runtime (dotnet.js).
 */
const GodotMono = {
	$GodotMono__deps: ['$GodotRuntime', '$GodotConfig'],
	$GodotMono: {
		runtime_initialized: false,
		dotnet_exports: null,

		init_dotnet: function(wasm_file) {
			if (GodotMono.runtime_initialized) return Promise.resolve();

			return new Promise((resolve, reject) => {
				GodotRuntime.print("Initializing .NET WASM runtime...");
				// Load dotnet.js from the same path as the Godot executable
				const dotnet_js_url = GodotConfig.locateFile("dotnet.js");
				
				const script = document.createElement("script");
				script.src = dotnet_js_url;
				script.onload = () => {
					window.dotnet.create().then((dotnet) => {
						GodotMono.dotnet_exports = dotnet.getAssemblyExports("GodotSharp.dll");
						GodotMono.runtime_initialized = true;
						GodotRuntime.print(".NET WASM runtime initialized successfully.");
						resolve();
					}).catch(reject);
				};
				script.onerror = reject;
				document.body.appendChild(script);
			});
		},
	},

	godot_mono_init__proxy: 'sync',
	godot_mono_init__sig: 'ii',
	godot_mono_init: function() {
		// Called from C++ during Engine initialization
		if (GodotMono.runtime_initialized) {
			return 1;
		}
		// Synchronous fallback (or we use Asyncify)
		GodotRuntime.error("GodotMono requires async initialization. Ensure Asyncify is enabled.");
		return 0;
	},
};

autoAddDeps(GodotMono, '$GodotMono');
mergeInto(LibraryManager.library, GodotMono);
