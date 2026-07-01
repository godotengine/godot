/* global Engine */
(async function () {
	const canvas = document.getElementById('canvas');

	const GODOT_CONFIG = {
		args: [],
		canvasResizePolicy: 2,
		executable: 'godot',
		mainPack: 'pong.pck',
		fileSizes: { 'godot.wasm': 0, 'pong.pck': 0 },
		gdextensionlibs: [],
		unloadAfterInit: false,
		canvas,
		focusCanvas: true,
	};

	if (typeof Engine === 'undefined') {
		return;
	}
	const missing = Engine.getMissingFeatures({ threads: false });
	if (missing.length) {
		return;
	}

	window.engine = new Engine(GODOT_CONFIG);

	// Start the Godot engine – no click overlay,
	await window.engine.startGame({
		onProgress: (current, total) => {
			// progress bar is optional; ignore for minimal setup
		},
	});

	// Load the bridge helper modules
	const [
		{ StringMarshaller },
		{ createVariantDecoder },
		{ GODOT_EXPORT_SIGNATURES },
		{ marshalVarargs },
		{ marshalArg },
		{ decodeReturn },
	] = await Promise.all([
		import('./Bridge_Functions/string_marshal.js'),
		import('./Bridge_Functions/variant_decoder.js'),
		import('./Bridge_Functions/export_signatures.js'),
		import('./Bridge_Functions/varargs_marshal.js'),
		import('./Bridge_Functions/marshal_cases.js'),
		import('./Bridge_Functions/return_handlers.js'),
	]);

	const TheGodotModule = window.engine.rtenv;

	function readUTF8(ptr) {
		if (!ptr) {
			return '';
		}
		const heap = new Uint8Array(TheGodotModule.HEAPU8.buffer);
		let end = ptr;
		while (heap[end] !== 0) {
			end++;
		}
		return new TextDecoder().decode(heap.slice(ptr, end));
	}

	const decodeVariant = createVariantDecoder(TheGodotModule, readUTF8);

	// Primary bridge entry point – called by .NET via JSImport
	globalThis.__callGodot = function (fn, ...rawArgs) {
		const wasmFn = TheGodotModule[fn];
		if (!wasmFn) {
			throw new Error(`Missing WASM export: ${fn}`);
		}

		let args = rawArgs;
		if (rawArgs.length === 1 && Array.isArray(rawArgs[0])) {
			args = rawArgs[0];
		}

		const sigEntry
			= GODOT_EXPORT_SIGNATURES[fn]
				?? GODOT_EXPORT_SIGNATURES[fn.startsWith('_') ? fn.slice(1) : `_${fn}`]
				?? {};
		const sig = sigEntry.params ?? [];
		const returnKind = sigEntry.returns ?? 'variant';

		const sm = new StringMarshaller(TheGodotModule);
		const tempPtrs = [];
		const variantPtrs = [];
		const marshaledArgs = [];

		for (let i = 0; i < args.length; i++) {
			marshalArg(
				TheGodotModule,
				sm,
				tempPtrs,
				variantPtrs,
				marshaledArgs,
				sig[i],
				args[i],
				i,
				marshalVarargs
			);
		}

		try {
			const raw = wasmFn(...marshaledArgs);
			return decodeReturn(TheGodotModule, raw, returnKind, readUTF8, decodeVariant);
		} finally {
			for (const vptr of variantPtrs) {
				TheGodotModule._variant_destroy(vptr);
			}
			for (const ptr of tempPtrs) {
				TheGodotModule._free(ptr);
			}
		}
	};

	// Boot .NET runtime and wire up signals / frame loop
	const dotnetJsUrl = new URL('cs/_framework/dotnet.js', window.location.href).href;
	const { dotnet } = await import(dotnetJsUrl);
	const runtime = await dotnet.create();
	globalThis.__dotnetModule = runtime.Module;

	const exports = await runtime.getAssemblyExports('Demo');
	if (!exports?.Interop?.InitInterop) {
		throw new Error('InitInterop missing');
	}

	exports.Interop.InitInterop();

	window._crossRuntimeNotifyCSharp = function (objectId, signalName, args) {
		exports.Godot.SignalManager.Receive(objectId, signalName, args);
	};

	globalThis.__stepFrame = function (delta) {
		exports.Interop.StepFrame(delta);
	};
})();
