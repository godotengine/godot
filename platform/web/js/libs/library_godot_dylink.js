// SPDX-License-Identifier: MIT
// 2dog: this file is part of https://2dog.dev

// 2dog: load GDExtension side modules against the statically linked main module.
// 2dog: the host must export their imports because Emscripten's dynamic linker requires relocatable builds.
const GodotDylink = {
	$GodotDylink__deps: ['$addFunction', '$updateTableMap', '$functionsInTableMap', '$wasmTable', '$alignMemory', '$zeroMemory', '$UTF8ArrayToString', 'malloc'],
	// 2dog: threaded builds: each pthread instantiates the loaded side modules before its entry point runs.
	$GodotDylink__postset: 'if (typeof PThread !== "undefined") { PThread.tlsInitFunctions.push(() => wasmExports["godot_web_sync_side_modules"]()); }',
	$GodotDylink: {
		libs: [null],
		lastError: '',
		stackPointer: null,

		// 2dog: side modules share a shadow stack because the main stack pointer is private.
		// 2dog: frames stay LIFO; longjmp through side frames cannot unwind this stack.
		STACK_SIZE: 1024 * 1024,

		metadata: function (module) {
			const sections = WebAssembly.Module.customSections(module, 'dylink.0');
			if (sections.length === 0) {
				throw new Error('not a side module (no dylink.0 section); build it with -sSIDE_MODULE');
			}
			const bytes = new Uint8Array(sections[0]);
			let offset = 0;
			const leb = () => {
				let ret = 0;
				let mul = 1;
				for (;;) {
					const b = bytes[offset++];
					ret += (b & 0x7f) * mul;
					mul *= 0x80;
					if (!(b & 0x80)) {
						return ret;
					}
				}
			};
			const str = () => {
				const len = leb();
				offset += len;
				return UTF8ArrayToString(bytes, offset - len, len);
			};
			const meta = { memorySize: 0, memoryAlign: 0, tableSize: 0, needed: [], weak: new Set(), tls: new Set() };
			while (offset < bytes.length) {
				const type = bytes[offset++];
				const size = leb();
				const end = offset + size;
				if (type === 1) { // 2dog: WASM_DYLINK_MEM_INFO
					meta.memorySize = leb();
					meta.memoryAlign = leb();
					meta.tableSize = leb();
				} else if (type === 2) { // 2dog: WASM_DYLINK_NEEDED
					for (let n = leb(); n > 0; n--) {
						meta.needed.push(str());
					}
				} else if (type === 3) { // 2dog: WASM_DYLINK_EXPORT_INFO
					for (let n = leb(); n > 0; n--) {
						const name = str();
						if (leb() & 0x100) { // 2dog: WASM_SYMBOL_TLS
							meta.tls.add(name);
						}
					}
				} else if (type === 4) { // 2dog: WASM_DYLINK_IMPORT_INFO
					for (let n = leb(); n > 0; n--) {
						str(); // 2dog: Module name.
						const name = str();
						if ((leb() & 0x3) === 0x1) { // 2dog: WASM_SYMBOL_BINDING_WEAK
							meta.weak.add(name);
						}
					}
				}
				offset = end;
			}
			return meta;
		},

		has: function (obj, name) {
			return Object.prototype.hasOwnProperty.call(obj, name);
		},

		// 2dog: resolve symbols from wasm exports, imported JS library functions, or Module exports.
		mainSymbol: function (name) {
			if (GodotDylink.has(wasmExports, name)) {
				return wasmExports[name];
			}
			if (GodotDylink.has(wasmImports, name)) {
				return wasmImports[name];
			}
			return Module[`_${name}`];
		},

		sharedStackPointer: function () {
			if (!GodotDylink.stackPointer) {
				const low = _malloc(GodotDylink.STACK_SIZE);
				if (!low) {
					throw new Error('out of memory allocating the side module stack');
				}
				const high = (low + GodotDylink.STACK_SIZE) & ~15;
				GodotDylink.stackPointer = new WebAssembly.Global({ 'value': 'i32', 'mutable': true }, high);
				GodotDylink.stackPointer.high = high;
				GodotDylink.stackPointer.low = low;
			}
			return GodotDylink.stackPointer;
		},

		// 2dog: shared libraries give each GOT.func symbol and function export one fixed slot after the module's own
		// table segment, so their function pointers are the same on every thread.
		sharedSlots: function (module, base) {
			const slots = new Map();
			const add = (sym) => {
				if (!slots.has(sym)) {
					slots.set(sym, base + slots.size);
				}
			};
			for (const imp of WebAssembly.Module.imports(module)) {
				if (imp.module === 'GOT.func') {
					add(imp.name);
				}
			}
			for (const exp of WebAssembly.Module.exports(module)) {
				if (exp.kind === 'function') {
					add(exp.name);
				}
			}
			return slots;
		},

		// 2dog: threaded hosts link with -sTABLE_BASE, which leaves the slots below the main module's functions empty on
		// every thread. Shared libraries take theirs from there: the .NET jiterpreter reserves table slots per thread at
		// the table's end and requires the same indices on every thread, so side modules must not grow the table.
		sharedTableEnd: function () {
			if (GodotDylink.tableEnd === undefined) {
				let end = 1;
				while (end < wasmTable.length && !wasmTable.get(end)) {
					end++;
				}
				GodotDylink.tableEnd = end;
			}
			return GodotDylink.tableEnd;
		},

		// 2dog: shared is { index, layout } in threaded builds, where wasm tables are per thread: the first loader picks
		// the memory and table layout, and every other thread instantiates the module again at that layout.
		load: function (name, bytes, shared) {
			if (shared) {
				// 2dog: a worker keeps its instances across the pthreads it hosts.
				if (GodotDylink.libs[shared.index + 1]) {
					return shared.index + 1;
				}
			} else {
				// 2dog: reuse open libraries rather than instantiating them twice.
				const open = GodotDylink.libs.findIndex((l) => l && l.name === name);
				if (open > 0) {
					GodotDylink.libs[open].refs++;
					return open;
				}
			}

			// 2dog: compile before malloc can grow memory and detach the byte view.
			const module = new WebAssembly.Module(bytes);
			const meta = GodotDylink.metadata(module);
			if (meta.tls.size > 0) {
				throw new Error('thread-local exports are not supported (avoid exported thread_local variables)');
			}
			const needed = meta.needed.map((lib) => {
				const found = GodotDylink.libs.find((l) => l && l.name === lib);
				if (!found) {
					throw new Error(`needs '${lib}', which is not loaded (list it under [dependencies])`);
				}
				return found;
			});
			const fromNeeded = (sym) => {
				for (const lib of needed) {
					if (GodotDylink.has(lib.exports, sym)) {
						return lib.exports[sym];
					}
				}
				return undefined;
			};

			const layout = shared && shared.layout;
			const memAlign = Math.pow(2, meta.memoryAlign);
			let memoryBase = 0;
			if (layout) {
				// 2dog: the first loader allocated the data in shared memory; its start function initialized it once.
				memoryBase = layout.memoryBase;
			} else if (meta.memorySize) {
				const raw = _malloc(meta.memorySize + memAlign);
				if (!raw) {
					throw new Error('out of memory allocating the side module data');
				}
				memoryBase = alignMemory(zeroMemory(raw, meta.memorySize + memAlign), memAlign);
			}
			let tableBase = 0;
			let tableSize = meta.tableSize;
			let slots = null;
			if (shared) {
				if (layout) {
					tableBase = layout.tableBase;
					GodotDylink.tableEnd = layout.tableEnd;
				} else {
					// 2dog: the loader has every earlier library, so the next free slot follows the last of them.
					tableBase = GodotDylink.libs.reduce((next, l) => (l ? Math.max(next, l.layout.tableBase + l.layout.tableSize) : next), 1);
				}
				slots = GodotDylink.sharedSlots(module, tableBase + meta.tableSize);
				tableSize += slots.size;
				const tableEnd = GodotDylink.sharedTableEnd();
				if (tableBase + tableSize > tableEnd) {
					throw new Error(`needs ${tableSize} function table slots, but ${Math.max(tableEnd - tableBase, 0)} of those reserved for side modules are free (raise TwoDogWebSideModuleTableSlots)`);
				}
				for (let i = tableBase; i < tableBase + tableSize; i++) {
					if (wasmTable.get(i)) {
						throw new Error(`function table slot ${i} is already in use on this thread`);
					}
				}
			} else if (meta.tableSize) {
				tableBase = wasmTable.length;
				wasmTable.grow(meta.tableSize);
			}

			let exports = null;
			const unresolved = [];
			const got = [];
			const env = {};
			const imports = { 'env': env, 'wasi_snapshot_preview1': env, 'GOT.mem': {}, 'GOT.func': {} };
			for (const imp of WebAssembly.Module.imports(module)) {
				const sym = imp.name;
				if (imp.module === 'GOT.mem' || imp.module === 'GOT.func') {
					// 2dog: fill symbols after instantiation, before data relocations and constructors read them.
					const entry = new WebAssembly.Global({ 'value': 'i32', 'mutable': true }, 0);
					got.push({ entry, sym, func: imp.module === 'GOT.func' });
					imports[imp.module][sym] = entry;
				} else if (imp.module !== 'env' && imp.module !== 'wasi_snapshot_preview1') {
					throw new Error(`unsupported import module '${imp.module}' (${sym})`);
				} else if (imp.kind === 'memory') {
					env[sym] = wasmMemory;
				} else if (imp.kind === 'table') {
					env[sym] = wasmTable;
				} else if (sym === '__memory_base') {
					env[sym] = new WebAssembly.Global({ 'value': 'i32', 'mutable': false }, memoryBase);
				} else if (sym === '__table_base') {
					env[sym] = new WebAssembly.Global({ 'value': 'i32', 'mutable': false }, tableBase);
				} else if (sym === '__stack_pointer') {
					env[sym] = GodotDylink.sharedStackPointer();
				} else if (imp.kind === 'function') {
					let func = GodotDylink.mainSymbol(sym);
					if (typeof func !== 'function') {
						func = fromNeeded(sym);
					}
					if (typeof func !== 'function') {
						// 2dog: defer missing-symbol failures until calls; modules may define or never use the symbol.
						if (!meta.weak.has(sym)) {
							unresolved.push(sym);
						}
						func = (...args) => {
							const own = exports && exports[sym];
							if (typeof own !== 'function') {
								throw new Error(`${name}: called '${sym}', which the main module does not export`);
							}
							return own(...args);
						};
					}
					env[sym] = func;
				} else if (imp.kind === 'tag') {
					// 2dog: share main-module exception tags for interop; private tags work only within the side module.
					const tag = GodotDylink.mainSymbol(sym);
					env[sym] = tag instanceof WebAssembly.Tag ? tag : new WebAssembly.Tag({ 'parameters': ['i32'] });
				} else {
					const value = GodotDylink.mainSymbol(sym);
					if (value === undefined) {
						throw new Error(`unresolved ${imp.kind} import '${sym}' (not exported by the main module)`);
					}
					env[sym] = value;
				}
			}
			if (unresolved.length) {
				err(`GDExtension '${name}': imports not exported by the main module (calls will fail): ${unresolved.join(', ')}`);
			}

			const instance = new WebAssembly.Instance(module, imports);

			// 2dog: exported data symbols are relative to the module's memory base.
			exports = {};
			for (const [sym, value] of Object.entries(instance.exports)) {
				exports[sym] = value instanceof WebAssembly.Global ? value.value + memoryBase : value;
			}
			if (functionsInTableMap) {
				updateTableMap(tableBase, meta.tableSize);
			}
			if (slots) {
				for (const [sym, slot] of slots) {
					if (typeof exports[sym] === 'function') {
						wasmTable.set(slot, exports[sym]);
					}
				}
			}

			// 2dog: prefer a module's own definitions so extensions never bind to each other's data.
			for (const { entry, sym, func } of got) {
				let value = GodotDylink.has(exports, sym) ? exports[sym] : undefined;
				if (value === undefined) {
					value = GodotDylink.mainSymbol(sym);
					if (value instanceof WebAssembly.Global) {
						value = value.value;
					}
				}
				if (value === undefined) {
					value = fromNeeded(sym);
				}
				if (value === undefined) {
					if (meta.weak.has(sym)) {
						continue;
					}
					throw new Error(`unresolved ${func ? 'function' : 'data'} symbol '${sym}' (not exported by the main module)`);
				}
				if (func) {
					if (typeof value !== 'function') {
						throw new Error(`symbol '${sym}' is not a function`);
					}
					if (slots) {
						entry.value = slots.get(sym);
						wasmTable.set(entry.value, value);
					} else {
						// 2dog: reuse table slots through addFunction to preserve function-pointer equality.
						entry.value = addFunction(value);
					}
				} else {
					if (typeof value !== 'number') {
						throw new Error(`symbol '${sym}' is not a data address`);
					}
					entry.value = value;
				}
			}

			if (instance.exports['__set_stack_limits']) {
				const sp = GodotDylink.sharedStackPointer();
				instance.exports['__set_stack_limits'](sp.high, sp.low);
			}
			if (slots && functionsInTableMap) {
				updateTableMap(tableBase + meta.tableSize, slots.size);
			}
			// 2dog: shared memory holds one copy of the data, so only its first loader relocates and constructs it.
			if (!layout) {
				if (instance.exports['__wasm_apply_data_relocs']) {
					instance.exports['__wasm_apply_data_relocs']();
				}
				if (instance.exports['__wasm_call_ctors']) {
					instance.exports['__wasm_call_ctors']();
				}
			}

			const lib = { name, exports, refs: 1, slots, layout: { memoryBase, tableBase, tableSize, tableEnd: GodotDylink.tableEnd } };
			if (shared) {
				GodotDylink.libs[shared.index + 1] = lib;
				return shared.index + 1;
			}
			GodotDylink.libs.push(lib);
			return GodotDylink.libs.length - 1;
		},

		symbol: function (handle, sym) {
			const lib = GodotDylink.libs[handle];
			if (!lib) {
				throw new Error(`invalid library handle ${handle}`);
			}
			if (!GodotDylink.has(lib.exports, sym)) {
				throw new Error(`symbol '${sym}' not found in '${lib.name}'`);
			}
			const value = lib.exports[sym];
			if (typeof value !== 'function') {
				return value;
			}
			return lib.slots ? lib.slots.get(sym) : addFunction(value);
		},
	},

	// 2dog: the loader runs on the calling thread: with threads, wasm instances and function tables are per thread.
	godot_js_dylink_open__sig: 'ippi',
	godot_js_dylink_open: function (p_name, p_bytes, p_size) {
		const name = UTF8ToString(p_name);
		try {
			return GodotDylink.load(name, HEAPU8.subarray(p_bytes, p_bytes + p_size));
		} catch (e) {
			GodotDylink.lastError = e.message || String(e);
			return 0;
		}
	},

	// 2dog: threaded builds; p_layout is null for the first loader, which returns the layout other threads load at.
	godot_js_dylink_thread_load__deps: ['$stringToNewUTF8'],
	godot_js_dylink_thread_load__sig: 'pippip',
	godot_js_dylink_thread_load: function (p_index, p_name, p_bytes, p_size, p_layout) {
		const name = UTF8ToString(p_name);
		try {
			const layout = p_layout ? JSON.parse(UTF8ToString(p_layout)) : null;
			const handle = GodotDylink.load(name, HEAPU8.subarray(p_bytes, p_bytes + p_size), { index: p_index, layout });
			return p_layout || stringToNewUTF8(JSON.stringify(GodotDylink.libs[handle].layout));
		} catch (e) {
			GodotDylink.lastError = e.message || String(e);
			if (p_layout) {
				err(`GDExtension '${name}' failed to load on this thread (calls from it will fail): ${GodotDylink.lastError}`);
			}
			return 0;
		}
	},

	godot_js_dylink_symbol__sig: 'pip',
	godot_js_dylink_symbol: function (p_handle, p_symbol) {
		try {
			return GodotDylink.symbol(p_handle, UTF8ToString(p_symbol));
		} catch (e) {
			GodotDylink.lastError = e.message || String(e);
			return 0;
		}
	},

	godot_js_dylink_close__sig: 'vi',
	godot_js_dylink_close: function (p_handle) {
		// 2dog: drop closed handles; reopening creates a new instance because wasm instances cannot unload.
		const lib = GodotDylink.libs[p_handle];
		if (p_handle > 0 && lib && --lib.refs === 0) {
			GodotDylink.libs[p_handle] = null;
		}
	},

	godot_js_dylink_error__deps: ['$stringToNewUTF8'],
	godot_js_dylink_error__sig: 'p',
	godot_js_dylink_error: function () {
		return stringToNewUTF8(GodotDylink.lastError);
	},
};

autoAddDeps(GodotDylink, '$GodotDylink');
addToLibrary(GodotDylink);
