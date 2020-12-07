const Engine = (function () {
	const preloader = new Preloader();

	let wasmExt = '.wasm';
	let unloadAfterInit = true;
	let loadPath = '';
	let loadPromise = null;
	let initPromise = null;
	let stderr = null;
	let stdout = null;
	let progressFunc = null;

	function load(basePath) {
		if (loadPromise == null) {
			loadPath = basePath;
			loadPromise = preloader.loadPromise(basePath + wasmExt);
			preloader.setProgressFunc(progressFunc);
			requestAnimationFrame(preloader.animateProgress);
		}
		return loadPromise;
	}

	function unload() {
		loadPromise = null;
	}

	/** @constructor */
	function Engine() { // eslint-disable-line no-shadow
		this.canvas = null;
		this.executableName = '';
		this.rtenv = null;
		this.customLocale = null;
		this.resizeCanvasOnStart = false;
		this.onExecute = null;
		this.onExit = null;
		this.persistentPaths = ['/userfs'];
		this.gdnativeLibs = [];
	}

	Engine.prototype.init = /** @param {string=} basePath */ function (basePath) {
		if (initPromise) {
			return initPromise;
		}
		if (loadPromise == null) {
			if (!basePath) {
				initPromise = Promise.reject(new Error('A base path must be provided when calling `init` and the engine is not loaded.'));
				return initPromise;
			}
			load(basePath);
		}
		let config = {};
		if (typeof stdout === 'function') {
			config.print = stdout;
		}
		if (typeof stderr === 'function') {
			config.printErr = stderr;
		}
		const me = this;
		initPromise = new Promise(function (resolve, reject) {
			config['locateFile'] = Utils.createLocateRewrite(loadPath);
			config['instantiateWasm'] = Utils.createInstantiatePromise(loadPromise);
			// Emscripten configuration.
			config['thisProgram'] = me.executableName;
			config['noExitRuntime'] = true;
			config['dynamicLibraries'] = me.gdnativeLibs;
			Godot(config).then(function (module) {
				module['initFS'](me.persistentPaths).then(function (fs_err) {
					me.rtenv = module;
					if (unloadAfterInit) {
						unload();
					}
					resolve();
					config = null;
				});
			});
		});
		return initPromise;
	};

	/** @type {function(string, string):Object} */
	Engine.prototype.preloadFile = function (file, path) {
		return preloader.preload(file, path);
	};

	/** @type {function(...string):Object} */
	Engine.prototype.start = function () {
		// Start from arguments.
		const args = [];
		for (let i = 0; i < arguments.length; i++) {
			args.push(arguments[i]);
		}
		const me = this;
		return me.init().then(function () {
			if (!me.rtenv) {
				return Promise.reject(new Error('The engine must be initialized before it can be started'));
			}

			if (!(me.canvas instanceof HTMLCanvasElement)) {
				me.canvas = Utils.findCanvas();
				if (!me.canvas) {
					return Promise.reject(new Error('No canvas found in page'));
				}
			}

			// Canvas can grab focus on click, or key events won't work.
			if (me.canvas.tabIndex < 0) {
				me.canvas.tabIndex = 0;
			}

			// Disable right-click context menu.
			me.canvas.addEventListener('contextmenu', function (ev) {
				ev.preventDefault();
			}, false);

			// Until context restoration is implemented warn the user of context loss.
			me.canvas.addEventListener('webglcontextlost', function (ev) {
				alert('WebGL context lost, please reload the page'); // eslint-disable-line no-alert
				ev.preventDefault();
			}, false);

			// Browser locale, or custom one if defined.
			let locale = me.customLocale;
			if (!locale) {
				locale = navigator.languages ? navigator.languages[0] : navigator.language;
				locale = locale.split('.')[0];
			}
			// Godot configuration.
			me.rtenv['initConfig']({
				'resizeCanvasOnStart': me.resizeCanvasOnStart,
				'canvas': me.canvas,
				'locale': locale,
				'onExecute': function (p_args) {
					if (me.onExecute) {
						me.onExecute(p_args);
						return 0;
					}
					return 1;
				},
				'onExit': function (p_code) {
					me.rtenv['deinitFS']();
					if (me.onExit) {
						me.onExit(p_code);
					}
					me.rtenv = null;
				},
			});

			return new Promise(function (resolve, reject) {
				preloader.preloadedFiles.forEach(function (file) {
					me.rtenv['copyToFS'](file.path, file.buffer);
				});
				preloader.preloadedFiles.length = 0; // Clear memory
				me.rtenv['callMain'](args);
				initPromise = null;
				resolve();
			});
		});
	};

	Engine.prototype.startGame = function (execName, mainPack, extraArgs) {
		// Start and init with execName as loadPath if not inited.
		this.executableName = execName;
		const me = this;
		return Promise.all([
			this.init(execName),
			this.preloadFile(mainPack, mainPack),
		]).then(function () {
			let args = ['--main-pack', mainPack];
			if (extraArgs) {
				args = args.concat(extraArgs);
			}
			return me.start.apply(me, args);
		});
	};

	Engine.prototype.setWebAssemblyFilenameExtension = function (override) {
		if (String(override).length === 0) {
			throw new Error('Invalid WebAssembly filename extension override');
		}
		wasmExt = String(override);
	};

	Engine.prototype.setUnloadAfterInit = function (enabled) {
		unloadAfterInit = enabled;
	};

	Engine.prototype.setCanvas = function (canvasElem) {
		this.canvas = canvasElem;
	};

	Engine.prototype.setCanvasResizedOnStart = function (enabled) {
		this.resizeCanvasOnStart = enabled;
	};

	Engine.prototype.setLocale = function (locale) {
		this.customLocale = locale;
	};

	Engine.prototype.setExecutableName = function (newName) {
		this.executableName = newName;
	};

	Engine.prototype.setProgressFunc = function (func) {
		progressFunc = func;
	};

	Engine.prototype.setStdoutFunc = function (func) {
		const print = function (text) {
			let msg = text;
			if (arguments.length > 1) {
				msg = Array.prototype.slice.call(arguments).join(' ');
			}
			func(msg);
		};
		if (this.rtenv) {
			this.rtenv.print = print;
		}
		stdout = print;
	};

	Engine.prototype.setStderrFunc = function (func) {
		const printErr = function (text) {
			let msg = text;
			if (arguments.length > 1) {
				msg = Array.prototype.slice.call(arguments).join(' ');
			}
			func(msg);
		};
		if (this.rtenv) {
			this.rtenv.printErr = printErr;
		}
		stderr = printErr;
	};

	Engine.prototype.setOnExecute = function (onExecute) {
		this.onExecute = onExecute;
	};

	Engine.prototype.setOnExit = function (onExit) {
		this.onExit = onExit;
	};

	Engine.prototype.copyToFS = function (path, buffer) {
		if (this.rtenv == null) {
			throw new Error('Engine must be inited before copying files');
		}
		this.rtenv['copyToFS'](path, buffer);
	};

	Engine.prototype.setPersistentPaths = function (persistentPaths) {
		this.persistentPaths = persistentPaths;
	};

	Engine.prototype.setGDNativeLibraries = function (gdnativeLibs) {
		this.gdnativeLibs = gdnativeLibs;
	};

	Engine.prototype.requestQuit = function () {
		if (this.rtenv) {
			this.rtenv['request_quit']();
		}
	};

	// Closure compiler exported engine methods.
	/** @export */
	Engine['isWebGLAvailable'] = Utils.isWebGLAvailable;
	Engine['load'] = load;
	Engine['unload'] = unload;
	Engine.prototype['init'] = Engine.prototype.init;
	Engine.prototype['preloadFile'] = Engine.prototype.preloadFile;
	Engine.prototype['start'] = Engine.prototype.start;
	Engine.prototype['startGame'] = Engine.prototype.startGame;
	Engine.prototype['setWebAssemblyFilenameExtension'] = Engine.prototype.setWebAssemblyFilenameExtension;
	Engine.prototype['setUnloadAfterInit'] = Engine.prototype.setUnloadAfterInit;
	Engine.prototype['setCanvas'] = Engine.prototype.setCanvas;
	Engine.prototype['setCanvasResizedOnStart'] = Engine.prototype.setCanvasResizedOnStart;
	Engine.prototype['setLocale'] = Engine.prototype.setLocale;
	Engine.prototype['setExecutableName'] = Engine.prototype.setExecutableName;
	Engine.prototype['setProgressFunc'] = Engine.prototype.setProgressFunc;
	Engine.prototype['setStdoutFunc'] = Engine.prototype.setStdoutFunc;
	Engine.prototype['setStderrFunc'] = Engine.prototype.setStderrFunc;
	Engine.prototype['setOnExecute'] = Engine.prototype.setOnExecute;
	Engine.prototype['setOnExit'] = Engine.prototype.setOnExit;
	Engine.prototype['copyToFS'] = Engine.prototype.copyToFS;
	Engine.prototype['setPersistentPaths'] = Engine.prototype.setPersistentPaths;
	Engine.prototype['setGDNativeLibraries'] = Engine.prototype.setGDNativeLibraries;
	Engine.prototype['requestQuit'] = Engine.prototype.requestQuit;
	return Engine;
}());
if (typeof window !== 'undefined') {
	window['Engine'] = Engine;
}
