		// The following is concatenated with generated code, and acts as the end
		// of a wrapper for said code. See pre.js for the other part of the
		// wrapper.
		exposedLibs['PATH'] = PATH;
		exposedLibs['FS'] = FS;
		return Module;
	},
};

(function() {
	var engine = Engine;

	var DOWNLOAD_ATTEMPTS_MAX = 4;

	var basePath = null;
	var wasmFilenameExtensionOverride = null;
	var engineLoadPromise = null;

	var loadingFiles = {};

	function getPathLeaf(path) {

		while (path.endsWith('/'))
			path = path.slice(0, -1);
		return path.slice(path.lastIndexOf('/') + 1);
	}

	function getBasePath(path) {

		if (path.endsWith('/'))
			path = path.slice(0, -1);
		if (path.lastIndexOf('.') > path.lastIndexOf('/'))
			path = path.slice(0, path.lastIndexOf('.'));
		return path;
	}

	function getBaseName(path) {

		return getPathLeaf(getBasePath(path));
	}

	Engine = function Engine() {

		this.rtenv = null;

		var LIBS = {};

		var initPromise = null;
		var unloadAfterInit = true;

		var preloadedFiles = [];

		var resizeCanvasOnStart = true;
		var progressFunc = null;
		var preloadProgressTracker = {};
		var lastProgress = { loaded: 0, total: 0 };

		var canvas = null;
		var executableName = null;
		var locale = null;
		var stdout = null;
		var stderr = null;

		this.init = function(newBasePath) {

			if (!initPromise) {
				initPromise = Engine.load(newBasePath).then(
					instantiate.bind(this)
				);
				requestAnimationFrame(animateProgress);
				if (unloadAfterInit)
					initPromise.then(Engine.unloadEngine);
			}
			return initPromise;
		};

		function instantiate(wasmBuf) {

			var rtenvProps = {
				engine: this,
				ENV: {},
			};
			if (typeof stdout === 'function')
				rtenvProps.print = stdout;
			if (typeof stderr === 'function')
				rtenvProps.printErr = stderr;
			rtenvProps.instantiateWasm = function(imports, onSuccess) {
				WebAssembly.instantiate(wasmBuf, imports).then(function(result) {
					onSuccess(result.instance);
				});
				return {};
			};

			return new Promise(function(resolve, reject) {
				rtenvProps.onRuntimeInitialized = resolve;
				rtenvProps.onAbort = reject;
				rtenvProps.thisProgram = executableName;
				rtenvProps.engine.rtenv = Engine.RuntimeEnvironment(rtenvProps, LIBS);
			});
		}

		this.preloadFile = function(pathOrBuffer, destPath) {

			if (pathOrBuffer instanceof ArrayBuffer) {
				pathOrBuffer = new Uint8Array(pathOrBuffer);
			} else if (ArrayBuffer.isView(pathOrBuffer)) {
				pathOrBuffer = new Uint8Array(pathOrBuffer.buffer);
			}
			if (pathOrBuffer instanceof Uint8Array) {
				preloadedFiles.push({
					path: destPath,
					buffer: pathOrBuffer
				});
				return Promise.resolve();
			} else if (typeof pathOrBuffer === 'string') {
				return loadPromise(pathOrBuffer, preloadProgressTracker).then(function(xhr) {
					preloadedFiles.push({
						path: destPath || pathOrBuffer,
						buffer: xhr.response
					});
				});
			} else {
				throw Promise.reject("Invalid object for preloading");
			}
		};

		this.start = function() {

			return this.init().then(
				Function.prototype.apply.bind(synchronousStart, this, arguments)
			);
		};

		this.startGame = function(execName, mainPack) {

			executableName = execName;
			var mainArgs = [ '--main-pack', mainPack ];

			return Promise.all([
				// Load from directory,
				this.init(getBasePath(mainPack)),
				// ...but write to root where the engine expects it.
				this.preloadFile(mainPack, getPathLeaf(mainPack))
			]).then(
				Function.prototype.apply.bind(synchronousStart, this, mainArgs)
			);
		};

		function synchronousStart() {

			if (canvas instanceof HTMLCanvasElement) {
				this.rtenv.canvas = canvas;
			} else {
				var firstCanvas = document.getElementsByTagName('canvas')[0];
				if (firstCanvas instanceof HTMLCanvasElement) {
					this.rtenv.canvas = firstCanvas;
				} else {
					throw new Error("No canvas found");
				}
			}

			var actualCanvas = this.rtenv.canvas;
			// canvas can grab focus on click
			if (actualCanvas.tabIndex < 0) {
				actualCanvas.tabIndex = 0;
			}
			// necessary to calculate cursor coordinates correctly
			actualCanvas.style.padding = 0;
			actualCanvas.style.borderWidth = 0;
			actualCanvas.style.borderStyle = 'none';
			// disable right-click context menu
			actualCanvas.addEventListener('contextmenu', function(ev) {
				ev.preventDefault();
			}, false);
			// until context restoration is implemented
			actualCanvas.addEventListener('webglcontextlost', function(ev) {
				alert("WebGL context lost, please reload the page");
				ev.preventDefault();
			}, false);

			if (locale) {
				this.rtenv.locale = locale;
			} else {
				this.rtenv.locale = navigator.languages ? navigator.languages[0] : navigator.language;
			}
			this.rtenv.locale = this.rtenv.locale.split('.')[0];
			this.rtenv.resizeCanvasOnStart = resizeCanvasOnStart;

			preloadedFiles.forEach(function(file) {
				var dir = LIBS.PATH.dirname(file.path);
				try {
					LIBS.FS.stat(dir);
				} catch (e) {
					if (e.code !== 'ENOENT') {
						throw e;
					}
					LIBS.FS.mkdirTree(dir);
				}
				// With memory growth, canOwn should be false.
				LIBS.FS.createDataFile(file.path, null, new Uint8Array(file.buffer), true, true, false);
			}, this);

			preloadedFiles = null;
			initPromise = null;
			this.rtenv.callMain(arguments);
		}

		this.setProgressFunc = function(func) {
			progressFunc = func;
		};

		this.setResizeCanvasOnStart = function(enabled) {
			resizeCanvasOnStart = enabled;
		};

		function animateProgress() {

			var loaded = 0;
			var total = 0;
			var totalIsValid = true;
			var progressIsFinal = true;

			[loadingFiles, preloadProgressTracker].forEach(function(tracker) {
				Object.keys(tracker).forEach(function(file) {
					if (!tracker[file].final)
						progressIsFinal = false;
					if (!totalIsValid || tracker[file].total === 0) {
						totalIsValid = false;
						total = 0;
					} else {
						total += tracker[file].total;
					}
					loaded += tracker[file].loaded;
				});
			});
			if (loaded !== lastProgress.loaded || total !== lastProgress.total) {
				lastProgress.loaded = loaded;
				lastProgress.total = total;
				if (typeof progressFunc === 'function')
					progressFunc(loaded, total);
			}
			if (!progressIsFinal)
				requestAnimationFrame(animateProgress);
		}

		this.setCanvas = function(elem) {
			canvas = elem;
		};

		this.setExecutableName = function(newName) {

			executableName = newName;
		};

		this.setLocale = function(newLocale) {

			locale = newLocale;
		};

		this.setUnloadAfterInit = function(enabled) {

			if (enabled && !unloadAfterInit && initPromise) {
				initPromise.then(Engine.unloadEngine);
			}
			unloadAfterInit = enabled;
		};

		this.setStdoutFunc = function(func) {

			var print = function(text) {
				if (arguments.length > 1) {
					text = Array.prototype.slice.call(arguments).join(" ");
				}
				func(text);
			};
			if (this.rtenv)
				this.rtenv.print = print;
			stdout = print;
		};

		this.setStderrFunc = function(func) {

			var printErr = function(text) {
				if (arguments.length > 1)
					text = Array.prototype.slice.call(arguments).join(" ");
				func(text);
			};
			if (this.rtenv)
				this.rtenv.printErr = printErr;
			stderr = printErr;
		};


	}; // Engine()

	Engine.RuntimeEnvironment = engine.RuntimeEnvironment;

	Engine.isWebGLAvailable = function(majorVersion = 1) {

		var testContext = false;
		try {
			var testCanvas = document.createElement('canvas');
			if (majorVersion === 1) {
				testContext = testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
			} else if (majorVersion === 2) {
				testContext = testCanvas.getContext('webgl2') || testCanvas.getContext('experimental-webgl2');
			}
		} catch (e) {}
		return !!testContext;
	};

	Engine.setWebAssemblyFilenameExtension = function(override) {

		if (String(override).length === 0) {
			throw new Error('Invalid WebAssembly filename extension override');
		}
		wasmFilenameExtensionOverride = String(override);
	}

	Engine.load = function(newBasePath) {

		if (newBasePath !== undefined) basePath = getBasePath(newBasePath);
		if (engineLoadPromise === null) {
			if (typeof WebAssembly !== 'object')
				return Promise.reject(new Error("Browser doesn't support WebAssembly"));
			// TODO cache/retrieve module to/from idb
			engineLoadPromise = loadPromise(basePath + '.' + (wasmFilenameExtensionOverride || 'wasm')).then(function(xhr) {
				return xhr.response;
			});
			engineLoadPromise = engineLoadPromise.catch(function(err) {
				engineLoadPromise = null;
				throw err;
			});
		}
		return engineLoadPromise;
	};

	Engine.unload = function() {
		engineLoadPromise = null;
	};

	function loadPromise(file, tracker) {
		if (tracker === undefined)
			tracker = loadingFiles;
		return new Promise(function(resolve, reject) {
			loadXHR(resolve, reject, file, tracker);
		});
	}

	function loadXHR(resolve, reject, file, tracker) {

		var xhr = new XMLHttpRequest;
		xhr.open('GET', file);
		if (!file.endsWith('.js')) {
			xhr.responseType = 'arraybuffer';
		}
		['loadstart', 'progress', 'load', 'error', 'abort'].forEach(function(ev) {
			xhr.addEventListener(ev, onXHREvent.bind(xhr, resolve, reject, file, tracker));
		});
		xhr.send();
	}

	function onXHREvent(resolve, reject, file, tracker, ev) {

		if (this.status >= 400) {

			if (this.status < 500 || ++tracker[file].attempts >= DOWNLOAD_ATTEMPTS_MAX) {
				reject(new Error("Failed loading file '" + file + "': " + this.statusText));
				this.abort();
				return;
			} else {
				setTimeout(loadXHR.bind(null, resolve, reject, file, tracker), 1000);
			}
		}

		switch (ev.type) {
			case 'loadstart':
				if (tracker[file] === undefined) {
					tracker[file] = {
						total: ev.total,
						loaded: ev.loaded,
						attempts: 0,
						final: false,
					};
				}
				break;

			case 'progress':
				tracker[file].loaded = ev.loaded;
				tracker[file].total = ev.total;
				break;

			case 'load':
				tracker[file].final = true;
				resolve(this);
				break;

			case 'error':
				if (++tracker[file].attempts >= DOWNLOAD_ATTEMPTS_MAX) {
					tracker[file].final = true;
					reject(new Error("Failed loading file '" + file + "'"));
				} else {
					setTimeout(loadXHR.bind(null, resolve, reject, file, tracker), 1000);
				}
				break;

			case 'abort':
				tracker[file].final = true;
				reject(new Error("Loading file '" + file + "' was aborted."));
				break;
		}
	}
})();
