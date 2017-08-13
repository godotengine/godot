		return Module;
	},
};

(function() {
	var engine = Engine;

	var USING_WASM = engine.USING_WASM;
	var DOWNLOAD_ATTEMPTS_MAX = 4;

	var basePath = null;
	var engineLoadPromise = null;

	var loadingFiles = {};

	function getBasePath(path) {

		if (path.endsWith('/'))
			path = path.slice(0, -1);
		if (path.lastIndexOf('.') > path.lastIndexOf('/'))
			path = path.slice(0, path.lastIndexOf('.'));
		return path;
	}

	function getBaseName(path) {

		path = getBasePath(path);
		return path.slice(path.lastIndexOf('/') + 1);
	}

	Engine = function Engine() {

		this.rtenv = null;

		var gameInitPromise = null;
		var unloadAfterInit = true;
		var memorySize = 268435456;

		var progressFunc = null;
		var pckProgressTracker = {};
		var lastProgress = { loaded: 0, total: 0 };

		var canvas = null;
		var stdout = null;
		var stderr = null;

		this.initGame = function(mainPack) {

			if (!gameInitPromise) {

				if (mainPack === undefined) {
					if (basePath !== null) {
						mainPack = basePath + '.pck';
					} else {
						return Promise.reject(new Error("No main pack to load specified"));
					}
				}
				if (basePath === null)
					basePath = getBasePath(mainPack);

				gameInitPromise = Engine.initEngine().then(
					instantiate.bind(this)
				);
				var gameLoadPromise = loadPromise(mainPack, pckProgressTracker).then(function(xhr) { return xhr.response; });
				gameInitPromise = Promise.all([gameLoadPromise, gameInitPromise]).then(function(values) {
					// resolve with pck
					return new Uint8Array(values[0]);
				});
				if (unloadAfterInit)
					gameInitPromise.then(Engine.unloadEngine);
				requestAnimationFrame(animateProgress);
			}
			return gameInitPromise;
		};

		function instantiate(initializer) {

			var rtenvOpts = {
				noInitialRun: true,
				thisProgram: getBaseName(basePath),
				engine: this,
			};
			if (typeof stdout === 'function')
				rtenvOpts.print = stdout;
			if (typeof stderr === 'function')
				rtenvOpts.printErr = stderr;
			if (typeof WebAssembly === 'object' && initializer instanceof WebAssembly.Module) {
				rtenvOpts.instantiateWasm = function(imports, onSuccess) {
					WebAssembly.instantiate(initializer, imports).then(function(result) {
						onSuccess(result);
					});
					return {};
				};
			} else if (initializer.asm && initializer.mem) {
				rtenvOpts.asm = initializer.asm;
				rtenvOpts.memoryInitializerRequest = initializer.mem;
				rtenvOpts.TOTAL_MEMORY = memorySize;
			} else {
				throw new Error("Invalid initializer");
			}

			return new Promise(function(resolve, reject) {
				rtenvOpts.onRuntimeInitialized = resolve;
				rtenvOpts.onAbort = reject;
				rtenvOpts.engine.rtenv = Engine.RuntimeEnvironment(rtenvOpts);
			});
		}

		this.start = function(mainPack) {

			return this.initGame(mainPack).then(synchronousStart.bind(this));
		};

		function synchronousStart(pckView) {
			// TODO don't expect canvas when runninng as cli tool
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
			var context = false;
			try {
				context = actualCanvas.getContext('webgl2') || actualCanvas.getContext('experimental-webgl2');
			} catch (e) {}
			if (!context) {
				throw new Error("WebGL 2 not available");
			}

			// canvas can grab focus on click
			if (actualCanvas.tabIndex < 0) {
				actualCanvas.tabIndex = 0;
			}
			// necessary to calculate cursor coordinates correctly
			actualCanvas.style.padding = 0;
			actualCanvas.style.borderWidth = 0;
			actualCanvas.style.borderStyle = 'none';
			// until context restoration is implemented
			actualCanvas.addEventListener('webglcontextlost', function(ev) {
				alert("WebGL context lost, please reload the page");
				ev.preventDefault();
			}, false);

			this.rtenv.FS.createDataFile('/', this.rtenv.thisProgram + '.pck', pckView, true, true, true);
			gameInitPromise = null;
			this.rtenv.callMain();
		}

		this.setProgressFunc = function(func) {
			progressFunc = func;
		};

		function animateProgress() {

			var loaded = 0;
			var total = 0;
			var totalIsValid = true;
			var progressIsFinal = true;

			[loadingFiles, pckProgressTracker].forEach(function(tracker) {
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

		this.setAsmjsMemorySize = function(size) {
			memorySize = size;
		};

		this.setUnloadAfterInit = function(enabled) {

			if (enabled && !unloadAfterInit && gameInitPromise) {
				gameInitPromise.then(Engine.unloadEngine);
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

	Engine.initEngine = function(newBasePath) {

		if (newBasePath !== undefined) basePath = getBasePath(newBasePath);
		if (engineLoadPromise === null) {
			if (USING_WASM) {
				if (typeof WebAssembly !== 'object')
					return Promise.reject(new Error("Browser doesn't support WebAssembly"));
				// TODO cache/retrieve module to/from idb
				engineLoadPromise = loadPromise(basePath + '.wasm').then(function(xhr) {
					return WebAssembly.compile(xhr.response);
				});
			} else {
				var asmjsPromise = loadPromise(basePath + '.asm.js').then(function(xhr) {
					return asmjsModulePromise(xhr.response);
				});
				var memPromise = loadPromise(basePath + '.mem');
				engineLoadPromise = Promise.all([asmjsPromise, memPromise]).then(function(values) {
					return { asm: values[0], mem: values[1] };
				});
			}
			engineLoadPromise = engineLoadPromise.catch(function(err) {
				engineLoadPromise = null;
				throw err;
			});
		}
		return engineLoadPromise;
	};

	function asmjsModulePromise(module) {
		var elem = document.createElement('script');
		var script = new Blob([
			'Engine.asm = (function() { var Module = {};',
			module,
			'return Module.asm; })();'
		]);
		var url = URL.createObjectURL(script);
		elem.src = url;
		return new Promise(function(resolve, reject) {
			elem.addEventListener('load', function() {
				URL.revokeObjectURL(url);
				var asm = Engine.asm;
				Engine.asm = undefined;
				setTimeout(function() {
					// delay to reclaim compilation memory
					resolve(asm);
				}, 1);
			});
			elem.addEventListener('error', function() {
				URL.revokeObjectURL(url);
				reject("asm.js faiilure");
			});
			document.body.appendChild(elem);
		});
	}

	Engine.unloadEngine = function() {
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
		['loadstart', 'progress', 'load', 'error', 'timeout', 'abort'].forEach(function(ev) {
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
				loadXHR(resolve, reject, file);
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
			case 'timeout':
				if (++tracker[file].attempts >= DOWNLOAD_ATTEMPTS_MAX) {
					tracker[file].final = true;
					reject(new Error("Failed loading file '" + file + "'"));
				} else {
					loadXHR(resolve, reject, file);
				}
				break;

			case 'abort':
				tracker[file].final = true;
				reject(new Error("Loading file '" + file + "' was aborted."));
				break;
		}
	}
})();
