const Engine = (function () {
	const preloader = new Preloader();

	let loadPromise = null;
	let loadPath = '';
	let initPromise = null;

	function load(basePath) {
		if (loadPromise == null) {
			loadPath = basePath;
			loadPromise = preloader.loadPromise(`${loadPath}.wasm`);
			requestAnimationFrame(preloader.animateProgress);
		}
		return loadPromise;
	}

	function unload() {
		loadPromise = null;
	}

	/** @constructor */
	function Engine(opts) { // eslint-disable-line no-shadow
		this.config = new EngineConfig(opts);
		this.rtenv = null;
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
		preloader.setProgressFunc(this.config.onProgress);
		let config = this.config.getModuleConfig(loadPath, loadPromise);
		const me = this;
		initPromise = new Promise(function (resolve, reject) {
			Godot(config).then(function (module) {
				module['initFS'](me.config.persistentPaths).then(function (fs_err) {
					me.rtenv = module;
					if (me.config.unloadAfterInit) {
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
	Engine.prototype.start = function (override) {
		this.config.update(override);
		const me = this;
		return me.init().then(function () {
			if (!me.rtenv) {
				return Promise.reject(new Error('The engine must be initialized before it can be started'));
			}

			let config = {};
			try {
				config = me.config.getGodotConfig(function () {
					me.rtenv = null;
				});
			} catch (e) {
				return Promise.reject(e);
			}
			// Godot configuration.
			me.rtenv['initConfig'](config);

			// Preload GDNative libraries.
			const libs = [];
			me.config.gdnativeLibs.forEach(function (lib) {
				libs.push(me.rtenv['loadDynamicLibrary'](lib, { 'loadAsync': true }));
			});
			return Promise.all(libs).then(function () {
				return new Promise(function (resolve, reject) {
					preloader.preloadedFiles.forEach(function (file) {
						me.rtenv['copyToFS'](file.path, file.buffer);
					});
					preloader.preloadedFiles.length = 0; // Clear memory
					me.rtenv['callMain'](me.config.args);
					initPromise = null;
					resolve();
				});
			});
		});
	};

	Engine.prototype.startGame = function (override) {
		this.config.update(override);
		// Add main-pack argument.
		const exe = this.config.executable;
		const pack = this.config.mainPack || `${exe}.pck`;
		this.config.args = ['--main-pack', pack].concat(this.config.args);
		// Start and init with execName as loadPath if not inited.
		const me = this;
		return Promise.all([
			this.init(exe),
			this.preloadFile(pack, pack),
		]).then(function () {
			return me.start.apply(me);
		});
	};

	Engine.prototype.copyToFS = function (path, buffer) {
		if (this.rtenv == null) {
			throw new Error('Engine must be inited before copying files');
		}
		this.rtenv['copyToFS'](path, buffer);
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
	Engine.prototype['copyToFS'] = Engine.prototype.copyToFS;
	Engine.prototype['requestQuit'] = Engine.prototype.requestQuit;
	return Engine;
}());
if (typeof window !== 'undefined') {
	window['Engine'] = Engine;
}
