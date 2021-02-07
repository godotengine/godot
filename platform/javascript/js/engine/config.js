/** @constructor */
function EngineConfig(opts) {
	// Module config
	this.unloadAfterInit = true;
	this.onPrintError = function () {
		console.error.apply(console, Array.from(arguments)); // eslint-disable-line no-console
	};
	this.onPrint = function () {
		console.log.apply(console, Array.from(arguments)); // eslint-disable-line no-console
	};
	this.onProgress = null;

	// Godot Config
	this.canvas = null;
	this.executable = '';
	this.mainPack = null;
	this.locale = null;
	this.canvasResizePolicy = false;
	this.persistentPaths = ['/userfs'];
	this.gdnativeLibs = [];
	this.args = [];
	this.onExecute = null;
	this.onExit = null;
	this.update(opts);
}

EngineConfig.prototype.update = function (opts) {
	const config = opts || {};
	function parse(key, def) {
		if (typeof (config[key]) === 'undefined') {
			return def;
		}
		return config[key];
	}
	// Module config
	this.unloadAfterInit = parse('unloadAfterInit', this.unloadAfterInit);
	this.onPrintError = parse('onPrintError', this.onPrintError);
	this.onPrint = parse('onPrint', this.onPrint);
	this.onProgress = parse('onProgress', this.onProgress);

	// Godot config
	this.canvas = parse('canvas', this.canvas);
	this.executable = parse('executable', this.executable);
	this.mainPack = parse('mainPack', this.mainPack);
	this.locale = parse('locale', this.locale);
	this.canvasResizePolicy = parse('canvasResizePolicy', this.canvasResizePolicy);
	this.persistentPaths = parse('persistentPaths', this.persistentPaths);
	this.gdnativeLibs = parse('gdnativeLibs', this.gdnativeLibs);
	this.args = parse('args', this.args);
	this.onExecute = parse('onExecute', this.onExecute);
	this.onExit = parse('onExit', this.onExit);
};

EngineConfig.prototype.getModuleConfig = function (loadPath, loadPromise) {
	const me = this;
	return {
		'print': this.onPrint,
		'printErr': this.onPrintError,
		'locateFile': Utils.createLocateRewrite(loadPath),
		'instantiateWasm': Utils.createInstantiatePromise(loadPromise),
		'thisProgram': me.executable,
		'noExitRuntime': true,
		'dynamicLibraries': [`${me.executable}.side.wasm`],
	};
};

EngineConfig.prototype.getGodotConfig = function (cleanup) {
	if (!(this.canvas instanceof HTMLCanvasElement)) {
		this.canvas = Utils.findCanvas();
		if (!this.canvas) {
			throw new Error('No canvas found in page');
		}
	}

	// Canvas can grab focus on click, or key events won't work.
	if (this.canvas.tabIndex < 0) {
		this.canvas.tabIndex = 0;
	}

	// Browser locale, or custom one if defined.
	let locale = this.locale;
	if (!locale) {
		locale = navigator.languages ? navigator.languages[0] : navigator.language;
		locale = locale.split('.')[0];
	}
	const onExit = this.onExit;
	// Godot configuration.
	return {
		'canvas': this.canvas,
		'canvasResizePolicy': this.canvasResizePolicy,
		'locale': locale,
		'onExecute': this.onExecute,
		'onExit': function (p_code) {
			cleanup(); // We always need to call the cleanup callback to free memory.
			if (typeof (onExit) === 'function') {
				onExit(p_code);
			}
		},
	};
};
