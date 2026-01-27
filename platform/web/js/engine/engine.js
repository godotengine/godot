/**
 * Projects exported for the Web expose the :js:class:`Engine` class to the JavaScript environment, that allows
 * fine control over the engine's start-up process.
 *
 * This API is built in an asynchronous manner and requires basic understanding
 * of `Promises <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises>`__.
 *
 * @module Engine
 * @header Web export JavaScript reference
 */
const Engine = (function () {
	const preloader = new Preloader();

	let loadPromise = null;
	let loadPath = '';
	let initPromise = null;

	/**
	 * @classdesc The ``Engine`` class provides methods for loading and starting exported projects on the Web. For default export
	 * settings, this is already part of the exported HTML page. To understand practical use of the ``Engine`` class,
	 * see :ref:`Custom HTML page for Web export <doc_customizing_html5_shell>`.
	 *
	 * @description Create a new Engine instance with the given configuration.
	 *
	 * @global
	 * @constructor
	 * @param {EngineConfig} initConfig The initial config for this instance.
	 */
	function Engine(initConfig) { // eslint-disable-line no-shadow
		this.config = new InternalConfig(initConfig);
		this.rtenv = null;
	}

	/**
	 * Load the engine from the specified base path.
	 *
	 * @param {string} basePath Base path of the engine to load.
	 * @param {number=} [size=0] The file size if known.
	 * @returns {Promise} A Promise that resolves once the engine is loaded.
	 *
	 * @function Engine.load
	 */
	Engine.load = function (basePath, size) {
		if (loadPromise == null) {
			loadPath = basePath;
			loadPromise = preloader.loadPromise(`${loadPath}.wasm`, size, true);
			requestAnimationFrame(preloader.animateProgress);
		}
		return loadPromise;
	};

	/**
	 * Unload the engine to free memory.
	 *
	 * This method will be called automatically depending on the configuration. See :js:attr:`unloadAfterInit`.
	 *
	 * @function Engine.unload
	 */
	Engine.unload = function () {
		loadPromise = null;
	};

	/**
	 * Safe Engine constructor, creates a new prototype for every new instance to avoid prototype pollution.
	 * @ignore
	 * @constructor
	 */
	function SafeEngine(initConfig) {
		const proto = /** @lends Engine.prototype */ {
			/**
			 * Initialize the engine instance. Optionally, pass the base path to the engine to load it,
			 * if it hasn't been loaded yet. See :js:meth:`Engine.load`.
			 *
			 * @param {string=} basePath Base path of the engine to load.
			 * @return {Promise} A ``Promise`` that resolves once the engine is loaded and initialized.
			 */
			init: function (basePath) {
				if (initPromise) {
					return initPromise;
				}

				preloader.init({
					fileSizes: this.config.fileSizes,
				});

				if (loadPromise == null) {
					if (!basePath) {
						const initPromiseError = new Error('A base path must be provided when calling `init` and the engine is not loaded.');
						initPromise = Promise.reject(initPromiseError);
						return initPromise;
					}

					Engine.load(basePath, this.config.fileSizes[`${basePath}.wasm`]);
				}

				const doInit = async () => {
					const loadResponse = await loadPromise;
					const clonedResponse = new Response(loadResponse.clone().body, { 'headers': [['content-type', 'application/wasm']] });
					const module = await Godot(this.config.getModuleConfig(loadPath, clonedResponse));
					const paths = this.config.persistentPaths;
					const err = await module['initFS'](paths);
					if (err != null) {
						window['console'].error('Error while initializing Godot:', err);
					}
					this.rtenv = module;
					if (this.config.unloadAfterInit) {
						Engine.unload();
					}
				};

				preloader.setProgressFunc(this.config.onProgress);
				initPromise = doInit();
				return initPromise;
			},

			/**
			 * Load a file so it is available in the instance's file system once it runs. Must be called **before** starting the
			 * instance.
			 *
			 * If not provided, the ``path`` is derived from the URL of the loaded file.
			 *
			 * @param {string|ArrayBuffer} file The file to preload.
			 *
			 * If a ``string`` the file will be loaded from that path.
			 *
			 * If an ``ArrayBuffer`` or a view on one, the buffer will used as the content of the file.
			 *
			 * @param {string=} path Path by which the file will be accessible. Required, if ``file`` is not a string.
			 *
			 * @returns {Promise} A Promise that resolves once the file is loaded.
			 */
			preloadFile: function (file, path) {
				return preloader.preload(file, path, this.config.fileSizes[file]);
			},

			/**
			 * Start the engine instance using the given override configuration (if any).
			 * :js:meth:`startGame <Engine.prototype.startGame>` can be used in typical cases instead.
			 *
			 * This will initialize the instance if it is not initialized. For manual initialization, see :js:meth:`init <Engine.prototype.init>`.
			 * The engine must be loaded beforehand.
			 *
			 * Fails if a canvas cannot be found on the page, or not specified in the configuration.
			 *
			 * @async
			 * @param {EngineConfig} override An optional configuration override.
			 * @return {void}
			 */
			start: async function (override) {
				this.config.update(override);
				const me = this;
				await me.init();

				if (!me.rtenv) {
					throw new Error('The engine must be initialized before it can be started');
				}

				let config = {};
				try {
					config = me.config.getGodotConfig(function () {
						me.rtenv = null;
					});
				} catch (err) {
					const newErr = new Error('Error geeting Godot config.');
					newErr.cause = err;
					throw newErr;
				}

				// Godot configuration.
				me.rtenv['initConfig'](config);
				await me.rtenv['initOS']();

				// Preload GDExtension libraries.
				if (me.config.gdextensionLibs.length > 0 && !me.rtenv['loadDynamicLibrary']) {
					throw new Error(
						'GDExtension libraries are not supported by this engine version. '
						+ 'Enable "Extensions Support" for your export preset and/or build your custom template with "dlink_enabled=yes".'
					);
				}

				try {
					for (const file of preloader.preloadedFiles) {
						me.rtenv['copyToFS'](file.path, file.buffer);
					}
					preloader.preloadedFiles.length = 0; // Clear memory
					me.rtenv['callMain'](me.config.args);
					initPromise = null;
					me.installServiceWorker();
				} catch (err) {
					const newErr = new Error('Error while initializing.');
					newErr.cause = err;
					throw newErr;
				}
			},

			/**
			 * Start the game instance using the given configuration override (if any).
			 *
			 * This will initialize the instance if it is not initialized. For manual initialization, see :js:meth:`init <Engine.prototype.init>`.
			 *
			 * This will load the engine if it is not loaded, and preload the main pck.
			 *
			 * This method expects the initial config (or the override) to have both the :js:attr:`executable` and :js:attr:`mainPack`
			 * properties set (normally done by the editor during export).
			 *
			 * @param {EngineConfig} override An optional configuration override.
			 * @return {Promise} Promise that resolves once the game started.
			 */
			startGame: async function (override) {
				this.config.update(override);
				this.insertImportMap();

				// Add main-pack argument.
				const exe = this.config.executable;
				let pack = this.config.mainPack || `${exe}.pck`;
				if (pack.endsWith('/')) {
					pack = pack.substring(0, pack.length - 1);
				}

				this.config.args = ['--main-pack', pack].concat(this.config.args);
				// Start and init with execName as loadPath if not inited.
				const me = this;
				const filesToPreload = [];

				if (pack.endsWith('.asyncpck')) {
					if (this.config.asyncPckData == null) {
						throw new Error('No Main Scene dependencies found.');
					}

					const asyncPckData = this.config['asyncPckData'];
					const asyncPckAssetsDir = asyncPckData['directories']['assets'];

					const asyncPckInitialLoadFilesSet = new Set();
					const asyncPckDataInitialLoad = asyncPckData['initialLoad'];
					for (const value of Object.values(asyncPckDataInitialLoad)) {
						for (const resourcePath of Object.values(value['files'])) {
							asyncPckInitialLoadFilesSet.add(resourcePath);
						}
					}

					const asyncPckInitialLoadFiles = Array.from(asyncPckInitialLoadFilesSet);

					const resToLocal = (pPath) => {
						const PREFIX_RES = 'res://';
						let path = pPath;
						if (path.startsWith(PREFIX_RES)) {
							path = path.substring('res://'.length);
						}
						return `${asyncPckAssetsDir}/${path}`;
					};

					for (const resourcePath of asyncPckInitialLoadFiles) {
						const pathToPreload = resToLocal(resourcePath);
						filesToPreload.push(this.preloadFile(pathToPreload, pathToPreload));
					}
				} else {
					filesToPreload.push(this.preloadFile(pack, pack));
				}

				await Promise.all([this.init(exe), ...filesToPreload]);

				return me.start.apply(me);
			},

			/**
			 * Create a file at the specified ``path`` with the passed as ``buffer`` in the instance's file system.
			 *
			 * @param {string} path The location where the file will be created.
			 * @param {ArrayBuffer} buffer The content of the file.
			 */
			copyToFS: function (path, buffer) {
				if (this.rtenv == null) {
					throw new Error('Engine must be inited before copying files');
				}
				this.rtenv['copyToFS'](path, buffer);
			},

			/**
			 * Request that the current instance quit.
			 *
			 * This is akin the user pressing the close button in the window manager, and will
			 * have no effect if the engine has crashed, or is stuck in a loop.
			 *
			 */
			requestQuit: function () {
				if (this.rtenv) {
					this.rtenv['request_quit']();
				}
			},

			/**
			 * Install the progressive-web app service worker.
			 * @returns {Promise} The service worker registration promise.
			 */
			installServiceWorker: function () {
				if (this.config.serviceWorker && 'serviceWorker' in navigator) {
					try {
						return navigator.serviceWorker.register(this.config.serviceWorker);
					} catch (e) {
						return Promise.reject(e);
					}
				}
				return Promise.resolve();
			},

			/**
			 * Install the JavaScript module import map.
			 */
			insertImportMap() {
				const IMPORTMAP_ID = 'godotengine-importmap-engine';
				if (document.getElementById(IMPORTMAP_ID)) {
					return;
				}
				const scriptElement = document.createElement('script');
				scriptElement.id = IMPORTMAP_ID;
				scriptElement.type = 'importmap';
				const scriptElementContent = {
					imports: {
						'@godotengine/utils/concurrencyQueueManager': `./${this.config.executable}.utils.concurrency.js`,
						'@godotengine/utils/wait': `./${this.config.executable}.utils.wait.js`,
					},
				};
				scriptElement.textContent = JSON.stringify(scriptElementContent, null, 2);
				document.head.insertAdjacentElement('beforeend', scriptElement);
			},
		};

		Engine.prototype = proto;
		// Closure compiler exported instance methods.
		Engine.prototype['init'] = Engine.prototype.init;
		Engine.prototype['preloadFile'] = Engine.prototype.preloadFile;
		Engine.prototype['start'] = Engine.prototype.start;
		Engine.prototype['startGame'] = Engine.prototype.startGame;
		Engine.prototype['copyToFS'] = Engine.prototype.copyToFS;
		Engine.prototype['requestQuit'] = Engine.prototype.requestQuit;
		Engine.prototype['installServiceWorker'] = Engine.prototype.installServiceWorker;
		// Also expose static methods as instance methods
		Engine.prototype['load'] = Engine.load;
		Engine.prototype['unload'] = Engine.unload;
		return new Engine(initConfig);
	}

	// Closure compiler exported static methods.
	SafeEngine['load'] = Engine.load;
	SafeEngine['unload'] = Engine.unload;

	// Feature-detection utilities.
	SafeEngine['isWebGLAvailable'] = Features.isWebGLAvailable;
	SafeEngine['isFetchAvailable'] = Features.isFetchAvailable;
	SafeEngine['isSecureContext'] = Features.isSecureContext;
	SafeEngine['isCrossOriginIsolated'] = Features.isCrossOriginIsolated;
	SafeEngine['isSharedArrayBufferAvailable'] = Features.isSharedArrayBufferAvailable;
	SafeEngine['isAudioWorkletAvailable'] = Features.isAudioWorkletAvailable;
	SafeEngine['getMissingFeatures'] = Features.getMissingFeatures;

	return SafeEngine;
}());
if (typeof window !== 'undefined') {
	window['Engine'] = Engine;
}
