/**************************************************************************/
/*  library_godot_os.js                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

const IDHandler = {
	$IDHandler: {
		_last_id: 0,
		_references: {},

		get: function (p_id) {
			return IDHandler._references[p_id];
		},

		add: function (p_data) {
			const id = ++IDHandler._last_id;
			IDHandler._references[id] = p_data;
			return id;
		},

		remove: function (p_id) {
			delete IDHandler._references[p_id];
		},
	},
};

autoAddDeps(IDHandler, '$IDHandler');
mergeInto(LibraryManager.library, IDHandler);

const GodotConfig = {
	$GodotConfig__postset: 'Module["initConfig"] = GodotConfig.init_config;',
	$GodotConfig__deps: ['$GodotRuntime'],
	$GodotConfig: {
		canvas: null,
		locale: 'en',
		canvas_resize_policy: 2, // Adaptive
		virtual_keyboard: false,
		persistent_drops: false,
		godot_pool_size: 4,
		on_execute: null,
		on_exit: null,
		mainPack: '',
		asyncPckData: null,
		fileSizes: null,

		init_config: function (p_opts) {
			GodotConfig.canvas_resize_policy = p_opts['canvasResizePolicy'];
			GodotConfig.canvas = p_opts['canvas'];
			GodotConfig.locale = p_opts['locale'] || GodotConfig.locale;
			GodotConfig.virtual_keyboard = p_opts['virtualKeyboard'];
			GodotConfig.persistent_drops = !!p_opts['persistentDrops'];
			GodotConfig.godot_pool_size = p_opts['godotPoolSize'];
			GodotConfig.on_execute = p_opts['onExecute'];
			GodotConfig.on_exit = p_opts['onExit'];
			GodotConfig.mainPack = p_opts['mainPack'];
			GodotConfig.asyncPckData = p_opts['asyncPckData'] ?? null;
			GodotConfig.fileSizes = p_opts['fileSizes'] ?? null;
			if (p_opts['focusCanvas']) {
				GodotConfig.canvas.focus();
			}
		},

		locate_file: function (file) {
			return Module['locateFile'](file);
		},
		clear: function () {
			GodotConfig.canvas = null;
			GodotConfig.locale = 'en';
			GodotConfig.canvas_resize_policy = 2;
			GodotConfig.virtual_keyboard = false;
			GodotConfig.persistent_drops = false;
			GodotConfig.on_execute = null;
			GodotConfig.on_exit = null;
			GodotConfig.mainPack = '';
			GodotConfig.asyncPckData = null;
			GodotConfig.fileSizes = null;
		},
	},

	godot_js_config_canvas_id_get__proxy: 'sync',
	godot_js_config_canvas_id_get__sig: 'vii',
	godot_js_config_canvas_id_get: function (p_ptr, p_ptr_max) {
		GodotRuntime.stringToHeap(`#${GodotConfig.canvas.id}`, p_ptr, p_ptr_max);
	},

	godot_js_config_locale_get__proxy: 'sync',
	godot_js_config_locale_get__sig: 'vii',
	godot_js_config_locale_get: function (p_ptr, p_ptr_max) {
		GodotRuntime.stringToHeap(GodotConfig.locale, p_ptr, p_ptr_max);
	},
};

autoAddDeps(GodotConfig, '$GodotConfig');
mergeInto(LibraryManager.library, GodotConfig);

const GodotFS = {
	$GodotFS__deps: ['$FS', '$IDBFS', '$GodotRuntime'],
	$GodotFS__postset: [
		'Module["initFS"] = GodotFS.init;',
		'Module["copyToFS"] = GodotFS.copy_to_fs;',
	].join(''),
	$GodotFS: {
		// ERRNO_CODES works every odd version of emscripten, but this will break too eventually.
		ENOENT: 44,
		_idbfs: false,
		_syncing: false,
		_mount_points: [],

		is_persistent: function () {
			return GodotFS._idbfs ? 1 : 0;
		},

		// Initialize godot file system, setting up persistent paths.
		// Returns a promise that resolves when the FS is ready.
		// We keep track of mount_points, so that we can properly close the IDBFS
		// since emscripten is not doing it by itself. (emscripten GH#12516).
		init: function (persistentPaths) {
			GodotFS._idbfs = false;
			if (!Array.isArray(persistentPaths)) {
				return Promise.reject(new Error('Persistent paths must be an array'));
			}
			if (!persistentPaths.length) {
				return Promise.resolve();
			}
			GodotFS._mount_points = persistentPaths.slice();

			function createRecursive(dir) {
				try {
					FS.stat(dir);
				} catch (e) {
					if (e.errno !== GodotFS.ENOENT) {
						// Let mkdirTree throw in case, we cannot trust the above check.
						GodotRuntime.error(e);
					}
					FS.mkdirTree(dir);
				}
			}

			GodotFS._mount_points.forEach(function (path) {
				createRecursive(path);
				FS.mount(IDBFS, {}, path);
			});
			return new Promise(function (resolve, reject) {
				FS.syncfs(true, function (err) {
					if (err) {
						GodotFS._mount_points = [];
						GodotFS._idbfs = false;
						GodotRuntime.print(`IndexedDB not available: ${err.message}`);
					} else {
						GodotFS._idbfs = true;
					}
					resolve(err);
				});
			});
		},

		// Deinit godot file system, making sure to unmount file systems, and close IDBFS(s).
		deinit: function () {
			GodotFS._mount_points.forEach(function (path) {
				try {
					FS.unmount(path);
				} catch (e) {
					GodotRuntime.print('Already unmounted', e);
				}
				if (GodotFS._idbfs && IDBFS.dbs[path]) {
					IDBFS.dbs[path].close();
					delete IDBFS.dbs[path];
				}
			});
			GodotFS._mount_points = [];
			GodotFS._idbfs = false;
			GodotFS._syncing = false;
		},

		sync: function () {
			if (GodotFS._syncing) {
				GodotRuntime.error('Already syncing!');
				return Promise.resolve();
			}
			GodotFS._syncing = true;
			return new Promise(function (resolve, reject) {
				FS.syncfs(false, function (error) {
					if (error) {
						GodotRuntime.error(`Failed to save IDB file system: ${error.message}`);
					}
					GodotFS._syncing = false;
					resolve(error);
				});
			});
		},

		// Copies a buffer to the internal file system. Creating directories recursively.
		copy_to_fs: function (path, buffer) {
			const idx = path.lastIndexOf('/');
			let dir = '/';
			if (idx > 0) {
				dir = path.slice(0, idx);
			}
			try {
				FS.stat(dir);
			} catch (e) {
				if (e.errno !== GodotFS.ENOENT) {
					// Let mkdirTree throw in case, we cannot trust the above check.
					GodotRuntime.error(e);
				}
				FS.mkdirTree(dir);
			}
			FS.writeFile(path, new Uint8Array(buffer));
		},
	},
};
mergeInto(LibraryManager.library, GodotFS);

class AsyncPCKFile {
	static get Status() {
		return Object.freeze({
			STATUS_ERROR: 'STATUS_ERROR',
			STATUS_IDLE: 'STATUS_IDLE',
			STATUS_LOADING: 'STATUS_LOADING',
			STATUS_INSTALLED: 'STATUS_INSTALLED',
		});
	}

	constructor(pAsyncPCKPath, pPath, pSize) {
		this.asyncPCKPath = pAsyncPCKPath;
		this.path = pPath;
		{
			const assetsDir = GodotOS.asyncPCKGetAsyncPCKAssetsDir(this.asyncPCKPath);
			this.localPath = `${assetsDir}/${GodotOS._removeResPrefix(this.path)}`;
		}

		this._status = GodotOS.AsyncPCKFile.Status.STATUS_IDLE;
		this._installed = false;
		this._size = pSize;
		this._progress = 0;
		this._progressRatio = 0;
		this._loadPromise = null;
		this._error = null;
	}

	get status() {
		return this._status;
	}

	get size() {
		return this._size;
	}

	get progress() {
		return this._progress;
	}

	get progressRatio() {
		return this._progressRatio;
	}

	get error() {
		return this._error;
	}

	_addToProgress(pAddedBytes) {
		this._setProgress(this._progress + pAddedBytes);
	}

	_setProgress(pProgress) {
		let progress = pProgress;
		if (progress <= 0) {
			progress = 0;
		}
		this._progress = progress;
		if (this._progress > this._size) {
			this._size = progress;
		}
		if (this._size > 0) {
			this._progressRatio = this._size / progress;
		}
	}

	async load() {
		if (this._loadPromise != null) {
			return this._loadPromise;
		}

		if (this._installed) {
			GodotRuntime.print(
				`AsyncPCKFile "${this.path}" (of AsyncPck "${this.asyncPCKPath}") is already installed, skipping loading.`
			);
			return Promise.resolve();
		}
		if (this._status == GodotOS.AsyncPCKFile.Status.STATUS_LOADING) {
			GodotRuntime.print(
				`AsyncPCKFile "${this.path}" (of AsyncPck "${this.asyncPCKPath}") is currently loading, skipping loading.`
			);
			return Promise.resolve();
		}
		this._status = GodotOS.AsyncPCKFile.Status.STATUS_LOADING;

		this._loadPromise = this._load();
		return await this._loadPromise;
	}

	async _load() {
		try {
			const fileBuffer = await this._loadAttempt();

			GodotFS.copy_to_fs(this.localPath, fileBuffer);
			this._status = GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED;
		} catch (err) {
			this._status = GodotOS.AsyncPCKFile.Status.STATUS_ERROR;

			const newError = new Error(
				`AsyncPCKFile "${this.path}" (of AsyncPck "${this.asyncPCKPath}"): error while loading "${this.localPath}"`
			);
			newError.cause = err;
			this._error = newError;
			throw newError;
		} finally {
			this._loadPromise = null;
		}
	}

	async _loadAttempt(pRetryCount = 0) {
		try {
			const fileResponse = await GodotOS.asyncPCKFetch(this.localPath);
			if (!fileResponse.ok) {
				this._status = GodotOS.AsyncPCKFile.Status.STATUS_ERROR;
				throw new Error(`Couldn't load file "${this.localPath}".`);
			}

			const chunks = [];
			const reader = fileResponse.body.getReader();

			while (true) {
				const { done, value: chunk } = await reader.read(); // eslint-disable-line no-await-in-loop
				if (done) {
					break;
				}
				this._addToProgress(chunk.byteLength);
				chunks.push(chunk);
			}

			const fileBuffer = new Uint8Array(this._progress);
			let filePosition = 0;
			for (const chunk of chunks) {
				fileBuffer.set(chunk, filePosition);
				filePosition += chunk.byteLength;
			}

			return fileBuffer;
		} catch (err) {
			const newError = new Error(
				`AsyncPCKFile "${this.path}" (of AsyncPck "${this.asyncPCKPath}"): error while attempting to load "${this.localPath}". (attempt ${pRetryCount + 1}/${GodotOS._asyncPCKFetchMaxRetry})`
			);
			newError.cause = err;

			if (pRetryCount == GodotOS._asyncPCKFetchMaxRetry) {
				const maxRetryError = new Error(`Maximum retry count (${GodotOS._asyncPCKFetchMaxRetry}) reached.`);
				maxRetryError.cause = newError;
				throw maxRetryError;
			}

			GodotRuntime.error(newError);
			this._error = newError;

			// Exponent wait.
			await GodotOS._wait(GodotOS._asyncPCKWaitTimeBaseMs * 2 ** pRetryCount, 'ms');
			return this._loadAttempt(pRetryCount + 1);
		}
	}

	flagAsInstalled() {
		this._status = GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED;
		this._progress = this._size;
		this._progressRatio = 1.0;
	}

	getAsJsonObject() {
		let error = '';
		if (this._error != null) {
			error = this._error.message;
		}

		return {
			local_path: this.localPath,
			status: this._status,
			size: this._size,
			progress: this._progress,
			progress_ratio: this._progressRatio,
			error,
		};
	}
}

class AsyncPCKResource {
	static createAndInitialize(pAsyncPCK, pPath, pFiles, pDependencies = []) {
		const asyncPCKResource = new GodotOS.AsyncPCKResource(pAsyncPCK, pPath);
		asyncPCKResource.initialize(pFiles, pDependencies);
		asyncPCKResource.insertInInstallMap();
		return asyncPCKResource;
	}

	static create(pAsyncPCK, pPath) {
		const asyncPCKResource = new GodotOS.AsyncPCKResource(pAsyncPCK, pPath);
		asyncPCKResource.insertInInstallMap();
		return asyncPCKResource;
	}

	constructor(pAsyncPCKPath, pPath) {
		this.asyncPCKPath = pAsyncPCKPath;
		this.path = pPath;
		this.files = [];
		this.dependencies = [];

		this._initialized = false;
		this._loadPromise = null;
	}

	initialize(pFiles, pDependencies = []) {
		if (this._initialized) {
			throw new Error(`Cannot initialize AsyncPCKResource more than once. ("${this.path}" of "${this.asyncPCKPath}")`);
		}
		this._initialized = true;

		this.dependencies = pDependencies;

		for (const [filePath, fileDefinition] of Object.entries(pFiles)) {
			const asyncPCKFile = new GodotOS.AsyncPCKFile(this.asyncPCKPath, filePath, fileDefinition['size']);
			this.files.push(asyncPCKFile);
		}
	}

	get initialized() {
		return this._initialized;
	}

	get size() {
		return this.files.reduce((pAccumulatorValue, pFile) => pAccumulatorValue + pFile.size, 0);
	}

	get progress() {
		return this.files.reduce((pAccumulatorValue, pFile) => pAccumulatorValue + pFile.progress, 0);
	}

	get progressRatio() {
		const currentSize = this.size;
		if (currentSize <= 0) {
			return 0;
		}
		return this.progress / currentSize;
	}

	get status() {
		if (this.files.find(GodotOS.AsyncPCKResource.isStatusError) != null) {
			return GodotOS.AsyncPCKFile.Status.STATUS_ERROR;
		}
		if (this.files.length > 0 && this.files.every(GodotOS.AsyncPCKResource.isStatusInstalled)) {
			return GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED;
		}
		if (this.files.find(GodotOS.AsyncPCKResource.isStatusLoading) != null) {
			return GodotOS.AsyncPCKFile.Status.STATUS_LOADING;
		}
		return GodotOS.AsyncPCKFile.Status.STATUS_IDLE;
	}

	get errors() {
		return this.files
			.filter(GodotOS.AsyncPCKResource.isStatusError)
			.map((pFile) => pFile.error)
			.filter((pFileError) => pFileError !== '');
	}

	get allDependencies() {
		const dependenciesMap = new Map();
		for (const dependency of this.dependencies) {
			const asyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(this.asyncPCKPath, dependency);
			if (asyncPCKResource == null) {
				throw new Error(
					`Cannot get dependencies of a non-resource ("${dependency}" of "${this.asyncPCKPath}")`
				);
			}
			asyncPCKResource._getAllDependencies(dependenciesMap);
		}
		return Object.fromEntries(dependenciesMap);
	}

	_getAllDependencies(pDependenciesMap) {
		if (pDependenciesMap.has(this.path)) {
			return;
		}
		pDependenciesMap.set(this.path, this.getAsJsonObject({ withDependencies: false }));
		for (const dependency of this.dependencies) {
			const asyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(this.asyncPCKPath, dependency);
			if (asyncPCKResource == null) {
				throw new Error(
					`Cannot get dependencies of a non-resource ("${dependency}" of "${this.asyncPCKPath}")`
				);
			}
			asyncPCKResource._getAllDependencies(pDependenciesMap);
		}
	}

	get allDependenciesResources() {
		const dependenciesResources = [];
		const allDependencies = this.allDependencies;
		for (const dependency of Object.keys(allDependencies)) {
			const asyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(this.asyncPCKPath, dependency);
			if (asyncPCKResource == null) {
				throw new Error(
					`Cannot get dependencies of a non-resource ("${dependency}" of "${this.asyncPCKPath}")`
				);
			}
			dependenciesResources.push(asyncPCKResource);
		}
		return dependenciesResources;
	}

	async load() {
		if (this._loadPromise != null) {
			return this._loadPromise;
		}
		if (this.status == GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED) {
			return Promise.resolve();
		}
		this._loadPromise = this._load();
		return await this._loadPromise;
	}

	async _load() {
		try {
			await Promise.allSettled(
				this.files.map((pFile) => {
					if (pFile.status == GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED) {
						return Promise.resolve();
					}
					return pFile.load(pFile);
				})
			);
		} catch (err) {
			const newError = new Error(
				`AsyncPCKResource "${this.path}" (of AsyncPCK "${this.asyncPCKPath}"): error while loading"`
			);
			newError.cause = err;
			throw newError;
		} finally {
			this._loadPromise = null;
		}
	}

	flagAsInstalled() {
		for (const file of this.files) {
			file.flagAsInstalled();
		}
	}

	insertInInstallMap() {
		if (!GodotOS._asyncPCKInstallMap.has(this.asyncPCKPath)) {
			GodotOS._asyncPCKInstallMap.set(this.asyncPCKPath, new Map());
		}
		if (GodotOS._asyncPCKInstallMap.get(this.asyncPCKPath).has(this.path)) {
			const asyncPCKResource = GodotOS._asyncPCKInstallMap.get(this.asyncPCKPath).get(this.path);
			if (!asyncPCKResource.isTemporary) {
				throw new Error(
					`AsyncPCKResource "${this.path}" (of AsyncPCK "${this.asyncPCKPath}"): cannot install over a non-temporary AsyncPCKResource"`
				);
			}
		}
		GodotOS._asyncPCKInstallMap.get(this.asyncPCKPath).set(this.path, this);
	}

	removeFromInstallMap() {
		if (!GodotOS._asyncPCKInstallMap.has(this.asyncPCKPath)) {
			return;
		}
		GodotOS._asyncPCKInstallMap.get(this.asyncPCKPath).delete(this.path);
	}

	getAsJsonObject(pOptions = {}) {
		const { withDependencies = true } = pOptions;
		const jsonData = {};
		if (withDependencies) {
			const dependenciesResources = this.allDependenciesResources;
			jsonData['files'] = {};
			jsonData['files'][this.path] = this.getAsJsonObject({ withDependencies: false });
			for (const dependencyResource of dependenciesResources) {
				jsonData['files'][dependencyResource.path] = dependencyResource.getAsJsonObject({
					withDependencies: false,
				});
			}

			const jsonDataSize
				= this.size
					+ dependenciesResources.reduce(
						(pAccumulator, pDependencyResource) => pAccumulator + pDependencyResource.size,
						0
					);
			const jsonDataProgress
				= this.progress
					+ dependenciesResources.reduce(
						(pAccumulator, pDependencyResource) => pAccumulator + pDependencyResource.progress,
						0
					);
			let jsonDataProgressRatio = 0;
			if (jsonDataSize > 0) {
				jsonDataProgressRatio = jsonDataProgress / jsonDataSize;
			}

			const jsonDataErrors = Object.assign(
				{},
				(() => {
					const thisInstanceErrors = this.errors;
					if (thisInstanceErrors.length === 0) {
						return {};
					}
					return { [this.path]: thisInstanceErrors };
				})(),
				...dependenciesResources.map((pDependencyResource) => {
					const errors = pDependencyResource.errors;
					if (errors.length === 0) {
						return {};
					}
					return {
						[pDependencyResource.path]: pDependencyResource.errors,
					};
				})
			);

			jsonData['size'] = jsonDataSize;
			jsonData['progress'] = jsonDataProgress;
			jsonData['progress_ratio'] = jsonDataProgressRatio;
			jsonData['errors'] = jsonDataErrors;

			let status = this.status;
			if (
				status == GodotOS.AsyncPCKFile.Status.STATUS_IDLE
				|| status == GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED
			) {
				if (dependenciesResources.find(GodotOS.AsyncPCKResource.isStatusError) != null) {
					status = GodotOS.AsyncPCKFile.Status.STATUS_ERROR;
				}
				if (dependenciesResources.length > 0 && dependenciesResources.every(GodotOS.AsyncPCKResource.isStatusInstalled)) {
					status = GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED;
				}
				if (dependenciesResources.find(GodotOS.AsyncPCKResource.isStatusLoading) != null) {
					status = GodotOS.AsyncPCKFile.Status.STATUS_LOADING;
				}
			}
			jsonData['status'] = status;
		} else {
			jsonData['size'] = this.size;
			jsonData['progress'] = this.progress;
			jsonData['progress_ratio'] = this.progressRatio;
			jsonData['status'] = this.status;
			jsonData['errors'] = this.errors;
		}

		return jsonData;
	}

	static isStatus(pStatus, pFile) {
		return pFile.status == pStatus;
	}

	static isStatusError(pFile) {
		return GodotOS.AsyncPCKResource.isStatus(GodotOS.AsyncPCKFile.Status.STATUS_ERROR, pFile);
	}

	static isStatusLoading(pFile) {
		return GodotOS.AsyncPCKResource.isStatus(GodotOS.AsyncPCKFile.Status.STATUS_LOADING, pFile);
	}

	static isStatusInstalled(pFile) {
		return GodotOS.AsyncPCKResource.isStatus(GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED, pFile);
	}
}

const _GodotOS = {
	$GodotOS__deps: ['$GodotRuntime', '$GodotConfig', '$GodotFS'],
	$GodotOS__postset: [
		'Module["initOS"] = async () => { await GodotOS.init(); };',
		'Module["request_quit"] = function() { GodotOS.request_quit() };',
		'Module["onExit"] = GodotOS.cleanup;',
		'GodotOS._fs_sync_promise = Promise.resolve();',
		'GodotOS._asyncPCKInstallMap = new Map();',
	].join(' '),
	$GodotOS: {
		request_quit: function () {},
		_async_cbs: [],
		_fs_sync_promise: null,
		AsyncPCKFile: AsyncPCKFile,
		AsyncPCKResource: AsyncPCKResource,
		_asyncPCKInstallMap: null,
		_asyncPCKConcurrencyQueueManager: null,
		_asyncPCKFetchMaxRetry: 5,
		_asyncPCKWaitTimeBaseMs: 100,
		_mainPack: '',
		_prefixRes: 'res://',

		_trimLastSlash: function (pPath) {
			if (pPath.endsWith('/')) {
				return pPath.substring(0, pPath.length - 1);
			}
			return pPath;
		},
		_addResPrefix: function (pPath) {
			let path = pPath;
			if (!path.startsWith(GodotOS._prefixRes)) {
				path = GodotOS._prefixRes + path;
			}
			return path;
		},
		_removeResPrefix: function (pPath) {
			let path = pPath;
			if (path.startsWith(GodotOS._prefixRes)) {
				path = path.substring(GodotOS._prefixRes.length);
			}
			return path;
		},

		init: async function () {
			const { wait } = await import('@godotengine/utils/wait');
			GodotOS._wait = wait;

			GodotOS._mainPack = GodotConfig.mainPack ?? '';
			if (GodotOS._mainPack.endsWith('.asyncpck')) {
				const { ConcurrencyQueueManager } = await import('@godotengine/utils/concurrencyQueueManager');
				// eslint-disable-next-line require-atomic-updates -- We set `GodotOS._concurrencyQueueManager` only once: at init time.
				GodotOS._asyncPCKConcurrencyQueueManager = new ConcurrencyQueueManager();
				GodotOS.initAsyncPck();
			}
		},

		initAsyncPck: function () {
			const data = GodotConfig.asyncPckData;
			const fileSizes = GodotConfig.fileSizes;

			const initialLoad = data['initialLoad'];
			for (const [resourceKey, resourceValue] of Object.entries(initialLoad)) {
				const dependencies = {};
				for (const resourceDependency of resourceValue['files']) {
					let assetsPath = data.directories.assets;
					if (!assetsPath.endsWith('/')) {
						assetsPath += '/';
					}
					const fileSizePath = assetsPath + resourceDependency.substring('res://'.length);
					dependencies[resourceDependency] = {
						size: fileSizes[fileSizePath],
					};
				}
				let asyncPckResource = GodotOS.asyncPCKGetAsyncPCKResource(GodotOS._mainPack, resourceKey);
				if (asyncPckResource != null) {
					continue;
				}

				asyncPckResource = GodotOS.AsyncPCKResource.createAndInitialize(
					GodotOS._mainPack,
					resourceKey,
					dependencies,
					resourceValue?.dependencies ?? []
				);
				asyncPckResource.flagAsInstalled();
			}
		},

		atexit: function (p_promise_cb) {
			GodotOS._async_cbs.push(p_promise_cb);
		},

		cleanup: function (exit_code) {
			const cb = GodotConfig.on_exit;
			GodotFS.deinit();
			GodotConfig.clear();
			if (cb) {
				cb(exit_code);
			}
		},

		finish_async: function (callback) {
			GodotOS._fs_sync_promise
				.then(function (err) {
					const promises = [];
					GodotOS._async_cbs.forEach(function (cb) {
						promises.push(new Promise(cb));
					});
					return Promise.all(promises);
				})
				.then(function () {
					return GodotFS.sync(); // Final FS sync.
				})
				.then(function (err) {
					// Always deferred.
					setTimeout(function () {
						callback();
					}, 0);
				});
		},

		asyncPCKFetch: async function (...pArgs) {
			if (GodotOS._asyncPCKConcurrencyQueueManager == null) {
				throw new ReferenceError('`GodotOS._asyncPCKConcurrencyQueueManager` is null.');
			}
			return await GodotOS._asyncPCKConcurrencyQueueManager.queue(() => fetch(...pArgs));
		},

		asyncPCKGetAsyncPCKAssetsDir: function (pPckDir) {
			let pckDir = GodotOS._trimLastSlash(pPckDir);
			if (pckDir.endsWith('.asyncpck')) {
				pckDir = `${pckDir}/assets`;
			}
			return pckDir;
		},

		asyncPCKGetAsyncPCKResource: function (pPckDir, pPath) {
			if (!GodotOS._asyncPCKInstallMap.has(pPckDir)) {
				return null;
			}
			const path = GodotOS._addResPrefix(pPath);
			if (!GodotOS._asyncPCKInstallMap.get(pPckDir).has(path)) {
				return null;
			}
			return GodotOS._asyncPCKInstallMap.get(pPckDir).get(path);
		},

		asyncPCKInstallFile: async function (pPckDir, pPath) {
			let asyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(pPckDir, pPath);
			if (asyncPCKResource != null) {
				if (
					asyncPCKResource.status != GodotOS.AsyncPCKFile.Status.STATUS_LOADING
					&& asyncPCKResource.status != GodotOS.AsyncPCKFile.Status.STATUS_INSTALLED
				) {
					// `GodotOS.AsyncPCKResource.load()` returns it's loading promise if it exists.
					await asyncPCKResource.load();
				}
				return;
			}
			asyncPCKResource = GodotOS.AsyncPCKResource.create(pPckDir, pPath);

			const assetsDir = GodotOS.asyncPCKGetAsyncPCKAssetsDir(pPckDir);
			const path = GodotOS._removeResPrefix(pPath);
			const depsJsonPath = `${assetsDir}/${path}.deps.json`;
			const depsJsonResponse = await GodotOS.asyncPCKFetch(depsJsonPath);
			if (!depsJsonResponse.ok) {
				GodotRuntime.error(`Couldn't load dependencies file "${depsJsonPath}".`);
				asyncPCKResource.removeFromInstallMap();
				return;
			}

			const remapResponseJson = await depsJsonResponse.json();
			const dependencies = remapResponseJson['dependencies'];
			const resources = remapResponseJson['resources'];

			// Initialize the desired resource ASAP.
			asyncPCKResource.initialize(resources[pPath].files, Object.keys(dependencies));

			const localAsyncPCKResources = Object.entries(resources).map(([pResourcePath, pResourceDefinition]) => {
				let localAsyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(pPckDir, pResourcePath);
				const resourceFiles = pResourceDefinition['files'];
				const resourceDependencies = dependencies?.[pResourcePath] ?? [];

				if (localAsyncPCKResource == null) {
					localAsyncPCKResource = GodotOS.AsyncPCKResource.createAndInitialize(
						pPckDir,
						pResourcePath,
						resourceFiles,
						resourceDependencies
					);
				} else if (!localAsyncPCKResource.initialized) {
					localAsyncPCKResource.initialize(resourceFiles, resourceDependencies);
				}

				return localAsyncPCKResource;
			});

			await Promise.allSettled(
				localAsyncPCKResources.map(async (pAsyncPCKResource) => {
					await pAsyncPCKResource.load();
				})
			);
		},

		asyncPCKInstallFileGetStatus: function (pPckDir, pPath) {
			const path = GodotOS._addResPrefix(pPath);
			const asyncPCKResource = GodotOS.asyncPCKGetAsyncPCKResource(pPckDir, path);
			if (asyncPCKResource == null) {
				return null;
			}
			const jsonObject = asyncPCKResource.getAsJsonObject();
			return jsonObject;
		},
	},

	godot_js_os_finish_async__proxy: 'sync',
	godot_js_os_finish_async__sig: 'vi',
	godot_js_os_finish_async: function (p_callback) {
		const func = GodotRuntime.get_func(p_callback);
		GodotOS.finish_async(func);
	},

	godot_js_os_request_quit_cb__proxy: 'sync',
	godot_js_os_request_quit_cb__sig: 'vi',
	godot_js_os_request_quit_cb: function (p_callback) {
		GodotOS.request_quit = GodotRuntime.get_func(p_callback);
	},

	godot_js_os_fs_is_persistent__proxy: 'sync',
	godot_js_os_fs_is_persistent__sig: 'i',
	godot_js_os_fs_is_persistent: function () {
		return GodotFS.is_persistent();
	},

	godot_js_os_fs_sync__proxy: 'sync',
	godot_js_os_fs_sync__sig: 'vi',
	godot_js_os_fs_sync: function (callback) {
		const func = GodotRuntime.get_func(callback);
		GodotOS._fs_sync_promise = GodotFS.sync();
		GodotOS._fs_sync_promise.then(function (err) {
			func();
		});
	},

	godot_js_os_has_feature__proxy: 'sync',
	godot_js_os_has_feature__sig: 'ii',
	godot_js_os_has_feature: function (p_ftr) {
		const ftr = GodotRuntime.parseString(p_ftr);
		const ua = navigator.userAgent;
		if (ftr === 'web_macos') {
			return (ua.indexOf('Mac') !== -1) ? 1 : 0;
		}
		if (ftr === 'web_windows') {
			return (ua.indexOf('Windows') !== -1) ? 1 : 0;
		}
		if (ftr === 'web_android') {
			return (ua.indexOf('Android') !== -1) ? 1 : 0;
		}
		if (ftr === 'web_ios') {
			return ((ua.indexOf('iPhone') !== -1) || (ua.indexOf('iPad') !== -1) || (ua.indexOf('iPod') !== -1)) ? 1 : 0;
		}
		if (ftr === 'web_linuxbsd') {
			return ((ua.indexOf('CrOS') !== -1) || (ua.indexOf('BSD') !== -1) || (ua.indexOf('Linux') !== -1) || (ua.indexOf('X11') !== -1)) ? 1 : 0;
		}
		return 0;
	},

	godot_js_os_asyncpck_install_file__proxy: 'sync',
	godot_js_os_asyncpck_install_file__sig: 'ipp',
	godot_js_os_asyncpck_install_file: function (pPckDirPtr, pPathPtr) {
		const pckDir = GodotOS._trimLastSlash(GodotRuntime.parseString(pPckDirPtr));
		const path = GodotOS._addResPrefix(GodotOS._trimLastSlash(GodotRuntime.parseString(pPathPtr)));

		GodotOS.asyncPCKInstallFile(pckDir, path).catch((err) => {
			GodotRuntime.error(`GodotOS.installAsyncFile("${pckDir}", "${path}")`, err);
		});
		return 0;
	},

	godot_js_os_asyncpck_install_file_get_status__proxy: 'sync',
	godot_js_os_asyncpck_install_file_get_status__sig: 'ppp',
	godot_js_os_asyncpck_install_file_get_status: function (pPckDirPtr, pPathPtr, pReturnStringLengthPtr) {
		const pckDir = GodotOS._trimLastSlash(GodotRuntime.parseString(pPckDirPtr));
		const path = GodotOS._addResPrefix(GodotOS._trimLastSlash(GodotRuntime.parseString(pPathPtr)));

		const status = GodotOS.asyncPCKInstallFileGetStatus(pckDir, path);
		if (status == null) {
			return 0;
		}

		const statusJson = JSON.stringify(status, null, 2);
		const statusJsonPtr = GodotRuntime.allocString(statusJson);
		if (pReturnStringLengthPtr !== 0) {
			GodotRuntime.setHeapValue(pReturnStringLengthPtr, GodotRuntime.strlen(statusJson), 'i32');
		}
		return statusJsonPtr;
	},

	godot_js_os_execute__proxy: 'sync',
	godot_js_os_execute__sig: 'ii',
	godot_js_os_execute: function (p_json) {
		const json_args = GodotRuntime.parseString(p_json);
		const args = JSON.parse(json_args);
		if (GodotConfig.on_execute) {
			GodotConfig.on_execute(args);
			return 0;
		}
		return 1;
	},

	godot_js_os_shell_open__proxy: 'sync',
	godot_js_os_shell_open__sig: 'vi',
	godot_js_os_shell_open: function (p_uri) {
		window.open(GodotRuntime.parseString(p_uri), '_blank');
	},

	godot_js_os_hw_concurrency_get__proxy: 'sync',
	godot_js_os_hw_concurrency_get__sig: 'i',
	godot_js_os_hw_concurrency_get: function () {
		// TODO Godot core needs fixing to avoid spawning too many threads (> 24).
		const concurrency = navigator.hardwareConcurrency || 1;
		return concurrency < 2 ? concurrency : 2;
	},

	godot_js_os_thread_pool_size_get__proxy: 'sync',
	godot_js_os_thread_pool_size_get__sig: 'i',
	godot_js_os_thread_pool_size_get: function () {
		if (typeof PThread === 'undefined') {
			// Threads aren't supported, so default to `1`.
			return 1;
		}

		return GodotConfig.godot_pool_size;
	},

	godot_js_os_download_buffer__proxy: 'sync',
	godot_js_os_download_buffer__sig: 'viiii',
	godot_js_os_download_buffer: function (p_ptr, p_size, p_name, p_mime) {
		const buf = GodotRuntime.heapSlice(HEAP8, p_ptr, p_size);
		const name = GodotRuntime.parseString(p_name);
		const mime = GodotRuntime.parseString(p_mime);
		const blob = new Blob([buf], { type: mime });
		const url = window.URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = name;
		a.style.display = 'none';
		document.body.appendChild(a);
		a.click();
		a.remove();
		window.URL.revokeObjectURL(url);
	},
};

autoAddDeps(_GodotOS, '$GodotOS');
mergeInto(LibraryManager.library, _GodotOS);

/*
 * Godot event listeners.
 * Keeps track of registered event listeners so it can remove them on shutdown.
 */
const GodotEventListeners = {
	$GodotEventListeners__deps: ['$GodotOS'],
	$GodotEventListeners__postset: 'GodotOS.atexit(function(resolve, reject) { GodotEventListeners.clear(); resolve(); });',
	$GodotEventListeners: {
		handlers: [],

		has: function (target, event, method, capture) {
			return GodotEventListeners.handlers.findIndex(function (e) {
				return e.target === target && e.event === event && e.method === method && e.capture === capture;
			}) !== -1;
		},

		add: function (target, event, method, capture) {
			if (GodotEventListeners.has(target, event, method, capture)) {
				return;
			}
			function Handler(p_target, p_event, p_method, p_capture) {
				this.target = p_target;
				this.event = p_event;
				this.method = p_method;
				this.capture = p_capture;
			}
			GodotEventListeners.handlers.push(new Handler(target, event, method, capture));
			target.addEventListener(event, method, capture);
		},

		clear: function () {
			GodotEventListeners.handlers.forEach(function (h) {
				h.target.removeEventListener(h.event, h.method, h.capture);
			});
			GodotEventListeners.handlers.length = 0;
		},
	},
};
mergeInto(LibraryManager.library, GodotEventListeners);

const GodotPWA = {

	$GodotPWA__deps: ['$GodotRuntime', '$GodotEventListeners'],
	$GodotPWA: {
		hasUpdate: false,

		updateState: function (cb, reg) {
			if (!reg) {
				return;
			}
			if (!reg.active) {
				return;
			}
			if (reg.waiting) {
				GodotPWA.hasUpdate = true;
				cb();
			}
			GodotEventListeners.add(reg, 'updatefound', function () {
				const installing = reg.installing;
				GodotEventListeners.add(installing, 'statechange', function () {
					if (installing.state === 'installed') {
						GodotPWA.hasUpdate = true;
						cb();
					}
				});
			});
		},
	},

	godot_js_pwa_cb__proxy: 'sync',
	godot_js_pwa_cb__sig: 'vi',
	godot_js_pwa_cb: function (p_update_cb) {
		if ('serviceWorker' in navigator) {
			try {
				const cb = GodotRuntime.get_func(p_update_cb);
				navigator.serviceWorker.getRegistration().then(GodotPWA.updateState.bind(null, cb));
			} catch (e) {
				GodotRuntime.error('Failed to assign PWA callback', e);
			}
		}
	},

	godot_js_pwa_update__proxy: 'sync',
	godot_js_pwa_update__sig: 'i',
	godot_js_pwa_update: function () {
		if ('serviceWorker' in navigator && GodotPWA.hasUpdate) {
			try {
				navigator.serviceWorker.getRegistration().then(function (reg) {
					if (!reg || !reg.waiting) {
						return;
					}
					reg.waiting.postMessage('update');
				});
			} catch (e) {
				GodotRuntime.error(e);
				return 1;
			}
			return 0;
		}
		return 1;
	},
};

autoAddDeps(GodotPWA, '$GodotPWA');
mergeInto(LibraryManager.library, GodotPWA);
