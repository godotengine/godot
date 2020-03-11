var Preloader = /** @constructor */ function() {

	var DOWNLOAD_ATTEMPTS_MAX = 4;
	var progressFunc = null;
	var lastProgress = { loaded: 0, total: 0 };

	var loadingFiles = {};
	this.preloadedFiles = [];

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

	this.loadPromise = function(file) {
		return new Promise(function(resolve, reject) {
			loadXHR(resolve, reject, file, loadingFiles);
		});
	}

	this.preload = function(pathOrBuffer, destPath) {
		if (pathOrBuffer instanceof ArrayBuffer) {
			pathOrBuffer = new Uint8Array(pathOrBuffer);
		} else if (ArrayBuffer.isView(pathOrBuffer)) {
			pathOrBuffer = new Uint8Array(pathOrBuffer.buffer);
		}
		if (pathOrBuffer instanceof Uint8Array) {
			this.preloadedFiles.push({
				path: destPath,
				buffer: pathOrBuffer
			});
			return Promise.resolve();
		} else if (typeof pathOrBuffer === 'string') {
			var me = this;
			return this.loadPromise(pathOrBuffer).then(function(xhr) {
				me.preloadedFiles.push({
					path: destPath || pathOrBuffer,
					buffer: xhr.response
				});
				return Promise.resolve();
			});
		} else {
			throw Promise.reject("Invalid object for preloading");
		}
	};

	var animateProgress = function() {

		var loaded = 0;
		var total = 0;
		var totalIsValid = true;
		var progressIsFinal = true;

		Object.keys(loadingFiles).forEach(function(file) {
			const stat = loadingFiles[file];
			if (!stat.final) {
				progressIsFinal = false;
			}
			if (!totalIsValid || stat.total === 0) {
				totalIsValid = false;
				total = 0;
			} else {
				total += stat.total;
			}
			loaded += stat.loaded;
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
	this.animateProgress = animateProgress; // Also exposed to start it.

	this.setProgressFunc = function(callback) {
		progressFunc = callback;
	}
};
