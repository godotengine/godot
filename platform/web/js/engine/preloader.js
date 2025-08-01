const Preloader = /** @constructor */ function () { // eslint-disable-line no-unused-vars
	function trackResponse(response, load_status) {
		async function trackReader(reader) {
			const result = await reader.read();
			if (load_status.done) {
				return Promise.resolve();
			}
			if (result.value) {
				load_status.loaded += result.value.length;
			}
			if (!result.done) {
				return trackReader(reader);
			}
			load_status.done = true;
			return Promise.resolve();
		}

		// Start the reader in parallel of the main response.
		const reader = response.clone().body.getReader();
		trackReader(reader).catch((_err) => {
			// TODO: Handle `_err`
		});

		// We need to keep the original response intact as much as possible.
		return response;
	}

	function loadFetch(file, tracker, fileSize, raw) {
		tracker[file] = {
			total: fileSize || 0,
			loaded: 0,
			done: false,
		};
		return fetch(file).then(function (response) {
			if (!response.ok) {
				return Promise.reject(new Error(`Failed loading file '${file}'`));
			}
			trackResponse(response, tracker[file]);
			if (raw) {
				return response;
			}
			return response.arrayBuffer();
		});
	}

	function retry(func, attempts = 1) {
		function onerror(err) {
			if (attempts <= 1) {
				return Promise.reject(err);
			}
			return new Promise(function (resolve, reject) {
				setTimeout(function () {
					retry(func, attempts - 1).then(resolve).catch(reject);
				}, 1000);
			});
		}
		return func().catch(onerror);
	}

	const DOWNLOAD_ATTEMPTS_MAX = 4;
	const loadingFiles = {};
	const lastProgress = { loaded: 0, total: 0 };
	let progressFunc = null;

	const animateProgress = function () {
		let loaded = 0;
		let total = 0;
		let totalIsValid = true;
		let progressIsFinal = true;

		Object.keys(loadingFiles).forEach(function (file) {
			const stat = loadingFiles[file];
			if (!stat.done) {
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
			if (typeof progressFunc === 'function') {
				progressFunc(loaded, total);
			}
		}
		if (!progressIsFinal) {
			requestAnimationFrame(animateProgress);
		}
	};

	this.animateProgress = animateProgress;

	this.setProgressFunc = function (callback) {
		progressFunc = callback;
	};

	this.loadPromise = function (file, fileSize, raw = false) {
		return retry(loadFetch.bind(null, file, loadingFiles, fileSize, raw), DOWNLOAD_ATTEMPTS_MAX);
	};

	this.preloadedFiles = [];
	this.preload = function (pathOrBuffer, destPath, fileSize) {
		let buffer = null;
		if (typeof pathOrBuffer === 'string') {
			const me = this;
			return this.loadPromise(pathOrBuffer, fileSize).then(function (buf) {
				me.preloadedFiles.push({
					path: destPath || pathOrBuffer,
					buffer: buf,
				});
				return Promise.resolve();
			});
		} else if (pathOrBuffer instanceof ArrayBuffer) {
			buffer = new Uint8Array(pathOrBuffer);
		} else if (ArrayBuffer.isView(pathOrBuffer)) {
			buffer = new Uint8Array(pathOrBuffer.buffer);
		}
		if (buffer) {
			this.preloadedFiles.push({
				path: destPath,
				buffer: pathOrBuffer,
			});
			return Promise.resolve();
		}
		return Promise.reject(new Error('Invalid object for preloading'));
	};
};
