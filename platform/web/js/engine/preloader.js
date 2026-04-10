const Preloader = /** @constructor */ function () { // eslint-disable-line no-unused-vars
	const DOWNLOAD_ATTEMPTS_MAX = 4;
	const loadingFiles = {};
	const lastProgress = { loaded: 0, total: 0 };
	let progressFunc = null;
	let concurrencyQueueManager = null;
	let filesSizeTotal = 0;

	/**
	 * @typedef {{
	 *  path: string;
	 *  buffer: Uint8Array | null;
	 *  fileSize: number;
	 * }} PreloadedFile
	 * @type {Map<string | Uint8Array, PreloadedFile>}
	 */
	this.preloadedFiles = new Map();

	function getTrackedResponse(pResponse, pLoadStatus) {
		async function onLoadProgress(pReader, pController) {
			const { done, value } = await pReader.read();
			if (pLoadStatus.done) {
				return Promise.resolve();
			}
			if (done) {
				pLoadStatus.done = true;
				return Promise.resolve();
			}
			pController.enqueue(value);
			pLoadStatus.loaded += value.byteLength;
			return onLoadProgress(pReader, pController);
		}

		const reader = pResponse.clone().body.getReader();
		return new Response(new ReadableStream({
			start: async function (pController) {
				try {
					await onLoadProgress(reader, pController);
				} finally {
					pController.close();
				}
			},
		}), { headers: pResponse.headers });
	}

	async function loadFetch(pFile, pFileSize, pIsRaw) {
		if (pFile in loadingFiles) {
			loadingFiles[pFile].requested = true;
		} else {
			loadingFiles[pFile] = {
				file: pFile,
				total: pFileSize || 0,
				loaded: 0,
				requested: true,
				done: false,
			};
		}

		try {
			const response = await fetch(pFile);

			if (!response.ok) {
				throw new Error(`Got response ${response.status}: ${response.statusText}`);
			}
			const tr = getTrackedResponse(response, loadingFiles[pFile]);
			if (pIsRaw) {
				return Promise.resolve(tr);
			}

			return tr.arrayBuffer();
		} catch (error) {
			const newError = new Error(`loadFetch for "${pFile}" failed:`);
			newError.cause = error;
			throw newError;
		}
	}

	function retry(pCallback, pAttempts = 1) {
		function onerror(err) {
			if (pAttempts <= 1) {
				return Promise.reject(err);
			}
			return new Promise(function (resolve, reject) {
				setTimeout(function () {
					retry(pCallback, pAttempts - 1).then(resolve).catch(reject);
				}, 1000);
			});
		}
		return pCallback().catch(onerror);
	}

	this.animateProgress = () => {
		let loaded = 0;
		const requestedFiles = Object.values(loadingFiles)
			.filter((pLoadingFile) => pLoadingFile.requested);
		let progressIsFinal = false;
		if (requestedFiles.length > 0 && requestedFiles.every((pRequestedFile) => pRequestedFile.done)) {
			progressIsFinal = true;
		}

		// eslint-disable-next-line no-unused-vars
		for (const [_file, status] of Object.entries(loadingFiles)) {
			if (!status.requested) {
				continue;
			}

			if (!status.done) {
				progressIsFinal = false;
			}

			loaded += status.loaded;
		}

		if (loaded !== lastProgress.loaded || filesSizeTotal !== lastProgress.total) {
			lastProgress.loaded = loaded;
			lastProgress.total = filesSizeTotal;

			if (typeof progressFunc === 'function') {
				progressFunc(loaded, filesSizeTotal);
			}
		}
		if (!progressIsFinal) {
			window.requestAnimationFrame(() => this.animateProgress());
		}
	};

	this.setProgressFunc = function (pCallback) {
		progressFunc = pCallback;
	};

	this.loadPromise = async function (pFile, pFileSize, pIsRaw = false) {
		if (concurrencyQueueManager == null) {
			const { ConcurrencyQueueManager } = await import('@godotengine/utils/concurrencyQueueManager');
			// Another `loadPromise()` could have ended while awaiting.
			if (concurrencyQueueManager == null) {
				concurrencyQueueManager = new ConcurrencyQueueManager();
			}
		}

		try {
			return await concurrencyQueueManager.queue(() => retry(
				async () => await loadFetch(pFile, pFileSize, pIsRaw),
				DOWNLOAD_ATTEMPTS_MAX
			));
		} catch (error) {
			const newError = new Error(`An error occurred while running \`Preloader.loadPromise("${pFile}", ${pFileSize}, pIsRaw = ${pIsRaw})\``);
			newError.cause = error;
			throw error;
		}
	};

	this.preload = async (pPathOrBuffer, pDestPath, pFileSize) => {
		let buffer = null;
		if (typeof pPathOrBuffer === 'string') {
			const path = pPathOrBuffer;
			const preloadedFileEntry = this.preloadedFiles.get(path);
			const me = this;

			if (preloadedFileEntry == null) {
				filesSizeTotal += pFileSize;
			}

			buffer = await this.loadPromise(path, pFileSize);

			if (preloadedFileEntry == null) {
				me.preloadedFiles.set(path, {
					path: pDestPath ?? pPathOrBuffer,
					buffer,
					fileSize: pFileSize,
				});
			} else {
				preloadedFileEntry.buffer = buffer;
			}

			return;
		} else if (pPathOrBuffer instanceof ArrayBuffer) {
			buffer = new Uint8Array(pPathOrBuffer);
		} else if (ArrayBuffer.isView(pPathOrBuffer)) {
			buffer = new Uint8Array(pPathOrBuffer.buffer);
		}
		if (buffer == null) {
			throw new Error('Invalid object for preloading');
		}

		filesSizeTotal += pFileSize;
		this.preloadedFiles.set(buffer, {
			path: pDestPath,
			buffer,
			fileSize: pFileSize,
		});
	};

	this.preparePreload = (pPath, pDestPath, pFileSize) => {
		this.preloadedFiles.set(pPath, {
			path: pDestPath,
			buffer: null,
			fileSize: pFileSize,
		});
		filesSizeTotal += pFileSize;
	};

	this.init = (pOptions = {}) => {
		const {
			fileSizes: loadingFileSizes = {},
		} = pOptions;

		for (const [file, fileSize] of Object.entries(loadingFileSizes)) {
			loadingFiles[file] = {
				file,
				total: fileSize || 0,
				loaded: 0,
				requested: false,
				done: false,
			};
		}
	};
};
