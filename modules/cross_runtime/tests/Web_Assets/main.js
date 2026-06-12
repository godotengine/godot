(async function () {
	// UI helpers
	const statusOverlay = document.getElementById('status');
	const statusProgress = document.getElementById('status-progress');
	const statusNotice = document.getElementById('status-notice');

	let initializing = true;

	function setStatusMode(mode) {
		if (!initializing) {
			return;
		}

		if (mode === 'hidden') {
			statusOverlay.remove();
			initializing = false;
			return;
		}

		statusOverlay.style.visibility = 'visible';
		statusProgress.style.display = mode === 'progress' ? 'block' : 'none';
		statusNotice.style.display = mode === 'notice' ? 'block' : 'none';
	}

	function setStatusNotice(text) {
		statusNotice.innerText = text;
		statusNotice.style.display = 'block';
	}

	function displayFailureNotice(err) {
		setStatusNotice(err instanceof Error ? err.message : String(err));
		setStatusMode('notice');
	}

	// Start Godot
	const GODOT_CONFIG = {
		args: [],
		canvasResizePolicy: 2,
		executable: 'godot',
		mainPack: 'godot.pck',
		fileSizes: {
			'godot.wasm': 0,
			'godot.pck': 0,
		},
		gdextensionlibs: [],
		// Keep rtenv alive so we can read HEAPU8 after startGame
		unloadAfterInit: false,
	};

	const Engine = globalThis.Engine;
	if (typeof Engine !== 'function') {
		displayFailureNotice('Engine global is not available');
		return;
	}

	const missing = Engine.getMissingFeatures({ threads: true });
	if (missing.length) {
		displayFailureNotice(`Missing features:\n${missing.join('\n')}`);
		return;
	}

	window.engine = new Engine(GODOT_CONFIG);

	setStatusMode('progress');
	setStatusNotice('Loading engine...');

	try {
		await window.engine.startGame({
			onProgress: (current, total) => {
				if (current > 0 && total > 0) {
					statusProgress.value = current;
					statusProgress.max = total;
				} else {
					statusProgress.removeAttribute('value');
					statusProgress.removeAttribute('max');
				}
			},
		});
	} catch (err) {
		displayFailureNotice(err);
		return;
	}

	// Acquire Godot heap
	const heapU8 = window.engine?.rtenv?.['HEAPU8'];
	if (!(heapU8 instanceof Uint8Array)) {
		displayFailureNotice('Failed to obtain Godot HEAPU8 after startGame');
		return;
	}

	const heapBuffer = heapU8.buffer;

	// Launch simulation worker
	setStatusNotice('Starting simulation worker...');

	const dotnetJsUrl = new URL('cs/_framework/dotnet.js', window.location.href).href;

	const worker = new Worker(new URL('./dotnet_worker.js', import.meta.url), {
		type: 'module',
	});

	worker.onmessage = function (e) {
		if (e.data.type === 'ready') {
			window.interopReady = true;
			setStatusMode('hidden');
		} else if (e.data.type === 'error') {
			displayFailureNotice(`Worker error: ${e.data.message}`);
		}
	};

	worker.onerror = function (err) {
		displayFailureNotice(`Simulation worker crashed: ${err.message}`);
	};

	worker.postMessage({
		type: 'init',
		buffer: heapBuffer,
		dotnetUrl: dotnetJsUrl,
	});
})();
