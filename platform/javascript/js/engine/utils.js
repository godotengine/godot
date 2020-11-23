const Utils = { // eslint-disable-line no-unused-vars

	createLocateRewrite: function (execName) {
		function rw(path) {
			if (path.endsWith('.worker.js')) {
				return `${execName}.worker.js`;
			} else if (path.endsWith('.audio.worklet.js')) {
				return `${execName}.audio.worklet.js`;
			} else if (path.endsWith('.js')) {
				return `${execName}.js`;
			} else if (path.endsWith('.wasm')) {
				return `${execName}.wasm`;
			}
			return path;
		}
		return rw;
	},

	createInstantiatePromise: function (wasmLoader) {
		let loader = wasmLoader;
		function instantiateWasm(imports, onSuccess) {
			loader.then(function (xhr) {
				WebAssembly.instantiate(xhr.response, imports).then(function (result) {
					onSuccess(result['instance'], result['module']);
				});
			});
			loader = null;
			return {};
		}

		return instantiateWasm;
	},

	findCanvas: function () {
		const nodes = document.getElementsByTagName('canvas');
		if (nodes.length && nodes[0] instanceof HTMLCanvasElement) {
			return nodes[0];
		}
		return null;
	},

	isWebGLAvailable: function (majorVersion = 1) {
		let testContext = false;
		try {
			const testCanvas = document.createElement('canvas');
			if (majorVersion === 1) {
				testContext = testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
			} else if (majorVersion === 2) {
				testContext = testCanvas.getContext('webgl2') || testCanvas.getContext('experimental-webgl2');
			}
		} catch (e) {
			// Not available
		}
		return !!testContext;
	},
};
