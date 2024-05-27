const Features = {
	/**
	 * Check whether WebGL is available. Optionally, specify a particular version of WebGL to check for.
	 *
	 * @param {number=} [majorVersion=1] The major WebGL version to check for.
	 * @returns {boolean} If the given major version of WebGL is available.
	 * @function Engine.isWebGLAvailable
	 */
	isWebGLAvailable: function (majorVersion = 1) {
		try {
			return !!document.createElement('canvas').getContext(['webgl', 'webgl2'][majorVersion - 1]);
		} catch (e) { /* Not available */ }
		return false;
	},

	/**
	 * Check whether the Fetch API available and supports streaming responses.
	 *
	 * @returns {boolean} If the Fetch API is available and supports streaming responses.
	 * @function Engine.isFetchAvailable
	 */
	isFetchAvailable: function () {
		return 'fetch' in window && 'Response' in window && 'body' in window.Response.prototype;
	},

	/**
	 * Check whether the engine is running in a Secure Context.
	 *
	 * @returns {boolean} If the engine is running in a Secure Context.
	 * @function Engine.isSecureContext
	 */
	isSecureContext: function () {
		return window['isSecureContext'] === true;
	},

	/**
	 * Check whether the engine is cross origin isolated.
	 * This value is dependent on Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers sent by the server.
	 *
	 * @returns {boolean} If the engine is running in a Secure Context.
	 * @function Engine.isSecureContext
	 */
	isCrossOriginIsolated: function () {
		return window['crossOriginIsolated'] === true;
	},

	/**
	 * Check whether SharedBufferArray is available.
	 *
	 * Most browsers require the page to be running in a secure context, and the
	 * the server to provide specific CORS headers for SharedArrayBuffer to be available.
	 *
	 * @returns {boolean} If SharedArrayBuffer is available.
	 * @function Engine.isSharedArrayBufferAvailable
	 */
	isSharedArrayBufferAvailable: function () {
		return 'SharedArrayBuffer' in window;
	},

	/**
	 * Check whether the AudioContext supports AudioWorkletNodes.
	 *
	 * @returns {boolean} If AudioWorkletNode is available.
	 * @function Engine.isAudioWorkletAvailable
	 */
	isAudioWorkletAvailable: function () {
		return 'AudioContext' in window && 'audioWorklet' in AudioContext.prototype;
	},

	/**
	 * Return an array of missing required features (as string).
	 *
	 * @returns {Array<string>} A list of human-readable missing features.
	 * @function Engine.getMissingFeatures
	 * @param {{threads: (boolean|undefined)}} supportedFeatures
	 */
	getMissingFeatures: function (supportedFeatures = {}) {
		const {
			threads: supportsThreads = true,
		} = supportedFeatures;

		const missing = [];
		if (!Features.isWebGLAvailable(2)) {
			missing.push('WebGL2 - Check web browser configuration and hardware support');
		}
		if (!Features.isFetchAvailable()) {
			missing.push('Fetch - Check web browser version');
		}
		if (!Features.isSecureContext()) {
			missing.push('Secure Context - Check web server configuration (use HTTPS)');
		}

		if (supportsThreads) {
			if (!Features.isCrossOriginIsolated()) {
				missing.push('Cross-Origin Isolation - Check that the web server configuration sends the correct headers.');
			}
			if (!Features.isSharedArrayBufferAvailable()) {
				missing.push('SharedArrayBuffer - Check that the web server configuration sends the correct headers.');
			}
		}

		// Audio is normally optional since we have a dummy fallback.
		return missing;
	},
};
