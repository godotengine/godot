/**************************************************************************/
/*  library_godot_camera.js                                               */
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

/**
 * @typedef {{
 *   deviceId: string
 *   label: string
 *   index: number
 * }} CameraInfo
 *
 * @typedef {{
 *   video: HTMLVideoElement|null
 *   canvas: (HTMLCanvasElement|OffscreenCanvas)|null
 *   canvasContext: (CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D)|null
 *   stream: MediaStream|null
 *   animationFrameId: number|null
 *   permissionListener: Function|null
 *   permissionStatus: PermissionStatus|null
 *   trackProcessor: MediaStreamTrackProcessor|null
 *   frameReader: ReadableStreamDefaultReader|null
 *   worker: Worker|null
 *   useWebCodecsWorker: boolean
 *   useWebCodecs: boolean
 *   useWorker: boolean
 * }} CameraResource
 */

const GodotCamera = {
	$GodotCamera__deps: ['$GodotRuntime', '$GodotConfig', '$GodotOS'],
	$GodotCamera__postset: 'GodotOS.atexit(function(resolve, reject) { GodotCamera.cleanup(); resolve(); });',
	$GodotCamera: {
		/**
		 * Map to manage resources for each camera.
		 * @type {Map<string, CameraResource>}
		 */
		cameras: new Map(),

		/**
		 * Cached result of Web Worker support check.
		 * @type {boolean|null}
		 */
		workerSupported: null,

		/**
		 * Blob URL for the camera worker script.
		 * @type {string|null}
		 */
		workerBlobUrl: null,

		defaultMinimumCapabilities: {
			'width': {
				'min': 1,
				'max': 1280,
			},
			'height': {
				'min': 1,
				'max': 720,
			},
		},

		/**
		 * Common resolutions to check against camera capabilities.
		 */
		commonResolutions: [
			{ width: 320, height: 240 }, // QVGA (4:3)
			{ width: 352, height: 288 }, // CIF (4:3) - Video conferencing
			{ width: 640, height: 480 }, // VGA (4:3)
			{ width: 1024, height: 768 }, // XGA (4:3)
			{ width: 1280, height: 720 }, // HD 720p (16:9)
			{ width: 1280, height: 960 }, // SXGA- (4:3)
			{ width: 1600, height: 1200 }, // UXGA (4:3)
			{ width: 1920, height: 1080 }, // Full HD 1080p (16:9)
			{ width: 2560, height: 1440 }, // QHD 1440p (16:9)
			{ width: 3840, height: 2160 }, // 4K UHD 2160p (16:9)
		],

		/**
		 * Gets supported formats based on capabilities.
		 * @param {Object} capabilities MediaTrackCapabilities object
		 * @returns {Array<{width: number, height: number, frameRate: number}>} Supported resolutions with frame rate
		 */
		getSupportedFormats: function (capabilities) {
			const widthRange = capabilities.width || this.defaultMinimumCapabilities.width;
			const heightRange = capabilities.height || this.defaultMinimumCapabilities.height;
			const frameRateRange = capabilities.frameRate || { min: 1, max: 30 };
			const maxFrameRate = frameRateRange.max || 30;

			return this.commonResolutions
				.filter((res) => res.width >= (widthRange.min || 1)
					&& res.width <= (widthRange.max || 9999)
					&& res.height >= (heightRange.min || 1)
					&& res.height <= (heightRange.max || 9999))
				.map((res) => ({
					width: res.width,
					height: res.height,
					frameRate: maxFrameRate,
				}));
		},

		/**
		 * Checks if WebCodecs API is supported for camera capture.
		 * @returns {boolean} True if MediaStreamTrackProcessor and VideoFrame are available
		 */
		isWebCodecsSupported: function () {
			return 'MediaStreamTrackProcessor' in window && 'VideoFrame' in window;
		},

		/**
		 * Checks if Web Worker for camera frame processing is supported.
		 * Requires Worker, OffscreenCanvas, and createImageBitmap support.
		 * @returns {boolean} True if worker-based capture is supported
		 */
		isWorkerSupported: function () {
			if (this.workerSupported !== null) {
				return this.workerSupported;
			}

			try {
				// Check for Worker support
				if (typeof Worker === 'undefined') {
					this.workerSupported = false;
					return false;
				}

				// Check for OffscreenCanvas support (required in worker)
				if (typeof OffscreenCanvas === 'undefined') {
					this.workerSupported = false;
					return false;
				}

				// Check for createImageBitmap support (required for transferring frames)
				if (typeof createImageBitmap === 'undefined') {
					this.workerSupported = false;
					return false;
				}

				this.workerSupported = true;
				GodotRuntime.print('Web Worker for camera capture is supported');
			} catch (e) {
				GodotRuntime.print('Web Worker support check failed:', e.message);
				this.workerSupported = false;
			}

			return this.workerSupported;
		},

		/**
		 * Creates or returns the Blob URL for the camera worker script.
		 * The worker is embedded as inline code to avoid external file dependencies.
		 * @returns {string|null} Blob URL for the worker, or null if creation fails
		 */
		getWorkerBlobUrl: function () {
			if (this.workerBlobUrl) {
				return this.workerBlobUrl;
			}

			// Worker code embedded as a string.
			// Supports both VideoFrame (WebCodecs) and ImageBitmap (Canvas 2D) processing.
			const workerCode = `
// Worker state
let canvas = null;
let canvasContext = null;
let width = 0;
let height = 0;
let isCapturing = false;

// Process ImageBitmap using Canvas 2D (fallback method)
function processImageBitmap(imageBitmap) {
	const frameWidth = imageBitmap.width;
	const frameHeight = imageBitmap.height;

	if (canvas.width !== frameWidth || canvas.height !== frameHeight) {
		canvas.width = frameWidth;
		canvas.height = frameHeight;
		width = frameWidth;
		height = frameHeight;
		canvasContext = canvas.getContext('2d', { willReadFrequently: true });
	}

	canvasContext.drawImage(imageBitmap, 0, 0, width, height);
	const imageData = canvasContext.getImageData(0, 0, width, height);
	imageBitmap.close();

	return {
		pixelData: imageData.data,
		width: width,
		height: height,
	};
}

// Process VideoFrame using WebCodecs copyTo (most efficient)
async function processVideoFrame(videoFrame) {
	const frameWidth = videoFrame.displayWidth;
	const frameHeight = videoFrame.displayHeight;
	const bufferSize = frameWidth * frameHeight * 4;
	const pixelBuffer = new Uint8Array(bufferSize);

	try {
		await videoFrame.copyTo(pixelBuffer, {
			rect: { x: 0, y: 0, width: frameWidth, height: frameHeight },
			layout: [{ offset: 0, stride: frameWidth * 4 }],
			format: 'RGBA',
		});

		return {
			pixelData: pixelBuffer,
			width: frameWidth,
			height: frameHeight,
		};
	} finally {
		videoFrame.close();
	}
}

self.onmessage = async function(event) {
	const { type, data } = event.data;

	switch (type) {
	case 'init':
		canvas = new OffscreenCanvas(data.width || 640, data.height || 480);
		width = data.width || 640;
		height = data.height || 480;
		canvasContext = canvas.getContext('2d', { willReadFrequently: true });
		isCapturing = true;
		self.postMessage({ type: 'initialized' });
		break;

	case 'videoFrame':
		// WebCodecs VideoFrame processing
		if (!isCapturing) {
			if (data.videoFrame) {
				data.videoFrame.close();
			}
			return;
		}

		try {
			const result = await processVideoFrame(data.videoFrame);
			self.postMessage(
				{
					type: 'frameData',
					pixelData: result.pixelData,
					width: result.width,
					height: result.height,
					orientation: data.orientation,
					facingMode: data.facingMode,
				},
				[result.pixelData.buffer]
			);
		} catch (error) {
			self.postMessage({
				type: 'error',
				message: error.message,
			});
		}
		break;

	case 'frame':
		// Canvas 2D ImageBitmap processing (fallback)
		if (!isCapturing || !canvas) {
			if (data.imageBitmap) {
				data.imageBitmap.close();
			}
			return;
		}

		try {
			const result = processImageBitmap(data.imageBitmap);
			self.postMessage(
				{
					type: 'frameData',
					pixelData: result.pixelData,
					width: result.width,
					height: result.height,
					orientation: data.orientation,
					facingMode: data.facingMode,
				},
				[result.pixelData.buffer]
			);
		} catch (error) {
			self.postMessage({
				type: 'error',
				message: error.message,
			});
		}
		break;

	case 'stop':
		isCapturing = false;
		canvas = null;
		canvasContext = null;
		self.postMessage({ type: 'stopped' });
		break;

	default:
		console.warn('Unknown message type:', type);
	}
};
`;

			try {
				const blob = new Blob([workerCode], { type: 'application/javascript' });
				this.workerBlobUrl = URL.createObjectURL(blob);
				return this.workerBlobUrl;
			} catch (e) {
				GodotRuntime.print('Failed to create worker blob URL:', e.message);
				return null;
			}
		},

		/**
		 * Ensures cameras Map is properly initialized.
		 * @returns {Map<string, CameraResource>}
		 */
		ensureCamerasMap: function () {
			if (!this.cameras || !(this.cameras instanceof Map)) {
				this.cameras = new Map();
			}
			return this.cameras;
		},

		/**
		 * Cleanup all camera resources.
		 * @returns {void}
		 */
		cleanup: function () {
			this.api.stop();

			// Revoke worker blob URL to free memory.
			if (this.workerBlobUrl) {
				URL.revokeObjectURL(this.workerBlobUrl);
				this.workerBlobUrl = null;
			}
		},

		/**
		 * Sends a JSON result to the callback function.
		 * @param {Function} callback Callback function pointer
		 * @param {number} callbackPtr Context value to pass to callback
		 * @param {number} context Context value to pass to callback
		 * @param {Object} result Result object to stringify
		 * @returns {void}
		 */
		sendCamerasCallbackResult: function (callback, callbackPtr, context, result) {
			const jsonStr = JSON.stringify(result);
			const strPtr = GodotRuntime.allocString(jsonStr);
			callback(context, callbackPtr, strPtr);
			GodotRuntime.free(strPtr);
		},

		/**
		 * Sends pixel data or error to the callback function.
		 * @param {Function} callback Callback function pointer
		 * @param {number} context Context value to pass to callback
		 * @param {number} dataPtr Pointer to pixel data
		 * @param {number} dataLen Length of pixel data
		 * @param {number} width Image width
		 * @param {number} height Image height
		 * @param {number} orientation Screen orientation angle (0, 90, 180, 270)
		 * @param {number} facingMode Camera facing mode (0=unknown, 1=user/front, 2=environment/back)
		 * @param {string|null} errorMsg Error message if any
		 * @returns {void}
		 */
		sendGetPixelDataCallback: function (callback, context, dataPtr, dataLen, width, height, orientation, facingMode, errorMsg) {
			const errorMsgPtr = errorMsg ? GodotRuntime.allocString(errorMsg) : 0;
			callback(context, dataPtr, dataLen, width, height, orientation, facingMode, errorMsgPtr);
			if (errorMsgPtr) {
				GodotRuntime.free(errorMsgPtr);
			}
		},

		/**
		 * Converts facingMode string to numeric value.
		 * @param {MediaStream|null} stream Media stream to get facing mode from
		 * @returns {number} 0=unknown, 1=user/front, 2=environment/back
		 */
		getFacingMode: function (stream) {
			if (!stream) {
				return 0;
			}
			const [videoTrack] = stream.getVideoTracks();
			if (!videoTrack) {
				return 0;
			}
			const settings = videoTrack.getSettings();
			switch (settings.facingMode) {
			case 'user':
				return 1; // Front camera
			case 'environment':
				return 2; // Back camera
			default:
				return 0; // Unknown
			}
		},

		/**
		 * Sets up WebCodecs-based frame capture using MediaStreamTrackProcessor.
		 * @param {CameraResource} camera Camera resource
		 * @param {string} cameraId Camera identifier
		 * @param {Function} callback Callback function for frame data
		 * @param {number} context Context value to pass to callback
		 * @param {Function} deniedCallback Callback for permission denied
		 * @returns {void}
		 */
		setupWebCodecsCapture: function (camera, cameraId, callback, context, deniedCallback) {
			const [videoTrack] = camera.stream.getVideoTracks();

			camera.trackProcessor = new MediaStreamTrackProcessor({ track: videoTrack });
			camera.frameReader = camera.trackProcessor.readable.getReader();

			const processFrames = async () => {
				const cameras = GodotCamera.ensureCamerasMap();
				try {
					while (true) {
						const currentCamera = cameras.get(cameraId);
						if (!currentCamera || !currentCamera.useWebCodecs) {
							break;
						}

						// eslint-disable-next-line no-await-in-loop
						const { done, value: videoFrame } = await currentCamera.frameReader.read();
						if (done) {
							break;
						}

						try {
							const width = videoFrame.displayWidth;
							const height = videoFrame.displayHeight;
							const bufferSize = width * height * 4;
							const pixelBuffer = new Uint8Array(bufferSize);

							// eslint-disable-next-line no-await-in-loop
							await videoFrame.copyTo(pixelBuffer, {
								rect: { x: 0, y: 0, width, height },
								layout: [{ offset: 0, stride: width * 4 }],
								format: 'RGBA',
							});

							const dataPtr = GodotRuntime.malloc(pixelBuffer.length);
							GodotRuntime.heapCopy(HEAPU8, pixelBuffer, dataPtr);

							const screenOrientation = screen?.orientation?.angle ?? window.orientation ?? 0;
							const facingMode = GodotCamera.getFacingMode(currentCamera.stream);

							GodotCamera.sendGetPixelDataCallback(
								callback,
								context,
								dataPtr,
								pixelBuffer.length,
								width,
								height,
								screenOrientation,
								facingMode,
								null
							);

							GodotRuntime.free(dataPtr);
						} finally {
							videoFrame.close();
						}
					}
				} catch (error) {
					GodotRuntime.print('WebCodecs error, falling back to Canvas 2D:', error.message);
					const currentCamera = cameras.get(cameraId);
					if (currentCamera) {
						// Clean up WebCodecs resources before fallback.
						if (currentCamera.frameReader) {
							currentCamera.frameReader.cancel().catch(() => {});
							currentCamera.frameReader = null;
						}
						currentCamera.trackProcessor = null;
						currentCamera.useWebCodecs = false;

						// Fall back to Worker-based Canvas 2D if supported, else main thread.
						if (GodotCamera.isWorkerSupported()) {
							currentCamera.useWorker = true;
							GodotCamera.setupCanvas2DWorkerCapture(currentCamera, cameraId, callback, context, deniedCallback);
						} else {
							GodotCamera.setupCanvas2DCapture(currentCamera, cameraId, callback, context, deniedCallback);
						}
					}
				}
			};

			processFrames();
		},

		/**
		 * Sets up WebCodecs + Worker frame capture.
		 * Uses MediaStreamTrackProcessor to get VideoFrame, transfers to Worker for copyTo().
		 * This is the most efficient method as it offloads processing to a worker thread.
		 * @param {CameraResource} camera Camera resource
		 * @param {string} cameraId Camera identifier
		 * @param {Function} callback Callback function for frame data
		 * @param {number} context Context value to pass to callback
		 * @param {Function} deniedCallback Callback for permission denied
		 * @returns {void}
		 */
		setupWebCodecsWorkerCapture: function (camera, cameraId, callback, context, deniedCallback) {
			const workerUrl = this.getWorkerBlobUrl();
			if (!workerUrl) {
				GodotRuntime.print('Failed to create worker, falling back to WebCodecs main thread');
				this.setupWebCodecsCapture(camera, cameraId, callback, context, deniedCallback);
				return;
			}

			const [videoTrack] = camera.stream.getVideoTracks();
			const { width: _width, height: _height } = videoTrack.getSettings();

			try {
				camera.worker = new Worker(workerUrl);
			} catch (e) {
				GodotRuntime.print('Failed to create worker:', e.message);
				this.setupWebCodecsCapture(camera, cameraId, callback, context, deniedCallback);
				return;
			}

			// Set up MediaStreamTrackProcessor

			camera.trackProcessor = new MediaStreamTrackProcessor({ track: videoTrack });
			camera.frameReader = camera.trackProcessor.readable.getReader();

			// Handle messages from worker
			camera.worker.onmessage = (event) => {
				const { type } = event.data;

				switch (type) {
				case 'initialized':
					GodotRuntime.print('Camera worker initialized (WebCodecs mode)');
					break;

				case 'frameData': {
					const { pixelData, width, height, orientation, facingMode } = event.data;
					const dataPtr = GodotRuntime.malloc(pixelData.length);
					GodotRuntime.heapCopy(HEAPU8, pixelData, dataPtr);

					GodotCamera.sendGetPixelDataCallback(
						callback,
						context,
						dataPtr,
						pixelData.length,
						width,
						height,
						orientation,
						facingMode,
						null
					);

					GodotRuntime.free(dataPtr);
					break;
				}

				case 'error':
					GodotRuntime.print('Worker error:', event.data.message);
					// Fall back to Worker Canvas 2D on error
					camera.useWebCodecsWorker = false;
					if (camera.frameReader) {
						camera.frameReader.cancel().catch(() => {});
						camera.frameReader = null;
					}
					camera.trackProcessor = null;
					// Keep the worker for Canvas 2D fallback
					camera.useWorker = true;
					GodotCamera.setupCanvas2DWorkerCapture(camera, cameraId, callback, context, deniedCallback);
					break;

				case 'stopped':
					GodotRuntime.print('Camera worker stopped');
					break;

				default:
					break;
				}
			};

			camera.worker.onerror = (error) => {
				GodotRuntime.print('Worker error event:', error.message);
				camera.useWebCodecsWorker = false;
				if (camera.frameReader) {
					camera.frameReader.cancel().catch(() => {});
					camera.frameReader = null;
				}
				camera.trackProcessor = null;
				if (camera.worker) {
					camera.worker.terminate();
					camera.worker = null;
				}
				// Fall back to WebCodecs main thread
				camera.useWebCodecs = true;
				GodotCamera.setupWebCodecsCapture(camera, cameraId, callback, context, deniedCallback);
			};

			// Initialize the worker
			camera.worker.postMessage({
				type: 'init',
				data: { width: _width, height: _height },
			});

			// Start the frame reading loop
			const processFrames = async () => {
				const cameras = GodotCamera.ensureCamerasMap();
				try {
					while (true) {
						const currentCamera = cameras.get(cameraId);
						if (!currentCamera || !currentCamera.useWebCodecsWorker || !currentCamera.worker) {
							break;
						}

						// eslint-disable-next-line no-await-in-loop
						const { done, value: videoFrame } = await currentCamera.frameReader.read();
						if (done) {
							break;
						}

						const screenOrientation = screen?.orientation?.angle ?? window.orientation ?? 0;
						const facingMode = GodotCamera.getFacingMode(currentCamera.stream);

						// Transfer VideoFrame to worker
						currentCamera.worker.postMessage(
							{
								type: 'videoFrame',
								data: {
									videoFrame,
									orientation: screenOrientation,
									facingMode,
								},
							},
							[videoFrame]
						);
					}
				} catch (error) {
					GodotRuntime.print('WebCodecs Worker error, falling back:', error.message);
					const currentCamera = cameras.get(cameraId);
					if (currentCamera && currentCamera.useWebCodecsWorker) {
						currentCamera.useWebCodecsWorker = false;
						if (currentCamera.frameReader) {
							currentCamera.frameReader.cancel().catch(() => {});
							currentCamera.frameReader = null;
						}
						currentCamera.trackProcessor = null;

						// Fall back to Worker Canvas 2D
						if (currentCamera.worker) {
							currentCamera.useWorker = true;
							GodotCamera.setupCanvas2DWorkerCapture(currentCamera, cameraId, callback, context, deniedCallback);
						} else {
							// Fall back to WebCodecs main thread
							currentCamera.useWebCodecs = true;
							GodotCamera.setupWebCodecsCapture(currentCamera, cameraId, callback, context, deniedCallback);
						}
					}
				}
			};

			processFrames();
		},

		/**
		 * Sets up Canvas 2D-based frame capture (fallback method).
		 * @param {CameraResource} camera Camera resource
		 * @param {string} cameraId Camera identifier
		 * @param {Function} callback Callback function for frame data
		 * @param {number} context Context value to pass to callback
		 * @param {Function} deniedCallback Callback for permission denied
		 * @returns {void}
		 */
		setupCanvas2DCapture: function (camera, cameraId, callback, context, deniedCallback) {
			const [videoTrack] = camera.stream.getVideoTracks();
			const { width: _width, height: _height } = videoTrack.getSettings();

			if (!camera.canvas) {
				if (typeof OffscreenCanvas !== 'undefined') {
					camera.canvas = new OffscreenCanvas(_width, _height);
				} else {
					camera.canvas = document.createElement('canvas');
					camera.canvas.style.display = 'none';
					document.body.appendChild(camera.canvas);
				}
			}

			if (camera.canvas.width !== _width || camera.canvas.height !== _height) {
				camera.canvas.width = _width;
				camera.canvas.height = _height;
			}
			camera.canvasContext = camera.canvas.getContext('2d', { willReadFrequently: true });

			if (camera.animationFrameId) {
				cancelAnimationFrame(camera.animationFrameId);
			}

			const captureFrame = () => {
				const cameras = GodotCamera.ensureCamerasMap();
				const currentCamera = cameras.get(cameraId);
				if (!currentCamera) {
					return;
				}

				const { video, canvasContext, stream } = currentCamera;

				if (!stream || !stream.active) {
					GodotRuntime.print('Stream is not active, stopping');
					GodotCamera.api.stop(cameraId);
					return;
				}

				if (video.readyState === video.HAVE_ENOUGH_DATA) {
					try {
						canvasContext.drawImage(video, 0, 0, _width, _height);
						const imageData = canvasContext.getImageData(0, 0, _width, _height);
						const pixelData = imageData.data;

						const dataPtr = GodotRuntime.malloc(pixelData.length);
						GodotRuntime.heapCopy(HEAPU8, pixelData, dataPtr);

						const screenOrientation = screen?.orientation?.angle ?? window.orientation ?? 0;
						const facingMode = GodotCamera.getFacingMode(stream);

						GodotCamera.sendGetPixelDataCallback(
							callback,
							context,
							dataPtr,
							pixelData.length,
							_width,
							_height,
							screenOrientation,
							facingMode,
							null
						);

						GodotRuntime.free(dataPtr);
					} catch (error) {
						GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, 0, 0, error.message);

						if (error.name === 'SecurityError' || error.name === 'NotAllowedError') {
							GodotRuntime.print('Security error, stopping stream:', error);
							GodotCamera.api.stop(cameraId);
							deniedCallback(context);
						}
						return;
					}
				}

				currentCamera.animationFrameId = requestAnimationFrame(captureFrame);
			};

			camera.animationFrameId = requestAnimationFrame(captureFrame);
		},

		/**
		 * Sets up Web Worker-based Canvas 2D frame capture.
		 * Offloads drawImage and getImageData to a worker thread.
		 * @param {CameraResource} camera Camera resource
		 * @param {string} cameraId Camera identifier
		 * @param {Function} callback Callback function for frame data
		 * @param {number} context Context value to pass to callback
		 * @param {Function} deniedCallback Callback for permission denied
		 * @returns {void}
		 */
		setupCanvas2DWorkerCapture: function (camera, cameraId, callback, context, deniedCallback) {
			const workerUrl = this.getWorkerBlobUrl();
			if (!workerUrl) {
				GodotRuntime.print('Failed to create worker, falling back to main thread Canvas 2D');
				camera.useWorker = false;
				this.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
				return;
			}

			const [videoTrack] = camera.stream.getVideoTracks();
			const { width: _width, height: _height } = videoTrack.getSettings();

			try {
				camera.worker = new Worker(workerUrl);
			} catch (e) {
				GodotRuntime.print('Failed to create worker:', e.message);
				camera.useWorker = false;
				this.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
				return;
			}

			// Handle messages from worker.
			camera.worker.onmessage = (event) => {
				const { type } = event.data;

				switch (type) {
				case 'initialized':
					GodotRuntime.print('Camera worker initialized');
					break;

				case 'frameData': {
					const { pixelData, width, height, orientation, facingMode } = event.data;
					const dataPtr = GodotRuntime.malloc(pixelData.length);
					GodotRuntime.heapCopy(HEAPU8, pixelData, dataPtr);

					GodotCamera.sendGetPixelDataCallback(
						callback,
						context,
						dataPtr,
						pixelData.length,
						width,
						height,
						orientation,
						facingMode,
						null
					);

					GodotRuntime.free(dataPtr);
					break;
				}

				case 'error':
					GodotRuntime.print('Worker error:', event.data.message);
					// Fall back to main thread on error.
					camera.useWorker = false;
					if (camera.worker) {
						camera.worker.terminate();
						camera.worker = null;
					}
					GodotCamera.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
					break;

				case 'stopped':
					GodotRuntime.print('Camera worker stopped');
					break;

				default:
					break;
				}
			};

			camera.worker.onerror = (error) => {
				GodotRuntime.print('Worker error event:', error.message);
				camera.useWorker = false;
				if (camera.worker) {
					camera.worker.terminate();
					camera.worker = null;
				}
				GodotCamera.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
			};

			// Initialize the worker.
			camera.worker.postMessage({
				type: 'init',
				data: { width: _width, height: _height },
			});

			// Start the frame capture loop.
			const captureFrame = () => {
				const cameras = GodotCamera.ensureCamerasMap();
				const currentCamera = cameras.get(cameraId);
				if (!currentCamera || !currentCamera.useWorker || !currentCamera.worker) {
					return;
				}

				const { video, stream } = currentCamera;

				if (!stream || !stream.active) {
					GodotRuntime.print('Stream is not active, stopping');
					GodotCamera.api.stop(cameraId);
					return;
				}

				if (video.readyState === video.HAVE_ENOUGH_DATA) {
					try {
						// Create ImageBitmap from video (can be transferred to worker).
						createImageBitmap(video).then((imageBitmap) => {
							const cam = cameras.get(cameraId);
							if (!cam || !cam.useWorker || !cam.worker) {
								imageBitmap.close();
								return;
							}

							const screenOrientation = screen?.orientation?.angle ?? window.orientation ?? 0;
							const facingMode = GodotCamera.getFacingMode(cam.stream);

							cam.worker.postMessage(
								{
									type: 'frame',
									data: {
										imageBitmap,
										orientation: screenOrientation,
										facingMode,
									},
								},
								[imageBitmap]
							);
						}).catch((e) => {
							GodotRuntime.print('createImageBitmap error:', e.message);
						});
					} catch (error) {
						GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, 0, 0, error.message);

						if (error.name === 'SecurityError' || error.name === 'NotAllowedError') {
							GodotRuntime.print('Security error, stopping stream:', error);
							GodotCamera.api.stop(cameraId);
							deniedCallback(context);
						}
						return;
					}
				}

				currentCamera.animationFrameId = requestAnimationFrame(captureFrame);
			};

			camera.animationFrameId = requestAnimationFrame(captureFrame);
		},

		/**
		 * Cleans up resources for a specific camera.
		 * @param {CameraResource} camera Camera resource to cleanup
		 * @returns {void}
		 */
		cleanupCamera: function (camera, cameraId) {
			// Clean up Web Worker resources.
			if (camera.worker) {
				camera.worker.postMessage({ type: 'stop' });
				camera.worker.terminate();
				camera.worker = null;
			}

			// Clean up WebCodecs resources.
			if (camera.frameReader) {
				camera.frameReader.cancel().catch(() => {});
				camera.frameReader = null;
			}
			if (camera.trackProcessor) {
				camera.trackProcessor = null;
			}

			if (camera.animationFrameId) {
				cancelAnimationFrame(camera.animationFrameId);
			}

			if (camera.stream) {
				camera.stream.getTracks().forEach((track) => track.stop());
			}

			if (camera.video && camera.video.parentNode) {
				camera.video.parentNode.removeChild(camera.video);
			}

			if (camera.canvas && camera.canvas instanceof HTMLCanvasElement && camera.canvas.parentNode) {
				camera.canvas.parentNode.removeChild(camera.canvas);
			}

			if (camera.permissionListener && camera.permissionStatus) {
				camera.permissionStatus.removeEventListener('change', camera.permissionListener);
				camera.permissionListener = null;
				camera.permissionStatus = null;
			}

			// Null out references to help GC.
			camera.animationFrameId = null;
			camera.canvasContext = null;
			camera.stream = null;
			camera.video = null;
			camera.canvas = null;
			camera.useWebCodecsWorker = false;
			camera.useWebCodecs = false;
			camera.useWorker = false;
		},

		api: {
			/**
			 * Gets list of available cameras.
			 * Calls callback with JSON containing array of camera info.
			 * @param {number} context Context value to pass to callback
			 * @param {number} callbackPtr1 Pointer to callback function
			 * @param {number} callbackPtr2 Pointer to callback function
			 * @returns {Promise<void>}
			 */
			getCameras: async function (context, callbackPtr1, callbackPtr2) {
				const callback = GodotRuntime.get_func(callbackPtr2);
				const result = { error: null, cameras: null };

				try {
					// Request camera access permission first.
					const initialStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
					initialStream.getTracks().forEach((track) => track.stop());

					const devices = await navigator.mediaDevices.enumerateDevices();
					const videoDevices = devices.filter((device) => device.kind === 'videoinput');

					// Get capabilities for each camera device.
					const cameraPromises = videoDevices.map(async (device, index) => {
						let formats = [];
						try {
							const stream = await navigator.mediaDevices.getUserMedia({
								video: { deviceId: { exact: device.deviceId } },
								audio: false,
							});
							const [videoTrack] = stream.getVideoTracks();
							const capabilities = videoTrack?.getCapabilities() || GodotCamera.defaultMinimumCapabilities;
							formats = GodotCamera.getSupportedFormats(capabilities);
							stream.getTracks().forEach((track) => track.stop());
						} catch (e) {
							// If we can't get capabilities, use default formats.
							formats = GodotCamera.getSupportedFormats(GodotCamera.defaultMinimumCapabilities);
						}

						return {
							index,
							id: device.deviceId,
							label: device.label || `Camera ${index}`,
							formats,
						};
					});

					result.cameras = await Promise.all(cameraPromises);
				} catch (error) {
					result.error = error.message;
				}

				GodotCamera.sendCamerasCallbackResult(callback, callbackPtr1, context, result);
			},

			/**
			 * Starts capturing pixel data from camera.
			 * Continuously calls callback with pixel data.
			 * @param {number} context Context value to pass to callback
			 * @param {string|null} deviceId Camera device ID
			 * @param {number} width Desired capture width
			 * @param {number} height Desired capture height
			 * @param {number} callbackPtr Pointer to callback function
			 * @param {number} deniedCallbackPtr Pointer to callback function
			 * @returns {Promise<void>}
			 */
			getPixelData: async function (context, deviceId, width, height, callbackPtr, deniedCallbackPtr) {
				const callback = GodotRuntime.get_func(callbackPtr);
				const deniedCallback = GodotRuntime.get_func(deniedCallbackPtr);
				const cameraId = deviceId || 'default';

				try {
					const camerasMap = GodotCamera.ensureCamerasMap();
					let camera = camerasMap.get(cameraId);
					if (!camera) {
						camera = {
							video: null,
							canvas: null,
							canvasContext: null,
							stream: null,
							animationFrameId: null,
							permissionListener: null,
							permissionStatus: null,
							trackProcessor: null,
							frameReader: null,
							worker: null,
							useWebCodecsWorker: false,
							useWebCodecs: false,
							useWorker: false,
						};
						camerasMap.set(cameraId, camera);
					}

					if (!camera.stream) {
						// Create video element (needed for Canvas 2D fallback).
						camera.video = document.createElement('video');
						camera.video.style.display = 'none';
						camera.video.autoplay = true;
						camera.video.playsInline = true;
						document.body.appendChild(camera.video);

						const constraints = {
							video: {
								deviceId: deviceId ? { exact: deviceId } : undefined,
								width: { ideal: width },
								height: { ideal: height },
							},
						};
						// eslint-disable-next-line require-atomic-updates
						camera.stream = await navigator.mediaDevices.getUserMedia(constraints);

						const [videoTrack] = camera.stream.getVideoTracks();
						videoTrack.addEventListener('ended', () => {
							GodotRuntime.print('Camera track ended, stopping stream');
							GodotCamera.api.stop(cameraId);
						});

						if (navigator.permissions && navigator.permissions.query) {
							try {
								const permissionStatus = await navigator.permissions.query({ name: 'camera' });
								// eslint-disable-next-line require-atomic-updates
								camera.permissionStatus = permissionStatus;
								camera.permissionListener = () => {
									if (permissionStatus.state === 'denied') {
										GodotRuntime.print('Camera permission denied, stopping stream');
										if (camera.permissionListener) {
											permissionStatus.removeEventListener('change', camera.permissionListener);
										}
										GodotCamera.api.stop(cameraId);
										deniedCallback(context);
									}
								};
								permissionStatus.addEventListener('change', camera.permissionListener);
							} catch (e) {
								// Some browsers don't support 'camera' permission query.
								// This is not critical - we can still use the camera.
								GodotRuntime.print('Camera permission query not supported:', e.message);
							}
						}

						camera.video.srcObject = camera.stream;
						await camera.video.play();

						// Choose capture method (ordered by efficiency):
						// 1. WebCodecs + Worker: VideoFrame transferred to Worker, copyTo() in Worker
						// 2. Worker Canvas 2D: ImageBitmap transferred to Worker, drawImage + getImageData in Worker
						// 3. WebCodecs (main thread): copyTo() on main thread
						// 4. Canvas 2D (main thread): drawImage + getImageData on main thread
						if (GodotCamera.isWebCodecsSupported() && GodotCamera.isWorkerSupported()) {
							// eslint-disable-next-line require-atomic-updates
							camera.useWebCodecsWorker = true;
							GodotRuntime.print('Using WebCodecs + Worker for camera capture');
							GodotCamera.setupWebCodecsWorkerCapture(camera, cameraId, callback, context, deniedCallback);
						} else if (GodotCamera.isWorkerSupported()) {
							// eslint-disable-next-line require-atomic-updates
							camera.useWorker = true;
							GodotRuntime.print('Using Worker Canvas 2D for camera capture');
							GodotCamera.setupCanvas2DWorkerCapture(camera, cameraId, callback, context, deniedCallback);
						} else if (GodotCamera.isWebCodecsSupported()) {
							// eslint-disable-next-line require-atomic-updates
							camera.useWebCodecs = true;
							GodotRuntime.print('Using WebCodecs (main thread) for camera capture');
							GodotCamera.setupWebCodecsCapture(camera, cameraId, callback, context, deniedCallback);
						} else {
							GodotRuntime.print('Using Canvas 2D (main thread) for camera capture');
							GodotCamera.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
						}
					}
				} catch (error) {
					GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, 0, 0, error.message);
					if (error && (error.name === 'SecurityError' || error.name === 'NotAllowedError')) {
						deniedCallback(context);
					}
				}
			},

			/**
			 * Stops camera stream(s).
			 * @param {string|null} deviceId Device ID to stop, or null to stop all
			 * @returns {void}
			 */
			stop: function (deviceId) {
				const cameras = GodotCamera.ensureCamerasMap();

				if (deviceId && cameras.has(deviceId)) {
					const camera = cameras.get(deviceId);
					if (camera) {
						GodotCamera.cleanupCamera(camera, deviceId);
					}
					cameras.delete(deviceId);
				} else {
					cameras.forEach((camera, id) => {
						if (camera) {
							GodotCamera.cleanupCamera(camera, id);
						}
					});
					cameras.clear();
				}
			},
		},
	},

	/**
	 * Native binding for getting list of cameras.
	 * @param {number} context Context value to pass to callback
	 * @param {number} callbackPtr1 Pointer to callback function
	 * @param {number} callbackPtr2 Pointer to callback function
	 * @returns {Promise<void>}
	 */
	godot_js_camera_get_cameras: function (context, callbackPtr1, callbackPtr2) {
		return GodotCamera.api.getCameras(context, callbackPtr1, callbackPtr2);
	},

	/**
	 * Native binding for getting pixel data from camera.
	 * @param {number} context Context value to pass to callback
	 * @param {number} deviceIdPtr Pointer to device ID string
	 * @param {number} width Desired capture width
	 * @param {number} height Desired capture height
	 * @param {number} callbackPtr Pointer to callback function
	 * @param {number} deniedCallbackPtr Pointer to denied callback function
	 * @returns {*}
	 */
	godot_js_camera_get_pixel_data: function (context, deviceIdPtr, width, height, callbackPtr, deniedCallbackPtr) {
		const deviceId = deviceIdPtr ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		return GodotCamera.api.getPixelData(context, deviceId, width, height, callbackPtr, deniedCallbackPtr);
	},

	/**
	 * Native binding for stopping camera stream.
	 * @param {number} deviceIdPtr Pointer to device ID string
	 * @returns {void}
	 */
	godot_js_camera_stop_stream: function (deviceIdPtr) {
		const deviceId = deviceIdPtr ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		GodotCamera.api.stop(deviceId);
	},
};

autoAddDeps(GodotCamera, '$GodotCamera');
mergeInto(LibraryManager.library, GodotCamera);
