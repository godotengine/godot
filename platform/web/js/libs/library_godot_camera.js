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
 * }} CameraInfo
 *
 * @typedef {{
 *   video: HTMLVideoElement|null
 *   canvas: HTMLCanvasElement|null
 *   canvasContext: CanvasRenderingContext2D|null
 *   stream: MediaStream|null
 *   animationFrameId: number|null
 *   permissionListener: Function|null
 *   permissionStatus: PermissionStatus|null
 *   trackProcessor: MediaStreamTrackProcessor|null
 *   frameReader: ReadableStreamDefaultReader|null
 *   useWebCodecs: boolean
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
		 * Serialized operation queue. All camera start/stop/switch operations
		 * are enqueued here so they never interleave in the JS event loop.
		 * @type {Promise<void>}
		 */
		opQueue: Promise.resolve(),

		/**
		 * Cancellation generation for each context pointer (C++ CameraFeedWeb*).
		 * An operation captures the current generation when it starts and is
		 * canceled only when abort() advances that generation. Future operations
		 * using the same context are therefore not canceled by an earlier stop.
		 * @type {Map<number, number>}
		 */
		contextGenerations: new Map(),

		/**
		 * Pixel format codes passed to the C callback.
		 * Must match the values used in modules/camera/camera_web.cpp.
		 */
		FORMAT_CODE_RGBA: 0,
		FORMAT_CODE_NV12: 1,
		FORMAT_CODE_I420: 2,

		/**
		 * Builds the VideoFrame.copyTo options for a given source format.
		 * Preserves 4:2:0 YUV sources (NV12, I420, I420A) as planar data so the
		 * C++ side can feed them to set_ycbcr_images without a color conversion.
		 * Other formats fall through to RGBA and rely on the browser to convert.
		 * @param {string} srcFormat VideoFrame.format ('NV12', 'I420', 'I420A', 'RGBA', ...)
		 * @param {{x:number,y:number,width:number,height:number}} rect
		 * @returns {{totalSize:number,options:Object,formatCode:number}}
		 */
		getCopyToOptions: function (srcFormat, rect) {
			const width = rect.width;
			const height = rect.height;
			// Round chroma dimensions up for odd sizes.
			const chromaWidth = (width + 1) >> 1;
			const chromaHeight = (height + 1) >> 1;
			const ySize = width * height;
			const chromaSize = chromaWidth * chromaHeight;

			if (srcFormat === 'NV12') {
				// Chrome rejects an explicit YUV `format`; omit it to copy the
				// native NV12 planes (Y + interleaved Cb/Cr).
				const uvStride = chromaWidth * 2;
				return {
					totalSize: ySize + uvStride * chromaHeight,
					options: {
						rect: rect,
						layout: [
							{ offset: 0, stride: width },
							{ offset: ySize, stride: uvStride },
						],
					},
					formatCode: GodotCamera.FORMAT_CODE_NV12,
				};
			}

			if (srcFormat === 'I420' || srcFormat === 'I420A') {
				// Y + U/V planes. I420A's trailing alpha plane is included for
				// copyTo (one layout entry per plane) but ignored downstream.
				const layout = [
					{ offset: 0, stride: width },
					{ offset: ySize, stride: chromaWidth },
					{ offset: ySize + chromaSize, stride: chromaWidth },
				];
				let totalSize = ySize + chromaSize * 2;
				if (srcFormat === 'I420A') {
					layout.push({ offset: totalSize, stride: width });
					totalSize += ySize;
				}
				return {
					totalSize: totalSize,
					options: {
						rect: rect,
						layout: layout,
					},
					formatCode: GodotCamera.FORMAT_CODE_I420,
				};
			}

			return {
				totalSize: width * height * 4,
				options: {
					rect: rect,
					layout: [{ offset: 0, stride: width * 4 }],
					format: 'RGBA',
				},
				formatCode: GodotCamera.FORMAT_CODE_RGBA,
			};
		},

		/**
		 * Gets the visible pixel rectangle to copy from a VideoFrame.
		 * @param {VideoFrame} videoFrame
		 * @returns {{x:number,y:number,width:number,height:number}}
		 */
		getVisibleFrameRect: function (videoFrame) {
			const rect = videoFrame.visibleRect;
			if (rect) {
				return {
					x: rect.x,
					y: rect.y,
					width: rect.width,
					height: rect.height,
				};
			}
			return {
				x: 0,
				y: 0,
				width: videoFrame.displayWidth,
				height: videoFrame.displayHeight,
			};
		},

		/**
		 * Cached Wasm heap pointer for frame data copy.
		 * Avoids per-frame malloc/free overhead and fragmentation.
		 * @type {number}
		 */
		_cachedDataPtr: 0,

		/**
		 * Size of the cached Wasm heap buffer.
		 * @type {number}
		 */
		_cachedDataSize: 0,

		/**
		 * Copies pixel data into the cached Wasm heap buffer, reallocating if the size changed.
		 * @param {Uint8Array} p_data - Pixel data to copy.
		 */
		_heapCopyPixelData(p_data) {
			if (GodotCamera._cachedDataSize !== p_data.length) {
				if (GodotCamera._cachedDataPtr) {
					GodotRuntime.free(GodotCamera._cachedDataPtr);
				}
				GodotCamera._cachedDataPtr = GodotRuntime.malloc(p_data.length);
				GodotCamera._cachedDataSize = p_data.length;
			}
			GodotRuntime.heapCopy(HEAPU8, p_data, GodotCamera._cachedDataPtr);
		},

		/**
		 * Common native camera resolutions, grouped by aspect ratio, checked
		 * against active camera capabilities.
		 */
		commonResolutions: [
			// 16:9
			{ width: 640, height: 360 }, // nHD (16:9)
			{ width: 848, height: 480 }, // FWVGA (16:9)
			{ width: 1280, height: 720 }, // HD 720p (16:9)
			{ width: 1920, height: 1080 }, // Full HD 1080p (16:9)
			{ width: 3840, height: 2160 }, // 4K UHD 2160p (16:9)
			// 16:10
			{ width: 1280, height: 800 }, // WXGA (16:10)
			{ width: 1920, height: 1200 }, // WUXGA (16:10)
			// 4:3
			{ width: 320, height: 240 }, // QVGA (4:3)
			{ width: 640, height: 480 }, // VGA (4:3)
			{ width: 800, height: 600 }, // SVGA (4:3)
			{ width: 1280, height: 960 }, // 960p (4:3)
			{ width: 1600, height: 1200 }, // UXGA (4:3)
			{ width: 2048, height: 1536 }, // QXGA (4:3)
			// max size
			{ width: 99999, height: 99999 },
		],

		/**
		 * Gets supported formats based on active track capabilities.
		 * @param {Object} capabilities MediaTrackCapabilities object
		 * @returns {Array<{width:number,height:number,frameRate:number}>} Supported resolutions with frame rate
		 */
		getSupportedFormats: function (capabilities) {
			const widthRange = capabilities.width || {};
			const heightRange = capabilities.height || {};
			const frameRateRange = capabilities.frameRate || {};
			const maxWidth = widthRange.max || 0;
			const maxHeight = heightRange.max || 0;
			const minWidth = widthRange.min || 1;
			const minHeight = heightRange.min || 1;
			const maxFrameRate = Math.round(frameRateRange.max || 0);

			if (maxWidth <= 0 || maxHeight <= 0) {
				return [];
			}

			return GodotCamera.commonResolutions
				.filter((res) => res.width >= minWidth
					&& res.width <= maxWidth
					&& res.height >= minHeight
					&& res.height <= maxHeight)
				.map((res) => ({
					width: res.width,
					height: res.height,
					frameRate: maxFrameRate,
				}))
				.sort((a, b) => {
					const areaA = a.width * a.height;
					const areaB = b.width * b.height;
					if (areaA !== areaB) {
						return areaA - areaB;
					}
					return a.width - b.width;
				});
		},

		/**
		 * Checks if WebCodecs API is supported for camera capture.
		 * @returns {boolean} True if MediaStreamTrackProcessor and VideoFrame are available
		 */
		isWebCodecsSupported: function () {
			return 'MediaStreamTrackProcessor' in window && 'VideoFrame' in window;
		},

		/**
		 * Awaits a promise with a timeout.
		 * @param {Promise<*>} promise Promise to await
		 * @param {number} timeoutMs Timeout in milliseconds
		 * @param {string} timeoutMessage Error message on timeout
		 * @returns {Promise<*>}
		 */
		waitWithTimeout: function (promise, timeoutMs, timeoutMessage) {
			let timeoutId = null;
			const timeoutPromise = new Promise((_, reject) => {
				timeoutId = setTimeout(() => {
					reject(new Error(timeoutMessage));
				}, timeoutMs);
			});
			return Promise.race([promise, timeoutPromise]).finally(() => {
				if (timeoutId !== null) {
					clearTimeout(timeoutId);
				}
			});
		},

		/**
		 * Opens a camera stream with a timeout and stops late streams after timeout.
		 * @param {Object} constraints getUserMedia constraints
		 * @param {number} timeoutMs Timeout in milliseconds
		 * @param {string} timeoutMessage Error message on timeout
		 * @returns {Promise<MediaStream>}
		 */
		getUserMediaWithTimeout: function (constraints, timeoutMs, timeoutMessage) {
			const streamPromise = navigator.mediaDevices.getUserMedia(constraints);
			return GodotCamera.waitWithTimeout(streamPromise, timeoutMs, timeoutMessage).catch((error) => {
				streamPromise.then((stream) => {
					GodotCamera.stopStream(stream);
				}, () => {});
				throw error;
			});
		},

		/**
		 * Opens a camera with retry on transient "device busy" errors, used when
		 * switching cameras: opening one camera right after stopping another can
		 * fail while the previous device is still being released (Android).
		 *
		 * Rationale (from Gecko/libwebrtc), which dictates the error handling:
		 * - A failed open is bounded, not infinite. libwebrtc's CameraCapturer
		 *   retries the open MAX_OPEN_CAMERA_ATTEMPTS=3 times, OPEN_CAMERA_DELAY_MS
		 *   =500ms apart, each guarded by OPEN_CAMERA_TIMEOUT=10000ms, then reports
		 *   failure. Gecko surfaces that to JS as an AbortError ("Starting video
		 *   failed", MediaManager.cpp). Worst case is ~11s.
		 * - So timeoutMs must exceed ~11s: getUserMedia then rejects cleanly with
		 *   that AbortError. A shorter race timeout would fire first and leave the
		 *   underlying request pending on Gecko's single, serialized VideoCapture
		 *   thread; retrying then just queues a second open behind the first and
		 *   makes the contention worse.
		 * - Hence we retry only on cleanly-thrown busy errors (AbortError /
		 *   NotReadableError), never on the race timeout.
		 * @param {Object} constraints getUserMedia constraints
		 * @param {number} attempts Maximum number of attempts
		 * @param {number} timeoutMs Per-attempt timeout (must exceed ~11s; see above)
		 * @param {string} timeoutMessage Error message on timeout
		 * @returns {Promise<MediaStream>}
		 */
		getUserMediaWithRetry: async function (constraints, attempts, timeoutMs, timeoutMessage) {
			let lastError = null;
			for (let i = 0; i < attempts; i++) {
				try {
					// eslint-disable-next-line no-await-in-loop
					return await GodotCamera.getUserMediaWithTimeout(constraints, timeoutMs, timeoutMessage);
				} catch (error) {
					lastError = error;
					const name = error ? error.name : null;
					if ((name !== 'NotReadableError' && name !== 'AbortError') || i === attempts - 1) {
						throw error;
					}
					// Back off to give the previous camera time to finish releasing.
					// eslint-disable-next-line no-await-in-loop
					await new Promise((resolve) => {
						setTimeout(resolve, 300 * (i + 1));
					});
				}
			}
			throw lastError;
		},

		/**
		 * Enqueues an async operation so that start/stop/switch calls never
		 * interleave in the JS event loop.
		 * @param {Function} fn Async function to run exclusively
		 * @returns {Promise<*>}
		 */
		runExclusive: function (fn) {
			if (!GodotCamera.opQueue || typeof GodotCamera.opQueue.then !== 'function') {
				GodotCamera.opQueue = Promise.resolve();
			}
			const next = GodotCamera.opQueue.then(() => fn());
			// Prevent a rejection from permanently blocking the queue.
			GodotCamera.opQueue = next.catch(() => {});
			return next;
		},

		/**
		 * Returns the current cancellation generation for a context.
		 * @param {number} context C++ CameraFeedWeb* context
		 * @returns {number} Current generation
		 */
		getContextGeneration: function (context) {
			return GodotCamera.contextGenerations.get(context) || 0;
		},

		/**
		 * Checks whether an operation was canceled after it started.
		 * @param {number} context C++ CameraFeedWeb* context
		 * @param {number} generation Generation captured at operation start
		 * @returns {boolean} True when the operation is stale
		 */
		isContextCanceled: function (context, generation) {
			return GodotCamera.getContextGeneration(context) !== generation;
		},

		/**
		 * Stops all tracks in a stream.
		 * @param {MediaStream|null} stream Stream to stop
		 * @returns {boolean} True if a stream was stopped
		 */
		stopStream: function (stream) {
			if (!stream) {
				return false;
			}
			stream.getTracks().forEach((track) => track.stop());
			return true;
		},

		/**
		 * Tears down resources for one or all cameras without going through the
		 * op-queue.  Called internally from within runExclusive() tasks where
		 * queuing again would deadlock, and from cleanup() at shutdown.
		 * @param {string|undefined} deviceId Device to stop, or undefined for all
		 */
		_stopInternal: function (deviceId) {
			const cameras = GodotCamera.cameras;
			if (deviceId && cameras.has(deviceId)) {
				const camera = cameras.get(deviceId);
				if (camera) {
					GodotCamera.cleanupCamera(camera);
				}
				cameras.delete(deviceId);
			} else if (!deviceId) {
				cameras.forEach((camera) => {
					if (camera) {
						GodotCamera.cleanupCamera(camera);
					}
				});
				cameras.clear();
			}
		},

		/**
		 * Cleanup all camera resources.
		 * @returns {void}
		 */
		cleanup: function () {
			GodotCamera._stopInternal();

			// Free cached Wasm heap buffer.
			if (this._cachedDataPtr) {
				GodotRuntime.free(this._cachedDataPtr);
				this._cachedDataPtr = 0;
				this._cachedDataSize = 0;
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
		 * Sends active camera formats to the callback function.
		 * @param {Function} callback Callback function pointer
		 * @param {number} context Context value to pass to callback
		 * @param {Object} result Result object to stringify
		 * @returns {void}
		 */
		sendFormatsCallbackResult: function (callback, context, result) {
			const jsonStr = JSON.stringify(result);
			const strPtr = GodotRuntime.allocString(jsonStr);
			callback(context, strPtr);
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
		 * @param {number} formatCode Pixel format code (FORMAT_CODE_RGBA, FORMAT_CODE_NV12, ...)
		 * @param {number} facingMode Camera facing mode (0=unknown, 1=user/front, 2=environment/back)
		 * @param {string|null} errorMsg Error message if any
		 * @returns {void}
		 */
		sendGetPixelDataCallback: function (callback, context, dataPtr, dataLen, width, height, formatCode, facingMode, errorMsg) {
			const errorMsgPtr = errorMsg ? GodotRuntime.allocString(errorMsg) : 0;
			callback(context, dataPtr, dataLen, width, height, formatCode, facingMode, errorMsgPtr);
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
		 * Gets the active format metadata from an active stream track.
		 * @param {MediaStreamTrack} videoTrack Active video track
		 * @param {number} facingMode Numeric facing mode
		 * @returns {{formats:Array<{width:number,height:number,frameRate:number}>,current:{width:number,height:number,frameRate:number,facingMode:number}}}
		 */
		getActiveFormats: function (videoTrack, facingMode) {
			const settings = videoTrack.getSettings ? videoTrack.getSettings() : {};
			const capabilities = videoTrack.getCapabilities ? videoTrack.getCapabilities() : {};
			const current = {
				width: settings.width || 0,
				height: settings.height || 0,
				frameRate: Math.round(settings.frameRate || 0),
				facingMode,
			};
			const formats = GodotCamera.getSupportedFormats(capabilities);
			const hasCurrentFormat = formats.some((format) => (format.width === current.width && format.height === current.height)
				|| (format.width === current.height && format.height === current.width));
			if (!hasCurrentFormat && current.width > 0 && current.height > 0) {
				formats.push({
					width: current.width,
					height: current.height,
					frameRate: current.frameRate,
				});
			}

			return {
				formats,
				current,
			};
		},

		/**
		 * Checks whether an async startup path still owns the active camera resource.
		 * @param {string} cameraId Camera identifier
		 * @param {CameraResource} camera Camera resource captured before awaiting
		 * @returns {boolean} True if the resource is still current
		 */
		isCurrentCamera: function (cameraId, camera) {
			return GodotCamera.cameras.get(cameraId) === camera;
		},

		/**
		 * Creates an empty camera resource.
		 * @returns {CameraResource} Camera resource
		 */
		createCameraResource: function () {
			return {
				video: null,
				canvas: null,
				canvasContext: null,
				stream: null,
				animationFrameId: null,
				permissionListener: null,
				permissionStatus: null,
				trackProcessor: null,
				frameReader: null,
				useWebCodecs: false,
				facingMode: 0,
			};
		},

		/**
		 * Creates the hidden video element used by the Canvas 2D fallback.
		 * @returns {HTMLVideoElement} Video element
		 */
		createVideoElement: function () {
			const video = document.createElement('video');
			video.style.display = 'none';
			video.autoplay = true;
			video.playsInline = true;
			video.muted = true;
			document.body.appendChild(video);
			return video;
		},

		/**
		 * Creates getUserMedia constraints for a camera request.
		 * @param {string|null} deviceId Camera device ID
		 * @param {number} width Desired width
		 * @param {number} height Desired height
		 * @returns {MediaStreamConstraints} Media constraints
		 */
		createConstraints: function (deviceId, width, height) {
			return {
				video: {
					deviceId: deviceId ? { exact: deviceId } : undefined,
					width: width > 0 ? { ideal: width } : undefined,
					height: height > 0 ? { ideal: height } : undefined,
					// Prefer a native sensor mode over a cropped/downscaled frame.
					resizeMode: { ideal: 'none' },
				},
			};
		},

		/**
		 * Discards a camera resource created by a failed or canceled start.
		 * @param {string} cameraId Camera identifier
		 * @param {CameraResource} camera Camera resource
		 */
		discardCamera: function (cameraId, camera) {
			GodotCamera.cleanupCamera(camera);
			if (GodotCamera.isCurrentCamera(cameraId, camera)) {
				GodotCamera.cameras.delete(cameraId);
			}
		},

		/**
		 * Opens a camera and starts frame capture for both regular activation and
		 * feed switching.
		 * @param {Object} request Camera start request
		 * @returns {Promise<CameraResource|null>} Camera resource, or null if canceled
		 */
		startCamera: async function (request) {
			const {
				context,
				operationGeneration,
				deviceId,
				width,
				height,
				callback,
				deniedCallback,
				formatsCallback,
				timeoutMessage,
				playTimeoutMessage,
				attempts = 2,
			} = request;
			const cameraId = deviceId || 'default';
			const cameras = GodotCamera.cameras;
			let camera = cameras.get(cameraId);
			if (!camera) {
				camera = GodotCamera.createCameraResource();
				cameras.set(cameraId, camera);
			}

			if (camera.stream) {
				return camera;
			}

			try {
				camera.video = GodotCamera.createVideoElement();
				const stream = await GodotCamera.getUserMediaWithRetry(
					GodotCamera.createConstraints(deviceId, width, height),
					attempts,
					12000,
					timeoutMessage
				);
				if (GodotCamera.isContextCanceled(context, operationGeneration)
					|| !GodotCamera.isCurrentCamera(cameraId, camera)) {
					GodotCamera.stopStream(stream);
					GodotCamera.discardCamera(cameraId, camera);
					return null;
				}
				camera.stream = stream;

				const [videoTrack] = stream.getVideoTracks();
				videoTrack.addEventListener('ended', () => {
					GodotCamera.api.stop(cameraId);
				});

				if (navigator.permissions && navigator.permissions.query) {
					try {
						const permissionStatus = await navigator.permissions.query({ name: 'camera' });
						camera.permissionStatus = permissionStatus;
						camera.permissionListener = () => {
							if (permissionStatus.state === 'denied') {
								GodotRuntime.print('Camera permission denied, stopping stream');
								GodotCamera.api.stop(cameraId);
								deniedCallback(context);
							}
						};
						permissionStatus.addEventListener('change', camera.permissionListener);
					} catch (e) {
						// Camera permission queries are not supported by all browsers.
					}
				}

				if (GodotCamera.isContextCanceled(context, operationGeneration)
					|| !GodotCamera.isCurrentCamera(cameraId, camera)) {
					GodotCamera.discardCamera(cameraId, camera);
					return null;
				}

				camera.video.srcObject = stream;
				await GodotCamera.waitWithTimeout(
					camera.video.play(),
					3000,
					playTimeoutMessage
				);
				if (GodotCamera.isContextCanceled(context, operationGeneration)
					|| !GodotCamera.isCurrentCamera(cameraId, camera)) {
					GodotCamera.discardCamera(cameraId, camera);
					return null;
				}

				camera.facingMode = GodotCamera.getFacingMode(stream);
				GodotCamera.sendFormatsCallbackResult(
					formatsCallback,
					context,
					GodotCamera.getActiveFormats(videoTrack, camera.facingMode)
				);

				if (GodotCamera.isWebCodecsSupported()) {
					camera.useWebCodecs = true;
					GodotCamera.setupWebCodecsCapture(camera, cameraId, callback, context, deniedCallback);
				} else {
					GodotCamera.setupCanvas2DCapture(camera, cameraId, callback, context, deniedCallback);
				}
				return camera;
			} catch (error) {
				GodotCamera.discardCamera(cameraId, camera);
				throw error;
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
				const cameras = GodotCamera.cameras;
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
							const frameRect = GodotCamera.getVisibleFrameRect(videoFrame);
							const copyTo = GodotCamera.getCopyToOptions(videoFrame.format, frameRect);
							const pixelBuffer = new Uint8Array(copyTo.totalSize);

							// eslint-disable-next-line no-await-in-loop
							await videoFrame.copyTo(pixelBuffer, copyTo.options);

							GodotCamera._heapCopyPixelData(pixelBuffer);

							GodotCamera.sendGetPixelDataCallback(
								callback,
								context,
								GodotCamera._cachedDataPtr,
								pixelBuffer.length,
								frameRect.width,
								frameRect.height,
								copyTo.formatCode,
								currentCamera.facingMode,
								null
							);
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

						GodotCamera.setupCanvas2DCapture(currentCamera, cameraId, callback, context, deniedCallback);
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
			if (camera.animationFrameId) {
				cancelAnimationFrame(camera.animationFrameId);
			}

			const captureFrame = () => {
				const cameras = GodotCamera.cameras;
				const currentCamera = cameras.get(cameraId);
				if (!currentCamera) {
					return;
				}

				const { video, stream } = currentCamera;

				if (!stream || !stream.active) {
					GodotRuntime.print('Stream is not active, stopping');
					GodotCamera.api.stop(cameraId);
					return;
				}

				if (video.readyState === video.HAVE_ENOUGH_DATA) {
					const _width = video.videoWidth;
					const _height = video.videoHeight;

					if (!currentCamera.canvas) {
						currentCamera.canvas = document.createElement('canvas');
						currentCamera.canvas.style.display = 'none';
						document.body.appendChild(currentCamera.canvas);
					}

					if (currentCamera.canvas.width !== _width || currentCamera.canvas.height !== _height) {
						currentCamera.canvas.width = _width;
						currentCamera.canvas.height = _height;
						currentCamera.canvasContext = null;
					}

					if (!currentCamera.canvasContext) {
						currentCamera.canvasContext = currentCamera.canvas.getContext('2d', { willReadFrequently: true });
					}

					try {
						currentCamera.canvasContext.drawImage(video, 0, 0, _width, _height);
						const imageData = currentCamera.canvasContext.getImageData(0, 0, _width, _height);
						const pixelData = imageData.data;

						GodotCamera._heapCopyPixelData(pixelData);

						GodotCamera.sendGetPixelDataCallback(
							callback,
							context,
							GodotCamera._cachedDataPtr,
							pixelData.length,
							_width,
							_height,
							GodotCamera.FORMAT_CODE_RGBA,
							currentCamera.facingMode,
							null
						);
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
		cleanupCamera: function (camera) {
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

			GodotCamera.stopStream(camera.stream);

			if (camera.video && camera.video.parentNode) {
				camera.video.pause();
				camera.video.srcObject = null;
				camera.video.load();
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
			camera.useWebCodecs = false;
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
					GodotCamera.stopStream(initialStream);

					const devices = await navigator.mediaDevices.enumerateDevices();
					const videoDevices = devices.filter((device) => device.kind === 'videoinput');

					result.cameras = videoDevices.map((device, index) => ({
						id: device.deviceId,
						label: device.label || `Camera ${index}`,
					}));
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
			 * @param {number} formatsCallbackPtr Pointer to formats callback function
			 * @returns {Promise<void>}
			 */
			getPixelData: async function (context, deviceId, width, height, callbackPtr, deniedCallbackPtr, formatsCallbackPtr) {
				const callback = GodotRuntime.get_func(callbackPtr);
				const deniedCallback = GodotRuntime.get_func(deniedCallbackPtr);
				const formatsCallback = GodotRuntime.get_func(formatsCallbackPtr);
				const operationGeneration = GodotCamera.getContextGeneration(context);

				await GodotCamera.runExclusive(async () => {
					try {
						if (GodotCamera.isContextCanceled(context, operationGeneration)) {
							return;
						}
						await GodotCamera.startCamera({
							context,
							operationGeneration,
							deviceId,
							width,
							height,
							callback,
							deniedCallback,
							formatsCallback,
							timeoutMessage: 'Camera start timed out while waiting for getUserMedia().',
							playTimeoutMessage: 'Camera start timed out while waiting for video.play().',
						});
					} catch (error) {
						if (GodotCamera.isContextCanceled(context, operationGeneration)) {
							return;
						}
						GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, 0, 0, error.message);
						if (error && (error.name === 'SecurityError' || error.name === 'NotAllowedError')) {
							deniedCallback(context);
						}
					}
				}); // end runExclusive
			},

			/**
			 * Stops camera stream(s).
			 * @param {string|null} deviceId Device ID to stop, or null to stop all
			 * @returns {void}
			 */
			stop: function (deviceId) {
				GodotCamera.runExclusive(() => {
					GodotCamera._stopInternal(deviceId || undefined);
				});
			},

			/**
			 * Cancels any in-flight open for the given context pointer.
			 * @param {number} context C++ CameraFeedWeb* context
			 */
			abort: function (context) {
				const generation = GodotCamera.getContextGeneration(context);
				GodotCamera.contextGenerations.set(context, generation + 1);
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
	 * @param {number} formatsCallbackPtr Pointer to formats callback function
	 * @returns {*}
	 */
	godot_js_camera_get_pixel_data: function (context, deviceIdPtr, width, height, callbackPtr, deniedCallbackPtr, formatsCallbackPtr) {
		const deviceId = deviceIdPtr ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		return GodotCamera.api.getPixelData(context, deviceId, width, height, callbackPtr, deniedCallbackPtr, formatsCallbackPtr);
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

	/**
	 * Native binding for canceling an in-flight camera open.
	 * @param {number} context C++ CameraFeedWeb* context
	 */
	godot_js_camera_abort: function (context) {
		GodotCamera.api.abort(context);
	},
};

autoAddDeps(GodotCamera, '$GodotCamera');
mergeInto(LibraryManager.library, GodotCamera);
