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
		 * @returns {Array<{width: number, height: number}>} Supported resolutions
		 */
		getSupportedFormats: function (capabilities) {
			const widthRange = capabilities.width || this.defaultMinimumCapabilities.width;
			const heightRange = capabilities.height || this.defaultMinimumCapabilities.height;

			return this.commonResolutions.filter((res) => res.width >= (widthRange.min || 1)
				&& res.width <= (widthRange.max || 9999)
				&& res.height >= (heightRange.min || 1)
				&& res.height <= (heightRange.max || 9999));
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
		 * Cleans up resources for a specific camera.
		 * @param {CameraResource} camera Camera resource to cleanup
		 * @returns {void}
		 */
		cleanupCamera: function (camera) {
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
						};
						camerasMap.set(cameraId, camera);
					}

					let _height, _width;
					if (!camera.stream) {
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
						({ width: _width, height: _height } = videoTrack.getSettings());
						if (typeof OffscreenCanvas !== 'undefined') {
							camera.canvas = new OffscreenCanvas(_width, _height);
						} else {
							camera.canvas = document.createElement('canvas');
							camera.canvas.style.display = 'none';
							document.body.appendChild(camera.canvas);
						}
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
					} else {
						// Get actual dimensions from existing stream.
						const [videoTrack] = camera.stream.getVideoTracks();
						({ width: _width, height: _height } = videoTrack.getSettings());
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

								// Get screen orientation

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
									null);

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
						GodotCamera.cleanupCamera(camera);
					}
					cameras.delete(deviceId);
				} else {
					cameras.forEach((camera) => {
						if (camera) {
							GodotCamera.cleanupCamera(camera);
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
