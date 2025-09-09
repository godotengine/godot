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
				'max': 1280,
			},
			'height': {
				'max': 1080,
			},
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
		 * @param {string|null} errorMsg Error message if any
		 * @returns {void}
		 */
		sendGetPixelDataCallback: function (callback, context, dataPtr, dataLen, width, height, errorMsg) {
			const errorMsgPtr = errorMsg ? GodotRuntime.allocString(errorMsg) : 0;
			callback(context, dataPtr, dataLen, width, height, errorMsgPtr);
			if (errorMsgPtr) {
				GodotRuntime.free(errorMsgPtr);
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
					// request camera access permission.
					const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
					const getCapabilities = function (deviceId) {
						const videoTrack = stream.getVideoTracks()
							.find((track) => track.getSettings().deviceId === deviceId);
						return videoTrack?.getCapabilities() || GodotCamera.defaultMinimumCapabilities;
					};
					const devices = await navigator.mediaDevices.enumerateDevices();
					result.cameras = devices
						.filter((device) => device.kind === 'videoinput')
						.map((device, index) => ({
							index,
							id: device.deviceId,
							label: device.label || `Camera ${index}`,
							capabilities: getCapabilities(device.deviceId),
						}));

					stream.getTracks().forEach((track) => track.stop());
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
						// Use requested dimensions when stream already exists.
						_width = width;
						_height = height;
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

								GodotCamera.sendGetPixelDataCallback(
									callback,
									context,
									dataPtr,
									pixelData.length,
									_width,
									_height,
									null);

								GodotRuntime.free(dataPtr);
							} catch (error) {
								GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, error.message);

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
					GodotCamera.sendGetPixelDataCallback(callback, context, 0, 0, 0, 0, error.message);
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
