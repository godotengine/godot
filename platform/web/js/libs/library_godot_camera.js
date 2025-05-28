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
 *   video: HTMLVideoElement?
 *   canvas: HTMLCanvasElement | OffscreenCanvas?
 *   canvasContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D?
 *   stream: MediaStream?
 *   animationFrameId: number?
 *   permissionListener: Function?
 *   permissionStatus: PermissionStatus?
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
		 * @param {number} context Context value to pass to callback
		 * @param {Object} result Result object to stringify
		 * @returns {void}
		 */
		sendCallbackResult: function (callback, context, result) {
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
		 * @param {string?} errorMsg Error message if any
		 * @returns {void}
		 */
		sendErrorCallback: function (callback, context, dataPtr, dataLen, width, height, errorMsg) {
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
		},

		api: {
			/**
			 * Gets list of available cameras.
			 * Calls callback with JSON containing array of camera info.
			 * @param {number} context Context value to pass to callback
			 * @param {number} callbackPtr Pointer to callback function
			 * @returns {Promise<void>}
			 */
			getCameras: async function (context, callbackPtr) {
				const callback = GodotRuntime.get_func(callbackPtr);
				const result = { error: null, cameras: null };

				try {
					// request camera access permission
					const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
					stream.getTracks().forEach((track) => track.stop());
					const devices = await navigator.mediaDevices.enumerateDevices();
					result.cameras = devices
						.filter((device) => device.kind === 'videoinput')
						.map((device, index) => ({
							index,
							id: device.deviceId,
							label: device.label || `Camera ${index}`,
						}));

					GodotCamera.api.stop();
				} catch (error) {
					result.error = error.message;
				}

				GodotCamera.sendCallbackResult(callback, context, result);
			},

			/**
			 * Gets capabilities of a specific camera.
			 * Calls callback with JSON containing camera capabilities.
			 * @param {number} deviceIdPtr Pointer to device ID string
			 * @param {number} context Context value to pass to callback
			 * @param {number} callbackPtr Pointer to callback function
			 * @returns {Promise<void>}
			 */
			getCameraCapabilities: async function (deviceIdPtr, context, callbackPtr) {
				const callback = GodotRuntime.get_func(callbackPtr);
				const deviceId = GodotRuntime.parseString(deviceIdPtr);
				const result = { error: null, capabilities: null };

				try {
					// request camera access permission
					const stream = await navigator.mediaDevices.getUserMedia({
						video: { deviceId: { exact: deviceId } },
						audio: false,
					});

					const videoTrack = stream.getVideoTracks()[0];
					result.capabilities = videoTrack.getCapabilities();

					stream.getTracks().forEach((track) => track.stop());
				} catch (error) {
					result.error = error.message;
				}

				GodotCamera.sendCallbackResult(callback, context, result);
			},

			/**
			 * Starts capturing pixel data from camera.
			 * Continuously calls callback with pixel data.
			 * @param {number} context Context value to pass to callback
			 * @param {string?} deviceId Camera device ID
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

					if (!camera.stream) {
						camera.video = document.createElement('video');
						camera.video.style.display = 'none';
						camera.video.autoplay = true;
						camera.video.playsInline = true;
						document.body.appendChild(camera.video);

						if (typeof OffscreenCanvas !== 'undefined') {
							camera.canvas = new OffscreenCanvas(width, height);
						} else {
							camera.canvas = document.createElement('canvas');
							camera.canvas.style.display = 'none';
							document.body.appendChild(camera.canvas);
						}

						const constraints = {
							video: {
								deviceId: deviceId ? { exact: deviceId } : undefined,
								width: { ideal: width },
								height: { ideal: height },
							},
						};
						// eslint-disable-next-line require-atomic-updates
						camera.stream = await navigator.mediaDevices.getUserMedia(constraints);

						const videoTrack = camera.stream.getVideoTracks()[0];
						if (videoTrack) {
							videoTrack.addEventListener('ended', () => {
								GodotRuntime.print('Camera track ended, stopping stream');
								GodotCamera.api.stop(deviceId);
							});
						}

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
											camera.permissionListener = null;
											camera.permissionStatus = null;
										}
										GodotCamera.api.stop(deviceId);
										deniedCallback(context);
									}
								};
								permissionStatus.addEventListener('change', camera.permissionListener);
							} catch (e) {
								GodotRuntime.error(e);
							}
						}

						camera.video.srcObject = camera.stream;
						await camera.video.play();
					}

					if (camera.canvas.width !== width || camera.canvas.height !== height) {
						camera.canvas.width = width;
						camera.canvas.height = height;
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
							GodotCamera.api.stop(deviceId);
							return;
						}

						if (video.readyState === video.HAVE_ENOUGH_DATA) {
							try {
								canvasContext.drawImage(video, 0, 0, width, height);
								const imageData = canvasContext.getImageData(0, 0, width, height);
								const pixelData = imageData.data;

								const dataPtr = GodotRuntime.malloc(pixelData.length);
								GodotRuntime.heapCopy(HEAPU8, pixelData, dataPtr);

								GodotCamera.sendErrorCallback(
									callback,
									context,
									dataPtr,
									pixelData.length,
									video.videoWidth,
									video.videoHeight,
									null
								);

								GodotRuntime.free(dataPtr);
							} catch (error) {
								GodotCamera.sendErrorCallback(callback, context, 0, 0, 0, 0, error.message);

								if (error.name === 'SecurityError' || error.name === 'NotAllowedError') {
									GodotRuntime.print('Security error, stopping stream:', error);
									GodotCamera.api.stop(deviceId);
								}
								return;
							}
						}

						currentCamera.animationFrameId = requestAnimationFrame(captureFrame);
					};

					camera.animationFrameId = requestAnimationFrame(captureFrame);
				} catch (error) {
					GodotCamera.sendErrorCallback(callback, context, 0, 0, 0, 0, error.message);
				}
			},

			/**
			 * Stops camera stream(s).
			 * @param {string?} deviceId Device ID to stop, or null to stop all
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
		const deviceId = deviceIdPtr && deviceIdPtr !== 0 ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		return GodotCamera.api.getPixelData(context, deviceId, width, height, callbackPtr, deniedCallbackPtr);
	},

	/**
	 * Native binding for stopping camera stream.
	 * @param {number} deviceIdPtr Pointer to device ID string
	 * @returns {void}
	 */
	godot_js_camera_stop_stream: function (deviceIdPtr) {
		const deviceId = deviceIdPtr && deviceIdPtr !== 0 ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		GodotCamera.api.stop(deviceId);
	},
};

autoAddDeps(GodotCamera, '$GodotCamera');
mergeInto(LibraryManager.library, GodotCamera);
