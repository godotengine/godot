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

const GodotCamera = {
	$GodotCamera__deps: ['$GodotRuntime', '$GodotConfig', '$GodotOS'],
	$GodotCamera__postset: 'GodotOS.atexit(function(resolve, reject) { GodotCamera.cleanup(); resolve(); });',
	$GodotCamera: {
		state: {
			videoElement: null,
			currentStream: null,
			isInitialized: false,
			activeCameraInfo: { deviceId: null, width: 0, height: 0 },
			availableCameras: [],
			animationFrameId: null,
			stopRequested: false,
		},

		utils: {
			createElements: function () {
				if (GodotCamera.state.isInitialized) {
					return;
				}

				const video = document.createElement('video');
				Object.assign(video, {
					autoplay: true,
					playsInline: true,
					muted: true,
				});
				Object.assign(video.style, {
					position: 'fixed',
					top: '-9999px',
					left: '-9999px',
				});

				if (GodotConfig.canvas) {
					GodotConfig.canvas.insertAdjacentElement('beforebegin', video);
				} else {
					document.body.appendChild(video);
					// GodotCamera: GodotConfig.canvas not found, appending video element to body
				}

				GodotCamera.state.videoElement = video;
				GodotCamera.state.isInitialized = true;
			},

			createConstraints: function (deviceId, width, height) {
				const videoConstraints = {};
				if (deviceId) {
					videoConstraints.deviceId = { exact: deviceId };
				}
				if (width) {
					videoConstraints.width = { ideal: width };
				}
				if (height) {
					videoConstraints.height = { ideal: height };
				}

				return {
					video: Object.keys(videoConstraints).length > 0 ? videoConstraints : true,
					audio: false,
				};
			},

			resetCameraInfo: function () {
				GodotCamera.state.activeCameraInfo = { deviceId: null, width: 0, height: 0 };
			},

			sendError: function (func, context, error, message) {
				const errorObj = { error, message };
				const cStr = GodotRuntime.allocString(JSON.stringify(errorObj));
				func(context, cStr);
				GodotRuntime.free(cStr);
			},

			sendPixelError: function (func, error, message) {
				const errorObj = { error, message };
				const cStr = GodotRuntime.allocString(JSON.stringify(errorObj));
				func(0, 0, 0, 0, 0, cStr);
				GodotRuntime.free(cStr);
			},
		},

		stream: {
			start: async function (deviceId, width, height) {
				const { state, utils } = GodotCamera;
				// GodotRuntime.print(`GodotCamera: stream.start called with deviceId=${deviceId}, width=${width}, height=${height}`);

				if (state.currentStream) {
					// GodotCamera: Found existing stream, checking compatibility
					const currentSettings = state.currentStream.getVideoTracks()[0]?.getSettings();
					if (deviceId === state.activeCameraInfo.deviceId && currentSettings) {
						// GodotCamera: Reusing existing compatible stream
						return {
							width: state.videoElement.videoWidth,
							height: state.videoElement.videoHeight,
							deviceId: state.activeCameraInfo.deviceId,
						};
					}
					// GodotCamera: Stopping incompatible existing stream
					GodotCamera.stream.stop();
				}

				const constraints = utils.createConstraints(deviceId, width, height);
				// GodotRuntime.print('GodotCamera: Created constraints:', JSON.stringify(constraints));

				try {
					// GodotCamera: Calling getUserMedia...
					const stream = await navigator.mediaDevices.getUserMedia(constraints);
					// GodotRuntime.print('GodotCamera: getUserMedia successful, got stream with tracks:', stream.getTracks().length);

					return new Promise((resolve, reject) => {
						const timeoutId = setTimeout(() => {
							GodotRuntime.error('GodotCamera: Timeout waiting for video metadata to load');
							reject(new Error('Timeout waiting for video metadata to load'));
						}, 10000);

						state.videoElement.onloadedmetadata = () => {
							clearTimeout(timeoutId);
							// GodotCamera: Video metadata loaded successfully

							state.currentStream = stream;
							const track = stream.getVideoTracks()[0];
							const settings = track?.getSettings() || {};
							// GodotRuntime.print('GodotCamera: Video track settings:', JSON.stringify(settings));

							state.activeCameraInfo = {
								deviceId: settings.deviceId || deviceId || null,
								width: state.videoElement.videoWidth,
								height: state.videoElement.videoHeight,
							};

							// GodotRuntime.print(`GodotCamera: Video dimensions: ${state.videoElement.videoWidth}x${state.videoElement.videoHeight}`);
							resolve({
								width: state.videoElement.videoWidth,
								height: state.videoElement.videoHeight,
								deviceId: state.activeCameraInfo.deviceId,
							});
						};

						state.videoElement.onerror = (event) => {
							clearTimeout(timeoutId);
							GodotRuntime.error('GodotCamera: Video Element Error:', event);
							utils.resetCameraInfo();
							reject(new Error('Video element error while loading video stream.'));
						};

						// GodotCamera: Setting video srcObject and waiting for metadata...
						state.videoElement.srcObject = stream;
					});
				} catch (err) {
					GodotRuntime.error('GodotCamera: getUserMedia error:', err.name, err.message);
					GodotCamera.stream.stop();
					utils.resetCameraInfo();
					throw err;
				}
			},

			stop: function (deviceId) {
				const { state } = GodotCamera;

				state.stopRequested = true;
				if (state.animationFrameId !== null) {
					cancelAnimationFrame(state.animationFrameId);
					state.animationFrameId = null;
				}

				const shouldStop = !deviceId || state.activeCameraInfo.deviceId === deviceId;

				if (shouldStop && state.currentStream) {
					state.currentStream.getTracks().forEach((track) => track.stop());
					state.currentStream = null;

					if (state.videoElement) {
						state.videoElement.srcObject = null;
						state.videoElement.pause();
						state.videoElement.style.transform = 'scaleX(1)';
					}
					GodotCamera.utils.resetCameraInfo();
				}
			},

		},

		api: {
			init: function () {
				GodotCamera.utils.createElements();
				return GodotCamera.state.isInitialized ? 1 : 0;
			},

			getCameras: async function (context, callbackPtr) {
				const func = GodotRuntime.get_func(callbackPtr);
				const { state, utils } = GodotCamera;

				if (!state.isInitialized) {
					utils.createElements();
				}

				if (!navigator.mediaDevices?.enumerateDevices) {
					GodotRuntime.error('GodotCamera: enumerateDevices is not supported in this browser.');
					state.availableCameras = [];
					utils.sendError(func, context, 'enumerateDevices_not_supported', 'enumerateDevices API is not supported in this browser.');
					return { error: 'enumerateDevices_not_supported', cameras: [] };
				}

				try {
					const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
					tempStream.getTracks().forEach((track) => track.stop());

					const devices = await navigator.mediaDevices.enumerateDevices();
					const videoCameras = devices.filter((device) => device.kind === 'videoinput');

					state.availableCameras = videoCameras.map((camera, index) => ({
						index,
						id: camera.deviceId,
						label: camera.label || `Camera ${index + 1}`,
					}));

					const result = { error: null, cameras: state.availableCameras, count: state.availableCameras.length };
					const cStr = GodotRuntime.allocString(JSON.stringify(result));
					func(context, cStr);
					GodotRuntime.free(cStr);
					return result;
				} catch (err) {
					GodotRuntime.error('GodotCamera: Error getting camera list:', err);
					state.availableCameras = [];
					utils.sendError(func, context, err.name || 'unknown_error', err.message || 'Error getting camera list.');
					return { error: err.name || 'unknown_error', cameras: [], count: 0 };
				}
			},

			getCameraCapabilities: async function (deviceIdPtr, context, callbackPtr) {
				const func = GodotRuntime.get_func(callbackPtr);
				const deviceId = GodotRuntime.parseString(deviceIdPtr);
				const { state, utils } = GodotCamera;

				if (!state.isInitialized) {
					utils.sendError(func, 0, 'not_initialized', 'GodotCamera is not initialized. Call init() first.');
					return { capabilities: null, error: 'not_initialized' };
				}

				if (!deviceId) {
					utils.sendError(func, 0, 'missing_device_id', 'Device ID is required to get capabilities.');
					return { capabilities: null, error: 'missing_device_id' };
				}

				if (!navigator.mediaDevices?.getUserMedia) {
					utils.sendError(func, 0, 'getUserMedia_not_supported', 'getUserMedia is not supported in this browser for capabilities check.');
					return { capabilities: null, error: 'getUserMedia_not_supported' };
				}

				try {
					const stream = await navigator.mediaDevices.getUserMedia({
						video: { deviceId: { exact: deviceId } },
						audio: false,
					});

					const videoTracks = stream.getVideoTracks();
					if (videoTracks.length === 0) {
						utils.sendError(func, 0, 'no_video_track_found', 'No video track found.');
						return { capabilities: null, error: 'no_video_track_found' };
					}

					const track = videoTracks[0];
					const capabilities = track.getCapabilities();

					stream.getTracks().forEach((t) => {
						if (t.readyState === 'live') {
							t.stop();
						}
					});

					const result = { capabilities, error: null, message: `Capabilities retrieved successfully for deviceId: ${deviceId}` };
					const cStr = GodotRuntime.allocString(JSON.stringify(result));
					func(context, cStr);
					GodotRuntime.free(cStr);
					return result;
				} catch (err) {
					utils.sendError(func, 0, err.name || 'get_capabilities_failed', err.message || 'Failed to get camera capabilities.');
					return { capabilities: null, error: err.name || 'get_capabilities_failed' };
				}
			},

			getPixelData: async function (context, deviceId, width, height, callbackPtr) {
				const func = GodotRuntime.get_func(callbackPtr);
				const { state, utils } = GodotCamera;
				// GodotRuntime.print(`GodotCamera: getPixelData called with deviceId=${deviceId}, width=${width}, height=${height}`);

				if (!deviceId) {
					GodotRuntime.error('GodotCamera: Missing device ID');
					utils.sendPixelError(func, 'missing_device_identifier', 'Device ID is required to get pixel data.');
					return;
				}

				if (!state.isInitialized) {
					// GodotCamera: Not initialized, creating elements
					utils.createElements();
				}

				try {
					// GodotCamera: Starting camera stream...
					await GodotCamera.stream.start(deviceId, width, height);
					// GodotCamera: Camera stream started successfully

					if (state.videoElement.readyState < state.videoElement.HAVE_CURRENT_DATA) {
						// GodotRuntime.print(`GodotCamera: Video not ready (readyState: ${state.videoElement.readyState}), waiting...`);

						const waitForVideoData = () => new Promise((resolve, _) => {
							state.videoElement.oncanplay = () => {
								// GodotCamera: Video canplay event fired
								state.videoElement.oncanplay = null;
								resolve();
							};

							if (state.videoElement.readyState >= state.videoElement.HAVE_ENOUGH_DATA) {
								// GodotCamera: Video already has enough data
								state.videoElement.oncanplay = null;
								resolve();
							}
						});

						const timeout = new Promise((_, reject) => {
							setTimeout(() => {
								GodotRuntime.error('GodotCamera: Video data timeout after 5 seconds');
								reject(new Error('Timed out waiting for video data to be ready.'));
							}, 5000);
						});

						await Promise.race([waitForVideoData(), timeout]);
						// GodotCamera: Video data is now ready
					}

					// GodotCamera: Creating offscreen canvas...
					const offscreenCanvas = typeof OffscreenCanvas !== 'undefined'
						? new OffscreenCanvas(width, height)
						: document.createElement('canvas');
					offscreenCanvas.width = width;
					offscreenCanvas.height = height;
					const offscreenContext = offscreenCanvas.getContext('2d', { willReadFrequently: true });
					// GodotCamera: Offscreen canvas created successfully

					// eslint-disable-next-line require-atomic-updates
					state.stopRequested = false;
					// GodotCamera: Starting render loop...
					const renderLoop = () => {
						if (state.stopRequested) {
							// GodotCamera: Render loop stopped by request
							state.animationFrameId = null;
							return;
						}

						try {
							offscreenContext.drawImage(state.videoElement, 0, 0, width, height);
							const imageData = offscreenContext.getImageData(0, 0, width, height);
							const dataSize = imageData.data.length;
							const dataPtr = GodotRuntime.malloc(dataSize);

							if (dataPtr === 0) {
								GodotRuntime.error('GodotCamera: Memory allocation failed');
								utils.sendPixelError(func, 'malloc_failed', 'Memory allocation failed for pixel data.');
								return;
							}

							HEAPU8.set(imageData.data, dataPtr);
							func(context, dataPtr, dataSize, imageData.width, imageData.height, 0);
							GodotRuntime.free(dataPtr);
						} catch (err) {
							GodotRuntime.error('GodotCamera: Render loop error:', err);
							utils.sendPixelError(func, err.name || 'get_pixel_data_failed', err.message || 'Failed to get pixel data.');
							return;
						}

						state.animationFrameId = requestAnimationFrame(renderLoop);
					};

					renderLoop();
				} catch (err) {
					GodotRuntime.error('GodotCamera: getPixelData error:', err);
					GodotCamera.stream.stop();
					utils.sendPixelError(func, err.name || 'get_pixel_data_failed', err.message || 'Failed to get pixel data.');
				}
			},

			cleanup: function () {
				const { state } = GodotCamera;
				if (!state.isInitialized) {
					return;
				}

				GodotCamera.stream.stop();

				if (state.videoElement?.parentNode) {
					state.videoElement.parentNode.removeChild(state.videoElement);
				}

				Object.assign(state, {
					videoElement: null,
					activeCameraInfo: { deviceId: null, width: 0, height: 0 },
					availableCameras: [],
					isInitialized: false,
				});
			},
		},
	},

	godot_js_camera_init: function () {
		return GodotCamera.api.init();
	},

	godot_js_camera_get_pixel_data: function (context, deviceIdPtr, width, height, callbackPtr) {
		const deviceId = deviceIdPtr && deviceIdPtr !== 0 ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		return GodotCamera.api.getPixelData(context, deviceId, width, height, callbackPtr);
	},

	godot_js_camera_stop_stream: function (deviceIdPtr) {
		const deviceId = deviceIdPtr && deviceIdPtr !== 0 ? GodotRuntime.parseString(deviceIdPtr) : undefined;
		GodotCamera.stream.stop(deviceId);
	},
};

autoAddDeps(GodotCamera, '$GodotCamera');
mergeInto(LibraryManager.library, GodotCamera);
