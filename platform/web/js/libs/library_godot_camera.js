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
		_videoElement: null,
		_canvasElement: null,
		_context: null,
		_currentStream: null,
		_isInitialized: false,
		_activeCameraInfo: {
			deviceId: null,
			width: 0,
			height: 0,
		},
		_availableCameras: [],
		_getPixelDataAnimationFrameId: null,
		_getPixelDataStopRequested: false,

		_initializeElements: function () {
			if (GodotCamera._isInitialized) {
				return;
			}
			const video = document.createElement('video');
			video.autoplay = true;
			video.playsInline = true;
			video.muted = true;
			video.style.position = 'fixed';
			video.style.top = '-9999px';
			video.style.left = '-9999px';

			if (GodotConfig.canvas) {
				GodotConfig.canvas.insertAdjacentElement('beforebegin', video);
			} else {
				document.body.appendChild(video);
				GodotRuntime.warn('GodotCamera: GodotConfig.canvas not found, appending video element to body.');
			}
			GodotCamera._videoElement = video;

			let canvas;
			if (typeof OffscreenCanvas !== 'undefined') {
				canvas = new OffscreenCanvas(1, 1);
			} else {
				canvas = document.createElement('canvas');
			}
			const context = canvas.getContext('2d', { willReadFrequently: true });
			GodotCamera._canvasElement = canvas;
			GodotCamera._context = context;

			GodotCamera._isInitialized = true;
		},

		_startStreamInternal: async function (deviceId, width, height) {
			if (GodotCamera._currentStream) {
				const currentSettings = GodotCamera._currentStream.getVideoTracks()[0]?.getSettings();
				if (deviceId && deviceId === GodotCamera._activeCameraInfo.deviceId && currentSettings) {
					if (width && height) {
						GodotCamera._canvasElement.width = width;
						GodotCamera._canvasElement.height = height;
					} else {
						GodotCamera._canvasElement.width = GodotCamera._videoElement.videoWidth;
						GodotCamera._canvasElement.height = GodotCamera._videoElement.videoHeight;
					}
					return {
						width: GodotCamera._videoElement.videoWidth,
						height: GodotCamera._videoElement.videoHeight,
						deviceId: GodotCamera._activeCameraInfo.deviceId,
					};
				}
				GodotCamera._stopStreamInternal();
			}

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

			const constraints = {
				video: Object.keys(videoConstraints).length > 0 ? videoConstraints : true,
				audio: false,
			};

			try {
				const stream = await navigator.mediaDevices.getUserMedia(constraints);
				return new Promise((resolve, reject) => {
					GodotCamera._videoElement.onloadedmetadata = () => {
						GodotCamera._currentStream = stream;
						GodotCamera._videoElement.srcObject = stream;
						const track = stream.getVideoTracks()[0];
						const settings = track?.getSettings() || {};
						GodotCamera._activeCameraInfo = {
							deviceId: settings.deviceId || (deviceId || null),
							width: GodotCamera._videoElement.videoWidth,
							height: GodotCamera._videoElement.videoHeight,
						};
						if (width && height) {
							GodotCamera._canvasElement.width = width;
							GodotCamera._canvasElement.height = height;
						} else {
							GodotCamera._canvasElement.width = GodotCamera._videoElement.videoWidth;
							GodotCamera._canvasElement.height = GodotCamera._videoElement.videoHeight;
						}

						resolve({
							width: GodotCamera._videoElement.videoWidth,
							height: GodotCamera._videoElement.videoHeight,
							deviceId: GodotCamera._activeCameraInfo.deviceId,
						});
					};
					GodotCamera._videoElement.onerror = (event) => {
						GodotRuntime.error('GodotCamera: Video Element Error:', event);
						GodotCamera._activeCameraInfo = { deviceId: null, width: 0, height: 0 };
						Promise.reject(new Error('Video element error while loading video stream.'));
					};
				});
			} catch (err) {
				GodotRuntime.error('GodotCamera: getUserMedia error in _startStreamInternal:', err.name, err.message);
				GodotCamera._stopStreamInternal();
				GodotCamera._activeCameraInfo = { deviceId: null, width: 0, height: 0 };
				return Promise.reject(err);
			}
		},

		_stopStreamInternal: function (deviceId) {
			GodotCamera._getPixelDataStopRequested = true;
			if (GodotCamera._getPixelDataAnimationFrameId !== null) {
				cancelAnimationFrame(GodotCamera._getPixelDataAnimationFrameId);
				GodotCamera._getPixelDataAnimationFrameId = null;
			}

			if (deviceId) {
				if (
					GodotCamera._currentStream
					&& GodotCamera._activeCameraInfo.deviceId === deviceId) {
					GodotCamera._currentStream.getTracks().forEach((track) => track.stop());
					GodotCamera._currentStream = null;
					if (GodotCamera._videoElement) {
						GodotCamera._videoElement.srcObject = null;
						GodotCamera._videoElement.pause();
						GodotCamera._videoElement.style.transform = 'scaleX(1)';
					}
					GodotCamera._activeCameraInfo = { deviceId: null, width: 0, height: 0 };
				}
			} else {
				// deviceIdがない場合は全デバイス停止（現状1つのみ管理なので従来通り）
				if (GodotCamera._currentStream) {
					GodotCamera._currentStream.getTracks().forEach((track) => track.stop());
					GodotCamera._currentStream = null;
				}
				if (GodotCamera._videoElement) {
					GodotCamera._videoElement.srcObject = null;
					GodotCamera._videoElement.pause();
					GodotCamera._videoElement.style.transform = 'scaleX(1)';
				}
				GodotCamera._activeCameraInfo = { deviceId: null, width: 0, height: 0 };
			}
		},

		init: function () {
			GodotCamera._initializeElements();
			return GodotCamera._isInitialized ? 1 : 0;
		},

		getCameras: function (context, p_callback_ptr) {
			const func = GodotRuntime.get_func(p_callback_ptr);
			let cameras_array_json;
			if (!GodotCamera._isInitialized) {
				GodotCamera._initializeElements();
			}
			if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
				GodotRuntime.error('GodotCamera: enumerateDevices is not supported in this browser.');
				GodotCamera._availableCameras = [];
				cameras_array_json = JSON.stringify({ error: 'enumerateDevices_not_supported', message: 'enumerateDevices API is not supported in this browser.', cameras: [] });
				const c_str = GodotRuntime.allocString(cameras_array_json);
				func(context, c_str);
				GodotRuntime.free(c_str);
				return;
			}
			navigator.mediaDevices.getUserMedia({ video: true }).then((tempStream) => {
				tempStream.getTracks().forEach((track) => track.stop());
				navigator.mediaDevices.enumerateDevices().then((devices) => {
					const videoCameras = devices.filter((device) => device.kind === 'videoinput');
					GodotCamera._availableCameras = videoCameras.map((camera, index) => ({
						index: index,
						id: camera.deviceId,
						label: camera.label || `Camera ${index + 1}`,
					}));
					cameras_array_json = JSON.stringify({ error: null, cameras: GodotCamera._availableCameras, count: GodotCamera._availableCameras.length });
					const c_str = GodotRuntime.allocString(cameras_array_json);
					func(context, c_str);
					GodotRuntime.free(c_str);
				}).catch((err) => {
					GodotRuntime.error('GodotCamera: Error getting camera list:', err);
					GodotCamera._availableCameras = [];
					cameras_array_json = JSON.stringify({ error: err.name || 'unknown_error', message: err.message || 'Error getting camera list.', cameras: [], count: 0 });
					const c_str = GodotRuntime.allocString(cameras_array_json);
					func(context, c_str);
					GodotRuntime.free(c_str);
				});
			}).catch((err) => {
				GodotRuntime.error('GodotCamera: Error getting camera list:', err);
				GodotCamera._availableCameras = [];
				cameras_array_json = JSON.stringify({ error: err.name || 'unknown_error', message: err.message || 'Error getting camera list.', cameras: [], count: 0 });
				const c_str = GodotRuntime.allocString(cameras_array_json);
				func(context, c_str);
				GodotRuntime.free(c_str);
			});
		},

		getCameraCapabilities: function (p_device_id_ptr, context, p_callback_ptr) {
			const func = GodotRuntime.get_func(p_callback_ptr);
			const deviceId = GodotRuntime.parseString(p_device_id_ptr);
			let result_obj;

			if (!GodotCamera._isInitialized && typeof GodotCamera.init === 'function') {
				result_obj = { capabilities: null, error: 'not_initialized', message: 'GodotCamera is not initialized. Call init() first.' };
				const c_err_str = GodotRuntime.allocString(JSON.stringify(result_obj));
				func(0, c_err_str);
				GodotRuntime.free(c_err_str);
				return;
			}

			if (!deviceId) {
				result_obj = { capabilities: null, error: 'missing_device_id', message: 'Device ID is required to get capabilities.' };
				const c_err_str = GodotRuntime.allocString(JSON.stringify(result_obj));
				func(0, c_err_str);
				GodotRuntime.free(c_err_str);
				return;
			}

			if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
				result_obj = { capabilities: null, error: 'getUserMedia_not_supported', message: 'getUserMedia is not supported in this browser for capabilities check.' };
				const c_err_str = GodotRuntime.allocString(JSON.stringify(result_obj));
				func(0, c_err_str);
				GodotRuntime.free(c_err_str);
				return;
			}

			navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } }, audio: false }).then((stream) => {
				const videoTracks = stream.getVideoTracks();
				if (videoTracks.length === 0) {
					result_obj = { capabilities: null, error: 'no_video_track_found', message: 'No video track found.' };
					const c_err_str = GodotRuntime.allocString(JSON.stringify(result_obj));
					func(0, c_err_str);
					GodotRuntime.free(c_err_str);
					return;
				}
				const track = videoTracks[0];
				const capabilities = track.getCapabilities();
				track.stop();
				stream.getTracks().forEach((t) => {
					if (t.readyState === 'live') {
						t.stop();
					}
				});
				result_obj = { capabilities: capabilities, error: null, message: `Capabilities retrieved successfully for deviceId: ${deviceId}` };
				const json_str = JSON.stringify(result_obj);
				const c_str = GodotRuntime.allocString(json_str);
				func(context, c_str);
				GodotRuntime.free(c_str);
			}).catch((err) => {
				result_obj = { capabilities: null, error: err.name || 'get_capabilities_failed', message: err.message || 'Failed to get camera capabilities.' };
				const c_err_str = GodotRuntime.allocString(JSON.stringify(result_obj));
				func(0, c_err_str);
				GodotRuntime.free(c_err_str);
			});
		},

		getPixelData: async function (context, deviceId, width, height, p_callback_ptr) {
			const func = GodotRuntime.get_func(p_callback_ptr);
			if (!deviceId) {
				const err_obj = { error: 'missing_device_identifier', message: 'Device ID or facing mode is required to get pixel data.' };
				const c_err_str = GodotRuntime.allocString(JSON.stringify(err_obj));
				func(0, 0, 0, 0, 0, c_err_str);
				GodotRuntime.free(c_err_str);
				return;
			}

			if (!GodotCamera._isInitialized) {
				GodotCamera._initializeElements();
			}

			try {
				await GodotCamera._startStreamInternal(deviceId, width, height);

				if (GodotCamera._videoElement.readyState < GodotCamera._videoElement.HAVE_CURRENT_DATA) {
					await new Promise((resolve, reject) => {
						const timeoutId = setTimeout(() => {
							GodotCamera._videoElement.oncanplay = null;
							Promise.reject(new Error('Timed out waiting for video data to be ready.'));
						}, 5000);
						GodotCamera._videoElement.oncanplay = () => {
							clearTimeout(timeoutId);
							GodotCamera._videoElement.oncanplay = null;
							resolve();
						};
						if (GodotCamera._videoElement.readyState >= GodotCamera._videoElement.HAVE_ENOUGH_DATA) {
							clearTimeout(timeoutId);
							GodotCamera._videoElement.oncanplay = null;
							resolve();
						}
					});
				}

				GodotCamera._canvasElement.width = GodotCamera._activeCameraInfo.width || GodotCamera._videoElement.videoWidth;
				GodotCamera._canvasElement.height = GodotCamera._activeCameraInfo.height || GodotCamera._videoElement.videoHeight;
				const ctx = GodotCamera._context;

				// requestAnimationFrameでループし、毎フレームfuncを呼び出す
				GodotCamera._getPixelDataStopRequested = false;
				const loop = () => {
					if (GodotCamera._getPixelDataStopRequested) {
						GodotCamera._getPixelDataAnimationFrameId = null;
						return;
					}
					try {
						ctx.drawImage(GodotCamera._videoElement, 0, 0, GodotCamera._canvasElement.width, GodotCamera._canvasElement.height);
						const imageData = ctx.getImageData(0, 0, GodotCamera._canvasElement.width, GodotCamera._canvasElement.height);
						const data_size = imageData.data.length;
						const data_ptr = Module._malloc(data_size);
						if (data_ptr === 0) {
							const err_obj = { error: 'malloc_failed', message: 'Memory allocation failed for pixel data.' };
							const c_err_str = GodotRuntime.allocString(JSON.stringify(err_obj));
							func(0, 0, 0, 0, 0, c_err_str);
							GodotRuntime.free(c_err_str);
							return;
						}
						HEAPU8.set(imageData.data, data_ptr);
						func(context, data_ptr, data_size, imageData.width, imageData.height, 0);
					} catch (err) {
						const err_obj = { error: err.name || 'get_pixel_data_failed', message: err.message || 'Failed to get pixel data.' };
						const c_err_str = GodotRuntime.allocString(JSON.stringify(err_obj));
						func(0, 0, 0, 0, 0, c_err_str);
						GodotRuntime.free(c_err_str);
						return;
					}
					GodotCamera._getPixelDataAnimationFrameId = requestAnimationFrame(loop);
				};
				loop();
			} catch (err) {
				const err_obj = { error: err.name || 'get_pixel_data_failed', message: err.message || 'Failed to get pixel data.' };
				GodotCamera._stopStreamInternal();
				const c_err_str = GodotRuntime.allocString(JSON.stringify(err_obj));
				func(0, 0, 0, 0, 0, c_err_str);
				GodotRuntime.free(c_err_str);
			}
		},

		stopStream: function (deviceId) {
			GodotCamera._stopStreamInternal(deviceId);
		},

		cleanup: function () {
			if (!GodotCamera._isInitialized) {
				return;
			}
			GodotCamera._stopStreamInternal();
			if (GodotCamera._videoElement) {
				if (GodotCamera._videoElement.parentNode) {
					GodotCamera._videoElement.parentNode.removeChild(GodotCamera._videoElement);
				}
				GodotCamera._videoElement = null;
			}
			GodotCamera._canvasElement = null;
			GodotCamera._context = null;
			GodotCamera._activeCameraInfo = { deviceId: null, width: 0, height: 0 };
			GodotCamera._availableCameras = [];
			GodotCamera._isInitialized = false;
		},
	},

	godot_js_camera_init__proxy: 'sync',
	godot_js_camera_init__sig: 'i',
	godot_js_camera_init: function () {
		return GodotCamera.init();
	},

	godot_js_camera_get_cameras__proxy: 'sync',
	godot_js_camera_get_cameras__sig: 'vpp',
	godot_js_camera_get_cameras: function (context, p_callback_ptr) {
		GodotCamera.getCameras(context, p_callback_ptr);
	},

	godot_js_camera_get_capabilities__proxy: 'sync',
	godot_js_camera_get_capabilities__sig: 'vipp',
	godot_js_camera_get_capabilities: function (p_device_id_ptr, context, p_callback_ptr) {
		GodotCamera.getCameraCapabilities(p_device_id_ptr, context, p_callback_ptr);
	},

	godot_js_camera_get_pixel_data__proxy: 'async',
	godot_js_camera_get_pixel_data__sig: 'vpiiip',
	godot_js_camera_get_pixel_data: function (context, device_id, width, height, p_callback_ptr) {
		GodotCamera.getPixelData(context, device_id, width, height, p_callback_ptr);
	},

	godot_js_camera_stop_stream__proxy: 'sync',
	godot_js_camera_stop_stream__sig: 'vi',
	godot_js_camera_stop_stream: function (p_device_id_ptr) {
		let deviceId;
		if (typeof p_device_id_ptr !== 'undefined' && p_device_id_ptr !== 0) {
			deviceId = GodotRuntime.parseString(p_device_id_ptr);
		}
		GodotCamera.stopStream(deviceId);
	},
};

autoAddDeps(GodotCamera, '$GodotCamera');
mergeInto(LibraryManager.library, GodotCamera);
