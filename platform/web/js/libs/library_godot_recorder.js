
/**
 * Web Audio Recorder using MediaRecorder API for web audio recording
 * Implementation option 1: MediaRecorder API
 */
const GodotAudioRecorder = {
	$GodotAudioRecorder__deps: ['$GodotAudio'],
	$GodotAudioRecorder__postset: [
		'Module["downloadRecordedVideo"] = GodotAudioRecorder.downloadRecordedVideo;',
		'Module["getRecordedVideoBlob"] = GodotAudioRecorder.getRecordedVideoBlob;',
		'Module["setRecorderCanvas"] = GodotAudioRecorder.setRecorderCanvas;',
	].join(''),
	$GodotAudioRecorder: {
		// Private state
		initialized: false,
		mediaRecorder: null,
		mediaStreamDestination: null,
		recordedChunks: [],
		isRecording: false,
		selectedMimeType: '',
		
		// Video recording related
		videoRecorderInitialized: false,
		videoMediaRecorder: null,
		videoStream: null,
		combinedStream: null,
		videoRecordedChunks: [],
		targetCanvas: null,  // Add explicit targetCanvas property
		videoFPS: 30,
		
		// Performance optimization cache
		cachedDataSize: 0,
		lastChunkCount: 0,
		cachedBlob: null,
		lastBlobCreationTime: 0,
		
		// New data state tracking (for quick checks)
		hasNewData: false,
		lastCheckedChunkCount: 0,
		
		/**
		 * Initialize recorder (does not start recording automatically)
		 */
		init: function() {
			if (!GodotAudio.ctx) {
				GodotRuntime.error('AudioContext not initialized for recording');
				return false;
			}
			
			try {
				// Create recording destination - MediaStreamDestination
				this.mediaStreamDestination = GodotAudio.ctx.createMediaStreamDestination();
				
				// Select supported MIME type
				this.selectedMimeType = this.getSupportedMimeType();
				
				// Connect master audio bus to recording destination
				const masterBus = GodotAudio.buses[0];
				if (masterBus) {
					masterBus.getOutputNode().connect(this.mediaStreamDestination);
				} else {
					GodotRuntime.error('GodotAudioRecorder: Master bus not found');
					return false;
				}
				
				this.initialized = true;
				return true;
				
			} catch (error) {
				GodotRuntime.error('GodotAudioRecorder init failed: ' + error.message);
				return false;
			}
		},
		
		/**
		 * Start recording
		 */
		startRecording: function() {
			if (!this.initialized) {
				GodotRuntime.error('GodotAudioRecorder: Not initialized');
				return false;
			}
			
			if (this.isRecording) {
				return true;
			}
			
			try {
				// Create MediaRecorder
				this.mediaRecorder = new MediaRecorder(this.mediaStreamDestination.stream, {
					mimeType: this.selectedMimeType,
					audioBitsPerSecond: 128000
				});
				
				this.recordedChunks = [];
				
				// Handle recorded data
				this.mediaRecorder.ondataavailable = (event) => {
					if (event.data.size > 0) {
						this.recordedChunks.push(event.data);
						
						// Clear cache, force recalculation
						this.cachedBlob = null;
						
						// Mark as new data (for quick check)
						this.hasNewData = true;
						
						// Significantly reduce log frequency: only output when data chunks are large (to avoid frequent small chunk logs)
						if (event.data.size > 1000) { // Increase threshold to 1KB
						}
					}
				};
				
				// Error handling
				this.mediaRecorder.onerror = (event) => {
					GodotRuntime.error('GodotAudioRecorder MediaRecorder error: ' + event.error);
				};
				
				// Recording stop handling
				this.mediaRecorder.onstop = () => {
				};
				
				// Start recording (one chunk every 100ms)
				this.mediaRecorder.start(100);
				this.isRecording = true;
				
				return true;
				
			} catch (error) {
				GodotRuntime.error('GodotAudioRecorder: Failed to start recording: ' + error.message);
				return false;
			}
		},
		
		/**
		 * Stop recording
		 */
		stopRecording: function() {
			if (this.mediaRecorder && this.isRecording) {
				this.mediaRecorder.stop();
				this.isRecording = false;
				return true;
			}
			return false;
		},
		
		/**
		 * Get recorded audio data as Blob (optimized version, uses cache)
		 */
		getRecordedAudioBlob: function() {
			if (this.recordedChunks.length === 0) {
				return null;
			}
			
			// Check if blob needs to be recreated (only when there is new data)
			if (this.cachedBlob && this.lastChunkCount === this.recordedChunks.length) {
				return this.cachedBlob;
			}
			
			// Create new blob and cache it
			this.cachedBlob = new Blob(this.recordedChunks, {
				type: this.selectedMimeType
			});
			this.lastChunkCount = this.recordedChunks.length;
			this.cachedDataSize = this.cachedBlob.size;
			
			// Only output verification info on first creation or significant size change (reduce log frequency)
			const currentTime = Date.now();
			if (!this.lastBlobCreationTime || (currentTime - this.lastBlobCreationTime) > 1000) {
				this.lastBlobCreationTime = currentTime;
			}
			
			return this.cachedBlob;
		},
		
		/**
		 * Get recorded audio data as ArrayBuffer (for C++ use)
		 */
		getRecordedAudioData: function() {
			const blob = this.getRecordedAudioBlob();
			if (!blob) {
				return null;
			}
			
			return blob.arrayBuffer();
		},
		
		/**
		 * Get supported MIME type
		 */
		getSupportedMimeType: function() {
			// Prioritize high-quality audio formats (note: these are all pure audio formats, without video)
			const types = [
				'audio/webm;codecs=opus',  // Best choice: Opus encoding, high quality, low latency
				'audio/webm',              // WebM audio container
				'audio/mp4',               // MP4 audio container
				'audio/wav'                // WAV format (uncompressed)
			];
			
			for (const type of types) {
				if (MediaRecorder.isTypeSupported(type)) {
					return type;
				}
			}
			
			// As a fallback, use webm (supported by almost all modern browsers)
			return 'audio/webm';
		},
		
		/**
		 * Check if recording is active
		 */
		isActiveRecording: function() {
			return this.isRecording;
		},
		
		/**
		 * Clean up resources
		 */
		cleanup: function() {
			this.stopRecording();
			this.recordedChunks = [];
			this.mediaRecorder = null;
			
			if (this.mediaStreamDestination) {
				this.mediaStreamDestination.disconnect();
				this.mediaStreamDestination = null;
			}
			
		},
		
		/**
		 * Quick check for new recorded data (lightweight API)
		 * Much faster than getRecordedAudioBlob() because it doesn't need to create a Blob object
		 * @returns {boolean} Whether there is new data
		 */
		hasNewRecordingData: function() {
			// Quick check: has the number of chunks changed?
			const currentChunkCount = this.recordedChunks.length;
			if (currentChunkCount !== this.lastCheckedChunkCount) {
				this.lastCheckedChunkCount = currentChunkCount;
				this.hasNewData = true;
				return true;
			}
			
			// Check for previously unconfirmed new data
			if (this.hasNewData) {
				this.hasNewData = false; // Reset flag
				return true;
			}
			
			return false;
		},
		
		/**
		 * Quick check for recorded data (any data, not just new data)
		 * Much faster than getRecordedAudioBlob() because it doesn't need to create a Blob object
		 * @returns {boolean} Whether there is recorded data
		 */
		hasRecordingData: function() {
			return this.recordedChunks.length > 0;
		},
		
		/**
		 * Initialize Canvas video recorder (audio and video merged recording)
		 * @param {number} fps Video frame rate, default 30
		 * @returns {boolean} Whether successful
		 */
		initVideoRecorder: function(fps = 30) {
			if (!GodotAudio.ctx) {
				GodotRuntime.error('AudioContext not initialized for video recording');
				return false;
			}
			
			try {
				// Set video frame rate
				this.videoFPS = fps;
				
				if(GodotAudioRecorder.targetCanvas == undefined || GodotAudioRecorder.targetCanvas == null) {
					GodotRuntime.error('GodotVideoRecorder: Game canvas not set, please call setRecorderCanvas() first');
					return false;
				}
				
				// 2. Get Canvas video stream
				this.videoStream = GodotAudioRecorder.targetCanvas.captureStream(fps);
				
				// 3. Create audio recording destination (if not already created)
				if (!this.mediaStreamDestination) {
					this.mediaStreamDestination = GodotAudio.ctx.createMediaStreamDestination();
					
					// Connect master audio bus to recording destination
					const masterBus = GodotAudio.buses[0];
					if (masterBus) {
						masterBus.getOutputNode().connect(this.mediaStreamDestination);
					}
				}
				
				// 4. Merge audio and video streams
				this.combinedStream = new MediaStream();
				
				// Add video track
				this.videoStream.getVideoTracks().forEach(track => {
					this.combinedStream.addTrack(track);
				});
				
				// Add audio track
				this.mediaStreamDestination.stream.getAudioTracks().forEach(track => {
					this.combinedStream.addTrack(track);
				});
				
				// 5. Select supported video MIME type
				this.selectedMimeType = this.getSupportedVideoMimeType();
				
				this.videoRecorderInitialized = true;
				
				return true;
				
			} catch (error) {
				GodotRuntime.error('GodotVideoRecorder: Failed to initialize - ' + error.message);
				return false;
			}
		},
		
		/**
		 * Get supported video MIME type
		 * @returns {string}
		 */
		getSupportedVideoMimeType: function() {
			// Prioritize high-quality video formats
			const videoTypes = [
				'video/webm;codecs=vp9,opus',     // VP9 video + Opus audio
				'video/webm;codecs=vp8,opus',     // VP8 video + Opus audio
				'video/webm;codecs=h264,opus',    // H.264 video + Opus audio
				'video/webm',                     // WebM default codec
				'video/mp4;codecs=h264,aac',      // H.264 video + AAC audio
				'video/mp4'                       // MP4 default codec
			];
			
			for (const type of videoTypes) {
				if (MediaRecorder.isTypeSupported(type)) {
					return type;
				}
			}
			
			// As a fallback, use webm
			return 'video/webm';
		},
		
		/**
		 * Start video recording (audio and video merged)
		 * @returns {boolean}
		 */
		startVideoRecording: function() {
			if (!this.videoRecorderInitialized) {
				GodotRuntime.error('GodotVideoRecorder: Not initialized');
				return false;
			}
			
			if (this.isRecording) {
				return true;
			}
			
			try {
				// Create video MediaRecorder (includes audio and video)
				this.videoMediaRecorder = new MediaRecorder(this.combinedStream, {
					mimeType: this.selectedMimeType,
					videoBitsPerSecond: 2500000, // 2.5 Mbps
					audioBitsPerSecond: 128000   // 128 kbps
				});
				
				this.videoRecordedChunks = [];
				
				// Set event handlers
				this.videoMediaRecorder.ondataavailable = (event) => {
					if (event.data.size > 0) {
						this.videoRecordedChunks.push(event.data);
						
						this.cachedBlob = null;
						this.hasNewData = true;
					}
				};
				
				this.videoMediaRecorder.onerror = (event) => {
					GodotRuntime.error('GodotVideoRecorder: MediaRecorder error - ' + event.error);
				};
				
				this.videoMediaRecorder.onstop = () => {
				};
				
				// Start recording (one chunk every 100ms)
				this.videoMediaRecorder.start(100);
				this.isRecording = true;
				
				
				return true;
				
			} catch (error) {
				GodotRuntime.error('GodotVideoRecorder: Failed to start recording - ' + error.message);
				return false;
			}
		},
		
		/**
		 * Stop video recording
		 * @returns {boolean}
		 */
		stopVideoRecording: function() {
			if (this.videoMediaRecorder && this.isRecording) {
				this.videoMediaRecorder.stop();
				this.isRecording = false;
				return true;
			}
			return false;
		},
		setRecorderCanvas: function(canvas) {
			GodotAudioRecorder.targetCanvas = canvas;
		},

		/**
		 * Get recorded video data as Blob (includes audio and video)
		 * @returns {Blob|null}
		 */
		getRecordedVideoBlob: function() {
			if (this.videoRecordedChunks.length === 0) {
				return null;
			}
			
			// Check if blob needs to be recreated
			if (this.cachedBlob && this.lastChunkCount === this.videoRecordedChunks.length) {
				return this.cachedBlob;
			}
			
			// Create new blob and cache it
			this.cachedBlob = new Blob(this.videoRecordedChunks, {
				type: this.selectedMimeType
			});
			this.lastChunkCount = this.videoRecordedChunks.length;
			this.cachedDataSize = this.cachedBlob.size;
			
			// Output verification info
			const currentTime = Date.now();
			if (!this.lastBlobCreationTime || (currentTime - this.lastBlobCreationTime) > 1000) {
				this.lastBlobCreationTime = currentTime;
			}
			
			return this.cachedBlob;
		},

		downloadRecordedVideo: function(filename) {
			const blob = GodotAudioRecorder.getRecordedVideoBlob();
			if (!blob) {
				GodotRuntime.error('GodotVideoRecorder: No recorded video data to download');
				return;
			}
			
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename || 'recorded_video.' + blob.type.split('/')[1].split(';')[0];
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
		},
		/**
		 * Clean up video recording resources
		 */
		cleanupVideoRecorder: function() {
			this.stopVideoRecording();
			
			this.videoRecordedChunks = [];
			this.videoMediaRecorder = null;
			
			if (this.videoStream) {
				this.videoStream.getTracks().forEach(track => track.stop());
				this.videoStream = null;
			}
			
			if (this.combinedStream) {
				this.combinedStream.getTracks().forEach(track => track.stop());
				this.combinedStream = null;
			}
			
			this.videoRecorderInitialized = false;
		},
		
		/**
		 * Check for new video recorded data
		 * @returns {boolean}
		 */
		hasNewVideoData: function() {
			const currentChunkCount = this.videoRecordedChunks.length;
			if (currentChunkCount !== this.lastCheckedChunkCount) {
				this.lastCheckedChunkCount = currentChunkCount;
				this.hasNewData = true;
				return true;
			}
			
			if (this.hasNewData) {
				this.hasNewData = false;
				return true;
			}
			
			return false;
		},
		
		/**
		 * Check if there is video recorded data
		 * @returns {boolean}
		 */
		hasVideoData: function() {
			return this.videoRecordedChunks.length > 0;
		},
	},
	
	// C++ interface functions
	godot_audio_recorder_init__proxy: 'sync',
	godot_audio_recorder_init__sig: 'i',
	/**
	 * Initialize Web audio recorder
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_audio_recorder_init: function() {
		return GodotAudioRecorder.init() ? 1 : 0;
	},
	
	godot_audio_recorder_start__proxy: 'sync', 
	godot_audio_recorder_start__sig: 'i',
	/**
	 * Start recording
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_audio_recorder_start: function() {
		return GodotAudioRecorder.startRecording() ? 1 : 0;
	},
	
	godot_audio_recorder_stop__proxy: 'sync',
	godot_audio_recorder_stop__sig: 'v',
	/**
	 * Stop recording
	 */
	godot_audio_recorder_stop: function() {
		GodotAudioRecorder.stopRecording();
	},
	
	godot_audio_recorder_is_recording__proxy: 'sync',
	godot_audio_recorder_is_recording__sig: 'i',
	/**
	 * Check if recording is active
	 * @returns {number} 1 if recording, 0 otherwise
	 */
	godot_audio_recorder_is_recording: function() {
		return GodotAudioRecorder.isActiveRecording() ? 1 : 0;
	},
	
	godot_audio_recorder_get_data_size__proxy: 'sync',
	godot_audio_recorder_get_data_size__sig: 'i',
	/**
	 * Get the size of recorded data (optimized version)
	 * @returns {number} Data size in bytes
	 */
	godot_audio_recorder_get_data_size: function() {
		// Fast path: if the number of chunks has not changed, return the cached size directly
		if (GodotAudioRecorder.lastChunkCount === GodotAudioRecorder.recordedChunks.length) {
			return GodotAudioRecorder.cachedDataSize;
		}
		
		// Recalculate only when there is new data
		const blob = GodotAudioRecorder.getRecordedAudioBlob();
		return blob ? blob.size : 0;
	},
	
	godot_audio_recorder_get_mime_type__proxy: 'sync',
	godot_audio_recorder_get_mime_type__sig: 'i',
	/**
	 * Get the MIME type string pointer of the recorded audio
	 * @returns {number} String pointer
	 */
	godot_audio_recorder_get_mime_type: function() {
		const mimeType = GodotAudioRecorder.getSupportedMimeType();
		return GodotRuntime.allocString(mimeType);
	},
	
	godot_audio_recorder_download_data__proxy: 'sync',
	godot_audio_recorder_download_data__sig: 'vi',
	/**
	 * Download recorded audio data (for testing)
	 * @param {number} filenamePtr Filename string pointer
	 */
	godot_audio_recorder_download_data: function(filenamePtr) {
		const blob = GodotAudioRecorder.getRecordedAudioBlob();
		if (!blob) {
			GodotRuntime.error('GodotAudioRecorder: No recorded data to download');
			return;
		}
		
		const filename = GodotRuntime.parseString(filenamePtr);
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = filename || 'recorded_audio.' + GodotAudioRecorder.getSupportedMimeType().split('/')[1].split(';')[0];
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
		
	},
	
	godot_audio_recorder_cleanup__proxy: 'sync',
	godot_audio_recorder_cleanup__sig: 'v',
	/**
	 * Clean up recorder resources
	 */
	godot_audio_recorder_cleanup: function() {
		GodotAudioRecorder.cleanup();
	},
	
	godot_audio_recorder_has_new_data__proxy: 'sync',
	godot_audio_recorder_has_new_data__sig: 'i',
	/**
	 * Lightweight check for new recorded data
	 * Much faster than get_data_size because it doesn't need to create a Blob object
	 * @returns {number} 1 if there is new data, 0 otherwise
	 */
	godot_audio_recorder_has_new_data: function() {
		return GodotAudioRecorder.hasNewRecordingData() ? 1 : 0;
	},
	
	godot_audio_recorder_has_data__proxy: 'sync',
	godot_audio_recorder_has_data__sig: 'i',
	/**
	 * Quick check for recorded data (lightweight, does not create Blob)
	 * @returns {number} 1 if there is data, 0 otherwise
	 */
	godot_audio_recorder_has_data: function() {
		return GodotAudioRecorder.hasRecordingData() ? 1 : 0;
	},
	
	// ========== New C++ interface functions for video recording ==========
	
	godot_video_recorder_init__proxy: 'sync',
	godot_video_recorder_init__sig: 'ii',
	/**
	 * Initialize Canvas video recorder (audio and video merged recording)
	 * @param {number} fps Video frame rate
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_video_recorder_init: function(fps) {
		return GodotAudioRecorder.initVideoRecorder(fps) ? 1 : 0;
	},
	
	godot_video_recorder_start__proxy: 'sync',
	godot_video_recorder_start__sig: 'i',
	/**
	 * Start video recording (audio and video merged)
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_video_recorder_start: function() {
		return GodotAudioRecorder.startVideoRecording() ? 1 : 0;
	},
	
	godot_video_recorder_stop__proxy: 'sync',
	godot_video_recorder_stop__sig: 'v',
	/**
	 * Stop video recording
	 */
	godot_video_recorder_stop: function() {
		GodotAudioRecorder.stopVideoRecording();
	},
	
	godot_video_recorder_is_recording__proxy: 'sync',
	godot_video_recorder_is_recording__sig: 'i',
	/**
	 * Check if video is being recorded
	 * @returns {number} 1 if recording, 0 otherwise
	 */
	godot_video_recorder_is_recording: function() {
		return GodotAudioRecorder.isRecording ? 1 : 0;
	},
	
	godot_video_recorder_get_data_size__proxy: 'sync',
	godot_video_recorder_get_data_size__sig: 'i',
	/**
	 * Get the size of recorded video data
	 * @returns {number} Data size in bytes
	 */
	godot_video_recorder_get_data_size: function() {
		const blob = GodotAudioRecorder.getRecordedVideoBlob();
		return blob ? blob.size : 0;
	},
	
	godot_video_recorder_get_mime_type__proxy: 'sync',
	godot_video_recorder_get_mime_type__sig: 'i',
	/**
	 * Get the MIME type string pointer of the recorded video
	 * @returns {number} String pointer
	 */
	godot_video_recorder_get_mime_type: function() {
		const mimeType = GodotAudioRecorder.selectedMimeType || 'video/webm';
		return GodotRuntime.allocString(mimeType);
	},
	
	godot_video_recorder_download_data__proxy: 'sync',
	godot_video_recorder_download_data__sig: 'vi',
	/**
	 * Download recorded video data
	 * @param {number} filenamePtr Filename string pointer
	 */
	godot_video_recorder_download_data: function(filenamePtr) {
		const filename = GodotRuntime.parseString(filenamePtr);
		GodotAudioRecorder.downloadRecordedVideo(filename);
	},
	
	godot_video_recorder_cleanup__proxy: 'sync',
	godot_video_recorder_cleanup__sig: 'v',
	/**
	 * Clean up video recorder resources
	 */
	godot_video_recorder_cleanup: function() {
		GodotAudioRecorder.cleanupVideoRecorder();
	},
	
	godot_video_recorder_has_new_data__proxy: 'sync',
	godot_video_recorder_has_new_data__sig: 'i',
	/**
	 * Check for new video recorded data
	 * @returns {number} 1 if there is new data, 0 otherwise
	 */
	godot_video_recorder_has_new_data: function() {
		return GodotAudioRecorder.hasNewVideoData() ? 1 : 0;
	},
	
	godot_video_recorder_has_data__proxy: 'sync',
	godot_video_recorder_has_data__sig: 'i',
	/**
	 * Check if there is video recorded data
	 * @returns {number} 1 if there is data, 0 otherwise
	 */
	godot_video_recorder_has_data: function() {
		return GodotAudioRecorder.hasVideoData() ? 1 : 0;
	},
};

// Add GodotAudioRecorder to GodotAudio object
if (typeof GodotAudio !== 'undefined') {
	GodotAudio.Recorder = GodotAudioRecorder;
}

autoAddDeps(GodotAudioRecorder, '$GodotAudioRecorder');
mergeInto(LibraryManager.library, GodotAudioRecorder);

/**
 * Web file download utility functions
 */
const GodotWebDownload = {
	$GodotWebDownload: {
		/**
		 * Download recorded audio data
		 * @param {string} filename Filename
		 */
		downloadRecordedAudio: function(filename) {
			const blob = GodotAudioRecorder.getRecordedAudioBlob();
			if (!blob) {
				GodotRuntime.error('GodotWebDownload: No recorded audio data to download');
				return false;
			}
			
			this.downloadBlob(blob, filename || 'recorded_audio.webm');
			return true;
		},
		
		/**
		 * Generic file download function
		 * @param {Blob} blob Data to download
		 * @param {string} filename Filename
		 */
		downloadBlob: function(blob, filename) {
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename;
			a.style.display = 'none';
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
			
		},
		
		/**
		 * Read data from file path and download (Web's virtual file system)
		 * @param {string} filePath File path
		 * @param {string} downloadName Filename for download
		 */
		downloadFile: function(filePath, downloadName) {
			try {
				// On the web, files are usually stored in Emscripten's virtual file system
				// We need to read the file content and create a download
				const fs = FS;
				if (!fs.analyzePath(filePath).exists) {
					GodotRuntime.error('GodotWebDownload: File does not exist: ' + filePath);
					return false;
				}
				
				const data = fs.readFile(filePath);
				const blob = new Blob([data], { type: 'application/octet-stream' });
				
				// Use original filename if no download name is specified
				const filename = downloadName || filePath.split('/').pop();
				this.downloadBlob(blob, filename);
				
				return true;
			} catch (error) {
				GodotRuntime.error('GodotWebDownload: Failed to download file: ' + error.message);
				return false;
			}
		}
	},
	
	// C++ interface functions
	godot_web_download_recorded_audio__proxy: 'sync',
	godot_web_download_recorded_audio__sig: 'ii',
	/**
	 * Download recorded audio data
	 * @param {number} filenamePtr Filename string pointer
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_web_download_recorded_audio: function(filenamePtr) {
		const filename = GodotRuntime.parseString(filenamePtr);
		return GodotWebDownload.downloadRecordedAudio(filename) ? 1 : 0;
	},
	
	godot_web_download_file__proxy: 'sync',
	godot_web_download_file__sig: 'iii',
	/**
	 * Download a file from the specified path
	 * @param {number} filePathPtr File path string pointer
	 * @param {number} downloadNamePtr Download filename string pointer (can be 0)
	 * @returns {number} 1 on success, 0 on failure
	 */
	godot_web_download_file: function(filePathPtr, downloadNamePtr) {
		const filePath = GodotRuntime.parseString(filePathPtr);
		const downloadName = downloadNamePtr ? GodotRuntime.parseString(downloadNamePtr) : null;
		return GodotWebDownload.downloadFile(filePath, downloadName) ? 1 : 0;
	}
};

autoAddDeps(GodotWebDownload, '$GodotWebDownload');
mergeInto(LibraryManager.library, GodotWebDownload);
