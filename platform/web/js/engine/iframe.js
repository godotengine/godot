const EngineIFrame = (function () {
	const iframeScript = function (name) {
		// IMPORTANT!!! This function will run in the iframe document, not in the current context.
		const canvas = document.getElementById('godot-canvas');
		let engine = null;
		let iframe_port = null;
		let pid = 0;
		function handleMessage(event) {
			/**
			 * @param {string} cmd
			 * @param {*=} msg
			 * @ignore
			 */
			function notify(cmd, msg) {
				if (!iframe_port) {
					return;
				}
				iframe_port.postMessage({ 'cmd': cmd, 'msg': msg, 'pid': pid });
			}
			const cmd = event.data['cmd'] || '';
			const msg = event.data['msg'] || '';
			if (cmd == 'start') {
				if (engine) {
					throw new Error('An instance is already running.');
				}
				if (event.ports.length < 1) {
					throw new Error('Not enough message ports received, cannot start.');
				}
				iframe_port = event.ports[0];
				const config = {
					...msg,
					'debugPort': event.ports[1] ?? null,
					'canvas': canvas,
					'onProgress': (current, total) => {
						notify('progress', [current, total]);
					},
					'onExecute': (path, args) => {
						const newPid = Math.round(Math.random() * (1 << 30));
						notify('execute', [newPid, path, args]);
						return newPid;
					},
					'onTerminatePID': (p_pid) => {
						notify('terminate', p_pid);
					},
					'onExit': () => {
						engine = null;
						notify('stopped');
						pid = 0;
					},
				};
				pid = config['pid'] ?? 0;
				engine = new Engine(config); // eslint-disable-line no-undef
				const args = config['args'] ?? [];
				const is_editor = args.indexOf((e) => e === '-e' || e === '--editor' || e === '--project-manager');
				engine.init(name).then(function () {
					if (is_editor) {
						try {
							// Avoid user creating project in the persistent root folder.
							engine.copyToFS('/home/web_user/keep', new Uint8Array(0));
						} catch (e) {
							// File exists
						}
					}
					requestAnimationFrame(function () {
						engine.start().then(function () {
							notify('started');
						});
					});
				});
			} else if (cmd == 'session') {
				if (event.ports.length < 1) {
					throw new Error('No enough message ports received, cannot start.');
				}
				engine.addDebuggerSession(event.ports[0]);
			} else if (cmd == 'stop') {
				if (!engine) {
					return;
				}
				engine.requestQuit();
			} else if (cmd == 'focus') {
				canvas.focus();
			} else {
				console.error('Invalid event', event); // eslint-disable-line no-console
			}
		}
		const loadPromise = new Promise((accept, reject) => {
			const s = document.createElement('script');
			s.src = `${name}.js`;
			s.onload = () => {
				accept();
			};
			document.body.appendChild(s);
		});
		window.onmessage = (event) => {
			loadPromise.then(() => {
				handleMessage(event);
			});
		};
	};

	function replaceIframe(name, from, styleInnerHTML) {
		const iframe = document.createElement('iframe');
		iframe.src = `${name}.iframe.html`;
		iframe.id = from.id;
		iframe.className = from.className;
		iframe.tabIndex = from.tabIndex;
		iframe.style.width = from.style.width;
		iframe.style.height = from.style.height;
		let resolve = null;
		iframe.onload = () => {
			const style = iframe.contentDocument.createElement('style');
			style.innerHTML = styleInnerHTML;
			iframe.contentDocument.body.appendChild(style);
			const canvas = iframe.contentDocument.createElement('canvas');
			canvas.id = 'godot-canvas';
			iframe.contentDocument.body.appendChild(canvas);
			// Add load/start script
			const source = `const start = ${iframeScript.toString()}; start(${JSON.stringify(name)});`;
			const script = iframe.contentDocument.createElement('script');
			script.type = 'application/javascript';
			script.appendChild(iframe.contentDocument.createTextNode(source));
			iframe.contentDocument.body.appendChild(script);
			resolve();
		};
		return [
			iframe,
			new Promise((accept, reject) => {
				resolve = accept;
				from.parentNode.replaceChild(iframe, from);
			}),
		];
	}

	/**
	 * @classdesc The ``EngineIFrame`` class provides methods for embedding an ``Engine`` instance into an iframe.
	 *
	 * .. attention::
	 *
	 *     This class is experimental and may change in future versions.
	 *
	 * @description Creates a new EngineIFrame instance with the given configuration.
	 * @global
	 * @constructor
	 * @param {string} name The name to pass to Engine.init.
	 * @param {HTMLIFrameElement} iframe The iframe element to replace.
	 * @param {HTMLStyleElement=} style Custom CSS rules to add to the iframe.
	 * @return {EngineIFrame}
	 */
	function EngineIFrame(name, iframe, style) {} // eslint-disable-line no-unused-vars, no-shadow
	const proto = (replace) => (/** @lends EngineIFrame.prototype */{
		/**
		 * The currently bound iframe.
		 * @export
		 * @type {HTMLIFrameElement}
		 */
		iframe: null,
		/**
		 * The Process ID of the current instance.
		 * @export
		 * @type {number}
		 */
		pid: 0,
		/**
		 * @ignore
		 * @export
		 * @type {?MessageChannel}
		 */
		iframeChannel: null,
		/**
		 * @ignore
		 * @export
		 * @type {?Array<MessagePort>}
		 */
		debugPorts: null,
		/**
		 * Starts a new instance with the given config.
		 * @export
		 * @param {EngineConfig} config The engine configuration to use when starting the engine. Function arguments are not permitted here.
		 * @param {function(Event)} onmessage The onmessage callback where to listen for iframe messages.
		 * @param {boolean=} debug If the instance should be started with a debug port.
		 */
		start: function (config, onmessage, debug) {
			this.pid = config.pid ?? Math.round(Math.random() * (1 << 30));
			this.iframeChannel = new MessageChannel();
			this.iframeChannel.port1.onmessage = onmessage;
			const transfers = [this.iframeChannel.port2];
			let debugSession = null;
			if (debug) {
				// Proxy the debug port so we want can send a 'close'
				// message even if we forcibly terminate the iframe.
				const debugChannel = new MessageChannel();
				const debugProxy = new MessageChannel();
				debugChannel.port1.onmessage = (evt) => debugProxy.port1.postMessage(evt.data);
				debugProxy.port1.onmessage = (evt) => debugChannel.port1.postMessage(evt.data);
				debugSession = debugProxy.port2;
				this.debugPorts = [debugProxy.port1, debugChannel.port1];
				transfers.push(debugChannel.port2);
			}
			this.iframe.contentWindow.postMessage({ 'cmd': 'start', 'msg': config }, this.iframe.contentWindow.origin, transfers);
			return debugSession;
		},
		/**
		 * Sends a stop request to the currently running instance.
		 * @export
		 */
		stop: function () {
			this.iframe.contentWindow.postMessage({ 'cmd': 'stop' });
		},
		/**
		 * Focus the wrapped iframe and notify the instance.
		 * @export
		 */
		focus: function () {
			this.iframe.focus();
			this.iframe.contentWindow.postMessage({ 'cmd': 'focus' });
		},
		/**
		 * Transfers a debug session to attach to the editor debugger (this must be an editor instance).
		 * @export
		 * @param {MessagePort} port The MessagePort to connect to the editor debugger.
		 */
		addSession: function (port) {
			this.iframe.contentWindow.postMessage({ 'cmd': 'session', 'msg': 0 }, this.iframe.contentWindow.origin, [port]);
		},
		/**
		 * Forcibly resets the state by replacing the iframe with a new one.
		 * @export
		 * @return {Promise} A Promise upon which fullfillment a new instance can be started.
		 */
		reset: function () {
			if (this.iframeChannel) {
				this.iframeChannel.port1.close();
				this.iframeChannel.port1.onmessage = null;
			}
			if (this.debugPorts) {
				this.debugPorts.forEach((port) => {
					// Make sure we always send a 'stop' message even if we
					// forcibly terminate the game.
					port.postMessage('stop');
					port.close();
					port.onmessage = null;
				});
			};
			const [next, promise] = replace(this.iframe);
			this.iframe = next;
			this.pid = 0;
			this.iframeChannel = null;
			this.debugPorts = null;
			return promise;
		},
	});
	/**
	 * @ignore
	 * @constructor
	 * @return {EngineIFrame}
	 */
	function SafeIframe(name, reference, style) {
		/**
		 * @ignore
		 * @constructor
		 * @return {EngineIFrame}
		 */
		function EngineIFrame(iframe) { // eslint-disable-line no-shadow
			this.iframe = iframe;
			this.reset();
		}
		const styleHTML = style ? style.innerHTML : '';
		EngineIFrame.prototype = proto((iframe) => replaceIframe(name, iframe, styleHTML));
		return new EngineIFrame(reference);
	};
	return SafeIframe;
})();
if (typeof window !== 'undefined') {
	window['EngineIFrame'] = EngineIFrame;
}
