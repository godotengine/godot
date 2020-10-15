/*************************************************************************/
/*  utils.js                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

Module['initFS'] = function(persistentPaths) {
	Module.mount_points = ['/userfs'].concat(persistentPaths);

	function createRecursive(dir) {
		try {
			FS.stat(dir);
		} catch (e) {
			if (e.errno !== ERRNO_CODES.ENOENT) {
				throw e;
			}
			FS.mkdirTree(dir);
		}
	}

	Module.mount_points.forEach(function(path) {
		createRecursive(path);
		FS.mount(IDBFS, {}, path);
	});
	return new Promise(function(resolve, reject) {
		FS.syncfs(true, function(err) {
			if (err) {
				Module.mount_points = [];
				Module.idbfs = false;
				console.log("IndexedDB not available: " + err.message);
			} else {
				Module.idbfs = true;
			}
			resolve(err);
		});
	});
};

Module['deinitFS'] = function() {
	Module.mount_points.forEach(function(path) {
		try {
			FS.unmount(path);
		} catch (e) {
			console.log("Already unmounted", e);
		}
		if (Module.idbfs && IDBFS.dbs[path]) {
			IDBFS.dbs[path].close();
			delete IDBFS.dbs[path];
		}
	});
	Module.mount_points = [];
};

Module['copyToFS'] = function(path, buffer) {
	var p = path.lastIndexOf("/");
	var dir = "/";
	if (p > 0) {
		dir = path.slice(0, path.lastIndexOf("/"));
	}
	try {
		FS.stat(dir);
	} catch (e) {
		if (e.errno !== ERRNO_CODES.ENOENT) {
			throw e;
		}
		FS.mkdirTree(dir);
	}
	// With memory growth, canOwn should be false.
	FS.writeFile(path, new Uint8Array(buffer), {'flags': 'wx+'});
}

Module.drop_handler = (function() {
	var upload = [];
	var uploadPromises = [];
	var uploadCallback = null;

	function readFilePromise(entry, path) {
		return new Promise(function(resolve, reject) {
			entry.file(function(file) {
				var reader = new FileReader();
				reader.onload = function() {
					var f = {
						"path": file.relativePath || file.webkitRelativePath,
						"name": file.name,
						"type": file.type,
						"size": file.size,
						"data": reader.result
					};
					if (!f['path'])
						f['path'] = f['name'];
					upload.push(f);
					resolve()
				};
				reader.onerror = function() {
					console.log("Error reading file");
					reject();
				}

				reader.readAsArrayBuffer(file);

				}, function(err) {
					console.log("Error!");
					reject();
				});
		});
	}

	function readDirectoryPromise(entry) {
		return new Promise(function(resolve, reject) {
			var reader = entry.createReader();
			reader.readEntries(function(entries) {
				for (var i = 0; i < entries.length; i++) {
					var ent = entries[i];
					if (ent.isDirectory) {
						uploadPromises.push(readDirectoryPromise(ent));
					} else if (ent.isFile) {
						uploadPromises.push(readFilePromise(ent));
					}
				}
				resolve();
			});
		});
	}

	function processUploadsPromises(resolve, reject) {
		if (uploadPromises.length == 0) {
			resolve();
			return;
		}
		uploadPromises.pop().then(function() {
			setTimeout(function() {
				processUploadsPromises(resolve, reject);
				//processUploadsPromises.bind(null, resolve, reject)
			}, 0);
		});
	}

	function dropFiles(files) {
		var args = files || [];
		var argc = args.length;
		var argv = stackAlloc((argc + 1) * 4);
		for (var i = 0; i < argc; i++) {
			HEAP32[(argv >> 2) + i] = allocateUTF8OnStack(args[i]);
		}
		HEAP32[(argv >> 2) + argc] = 0;
		// Defined in display_server_javascript.cpp
		ccall('_drop_files_callback', 'void', ['number', 'number'], [argv, argc]);
	}

	return function(ev) {
		ev.preventDefault();
		if (ev.dataTransfer.items) {
			// Use DataTransferItemList interface to access the file(s)
			for (var i = 0; i < ev.dataTransfer.items.length; i++) {
				const item = ev.dataTransfer.items[i];
				var entry = null;
				if ("getAsEntry" in item) {
					entry = item.getAsEntry();
				} else if ("webkitGetAsEntry" in item) {
					entry = item.webkitGetAsEntry();
				}
				if (!entry) {
					console.error("File upload not supported");
				} else if (entry.isDirectory) {
					uploadPromises.push(readDirectoryPromise(entry));
				} else if (entry.isFile) {
					uploadPromises.push(readFilePromise(entry));
				} else {
					console.error("Unrecognized entry...", entry);
				}
			}
		} else {
			console.error("File upload not supported");
		}
		uploadCallback = new Promise(processUploadsPromises).then(function() {
			const DROP = "/tmp/drop-" + parseInt(Math.random() * Math.pow(2, 31)) + "/";
			var drops = [];
			var files = [];
			upload.forEach((elem) => {
				var path = elem['path'];
				Module['copyToFS'](DROP + path, elem['data']);
				var idx = path.indexOf("/");
				if (idx == -1) {
					// Root file
					drops.push(DROP + path);
				} else {
					// Subdir
					var sub = path.substr(0, idx);
					idx = sub.indexOf("/");
					if (idx < 0 && drops.indexOf(DROP + sub) == -1) {
						drops.push(DROP + sub);
					}
				}
				files.push(DROP + path);
			});
			uploadPromises = [];
			upload = [];
			dropFiles(drops);
			var dirs = [DROP.substr(0, DROP.length -1)];
			files.forEach(function (file) {
				FS.unlink(file);
				var dir = file.replace(DROP, "");
				var idx = dir.lastIndexOf("/");
				while (idx > 0) {
					dir = dir.substr(0, idx);
					if (dirs.indexOf(DROP + dir) == -1) {
						dirs.push(DROP + dir);
					}
					idx = dir.lastIndexOf("/");
				}
			});
			// Remove dirs.
			dirs = dirs.sort(function(a, b) {
				var al = (a.match(/\//g) || []).length;
				var bl = (b.match(/\//g) || []).length;
				if (al > bl)
					return -1;
				else if (al < bl)
					return 1;
				return 0;
			});
			dirs.forEach(function(dir) {
				FS.rmdir(dir);
			});
		});
	}
})();

function EventHandlers() {
	function Handler(target, event, method, capture) {
		this.target = target;
		this.event = event;
		this.method = method;
		this.capture = capture;
	}

	var listeners = [];

	function has(target, event, method, capture) {
		return listeners.findIndex(function(e) {
			return e.target === target && e.event === event && e.method === method && e.capture == capture;
		}) !== -1;
	}

	this.add = function(target, event, method, capture) {
		if (has(target, event, method, capture)) {
			return;
		}
		listeners.push(new Handler(target, event, method, capture));
		target.addEventListener(event, method, capture);
	};

	this.remove = function(target, event, method, capture) {
		if (!has(target, event, method, capture)) {
			return;
		}
		target.removeEventListener(event, method, capture);
	};

	this.clear = function() {
		listeners.forEach(function(h) {
			h.target.removeEventListener(h.event, h.method, h.capture);
		});
		listeners.length = 0;
	};
}

Module.listeners = new EventHandlers();
