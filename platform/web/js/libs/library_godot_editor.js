/**************************************************************************/
/*  library_godot_editor.js                                               */
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

const GodotEditorInstall = {
	$GodotEditorInstall__deps: ['$FS', '$GodotFS'],
	$GodotEditorInstall: {
		source_root: '',
		promises: [],
		pending_files: [],
		total: 0,

		add_file: function (file) {
			GodotEditorInstall.promises.push(new Promise(function (resolve, reject) {
				const reader = new FileReader();
				GodotEditorInstall.total += 1;
				reader.onload = function () {
					const path = file.webkitRelativePath;
					const relativePath = path.startsWith(GodotEditorInstall.source_root) ? path.slice(GodotEditorInstall.source_root.length) : path;
					const f = {
						'path': relativePath,
						'name': file.name,
						'type': file.type,
						'size': file.size,
						'data': reader.result,
					};
					if (!f['path']) {
						f['path'] = f['name'];
					}
					GodotEditorInstall.pending_files.push(f);
					resolve();
				};
				reader.onerror = function () {
					GodotRuntime.print(`Error reading file ${file.webkitRelativePath}`);
					reject();
				};
				reader.readAsArrayBuffer(file);
			}));
		},

		process: function (resolve, reject) {
			if (GodotEditorInstall.promises.length === 0) {
				resolve();
				return;
			}
			GodotEditorInstall.promises.pop().then(function () {
				setTimeout(function () {
					GodotEditorInstall.process(resolve, reject);
				}, 0);
			});
		},

		install_folder: function (install_path, source_relative_path, files) {
			if (files) {
				// Use DataTransferItemList interface to access the file(s)
				GodotEditorInstall.source_root = source_relative_path;
				for (let i = 0; i < files.length; i++) {
					const item = files[i];
					GodotEditorInstall.add_file(item);
				}
			} else {
				GodotRuntime.error('File upload not supported');
			}
			return new Promise(GodotEditorInstall.process).then(function () {
				const file_paths = [];
				const INSTALL_ROOT = install_path;
				FS.mkdir(INSTALL_ROOT.slice(0, -1)); // Without trailing slash
				GodotEditorInstall.pending_files.forEach((elem) => {
					const path = elem['path'];
					GodotFS.copy_to_fs(INSTALL_ROOT + path, elem['data']);
					let idx = path.indexOf('/');
					if (idx === -1) {
						// Root file
						file_paths.push(INSTALL_ROOT + path);
					} else {
						// Subdir
						const sub = path.substr(0, idx);
						idx = sub.indexOf('/');
						if (idx < 0 && file_paths.indexOf(INSTALL_ROOT + sub) === -1) {
							file_paths.push(INSTALL_ROOT + sub);
						}
					}
				});
				GodotEditorInstall.promises = [];
				GodotEditorInstall.pending_files = [];
				GodotEditorInstall.total = 0;
			});
		},
	},
};
mergeInto(LibraryManager.library, GodotEditorInstall);

const GodotEditor = {
	$GodotEditor__deps: ['$GodotRuntime', '$GodotEditorInstall'],
	$GodotEditor: {
		utils: {
			toSnakeCase: function (str) {
				return str
				// Match words (sequences of letters/numbers)
					.match(/[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+/g)
				// Convert each word to lowercase
					.map((word) => word.toLowerCase())
				// Connect them with underscores
					.join('_');
			},
		},
	},

	godot_js_editor_show_open_project_dialog__proxy: 'sync',
	godot_js_editor_show_open_project_dialog__sig: 'vi',
	godot_js_editor_show_open_project_dialog: function (p_on_done_cb) {
		const on_done = GodotRuntime.get_func(p_on_done_cb);

		const RESULT_CANCELLED = 0;
		const RESULT_INVALID = 1;
		const RESULT_SUCCESS = 2;

		const file_selector = document.createElement('input');
		file_selector.type = 'file';
		file_selector.setAttribute('multiple', '');
		file_selector.webkitdirectory = true;

		file_selector.onchange = function (event) {
			function read_text_file(file) {
				return new Promise(function (resolve, reject) {
					const reader = new FileReader();
					reader.onload = function () {
						resolve(reader.result);
					};
					reader.onerror = function (err) {
						reject(err);
					};
					reader.readAsText(file);
				});
			}

			const extract_project_name = function (project_godot_file) {
				return read_text_file(project_godot_file).then(function (text) {
					const lines = text.split('\n');
					for (let l = 0; l < lines.length; l++) {
						const line = lines[l];
						if (line.startsWith('config/name=')) {
							return line.split('"')[1];
						}
					}
					return '';
				});
			};

			const validate_project_folder = function (files) {
				for (let f = 0; f < files.length; f++) {
					const file = files[f];
					if (file.name !== 'project.godot' || file.webkitRelativePath.split('/').length != 2) {
						continue;
					}
					return extract_project_name(file).then(function (name) {
						if (!name) {
							return Promise.reject(new Error('Selected path is not a godot project.'));
						}
						const loc = file.webkitRelativePath ?? '';
						return Promise.resolve([name, loc.includes('/') ? loc.split('/').slice(0, -1).join('/') : '']);
					});
				};
				return Promise.reject(new Error('Selected path is not a godot project.'));
			};

			let install_path = '';
			validate_project_folder(event.target.files).then(([project_name, project_location]) => {
				install_path = `/home/web_user/${GodotEditor.utils.toSnakeCase(project_name)}`;
				return GodotEditorInstall.install_folder(`${install_path}/`, project_location, event.target.files);
			}).then(() => {
				const cstr = GodotRuntime.allocString(install_path);
				on_done(RESULT_SUCCESS, cstr);
				GodotRuntime.free(cstr);
			}).catch((error) => {
				const cstr = GodotRuntime.allocString(error.message);
				on_done(RESULT_INVALID, cstr);
				GodotRuntime.free(cstr);
			});
		};
		file_selector.oncancel = function (event) {
			const cstr = GodotRuntime.allocString('');
			on_done(RESULT_CANCELLED, cstr);
			GodotRuntime.free(cstr);
		};
		file_selector.click();
	},

	godot_js_editor_show_import_project_zip_dialog__proxy: 'sync',
	godot_js_editor_show_import_project_zip_dialog__sig: 'vi',
	godot_js_editor_show_import_project_zip_dialog: function (p_on_done_cb) {
		const on_done = GodotRuntime.get_func(p_on_done_cb);

		const RESULT_CANCELLED = 0;
		const RESULT_INVALID = 1;
		const RESULT_SUCCESS = 2;

		const file_selector = document.createElement('input');
		file_selector.type = 'file';
		file_selector.accept = '.zip';

		file_selector.onchange = function (event) {
			if (event.target.files.length == 1) {
				const file = event.target.files[0];
				const reader = new FileReader();
				reader.onload = function () {
					const data = reader.result;
					const temp_dir = `/tmp/zip-${parseInt(Math.random() * (1 << 30), 10)}`;
					FS.mkdir(temp_dir);
					const temp_file = `${temp_dir}/${file.name}`;
					GodotFS.copy_to_fs(temp_file, data);
					const gd_install_path = GodotRuntime.allocString(temp_file);
					on_done(RESULT_SUCCESS, gd_install_path);
					GodotRuntime.free(gd_install_path);
					// keep the zip until project manager closes, so it can be used to install the project
					GodotOS.atexit(function (resolve, reject) {
						FS.unlink(temp_file);
						FS.rmdir(temp_dir);
						resolve();
					});
				};
				reader.onerror = function () {
					const empty_path = GodotRuntime.allocString('Error reading file');
					on_done(RESULT_INVALID, empty_path);
					GodotRuntime.free(empty_path);
				};
				reader.readAsArrayBuffer(file);
			} else {
				file_selector.oncancel();
			}
		};
		file_selector.oncancel = function (event) {
			const empty_path = GodotRuntime.allocString('');
			on_done(RESULT_CANCELLED, empty_path);
			GodotRuntime.free(empty_path);
		};
		file_selector.click();
	},
};

autoAddDeps(GodotEditor, '$GodotEditor');
mergeInto(LibraryManager.library, GodotEditor);
