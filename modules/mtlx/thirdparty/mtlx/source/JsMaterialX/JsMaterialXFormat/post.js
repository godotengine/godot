//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

// Wrapping code in an anonymous function to prevent clashes with the main module and other pre / post JS.
(function () {
    var nodeFs;                     // Required NodeJS packages
    var nodePath;
    var nodeProcess;
    var pathSep;                    // The path separator, depending on the environment
    var wasmPathSep = "/";          // The path separator in the WASM FS
    var ENVIRONMENT_IS_WEB;         // True if this code runs in a browser
    var ENVIRONMENT_IS_NODE;        // True if this code runs in NodeJS
    var PATH_LIST_SEPARATOR = ";";  // The separator symbol for search paths
    var callId = 0;                 // Gets appended to the WASM root folder, to separate concurrent function calls
    var MAX_CALL_ID = 99999;

    // Remove duplicate entries from an array.
    function removeDuplicates(array) {
        var seen = {};
        return array.filter(function(item) {
            return seen.hasOwnProperty(item) ? false : (seen[item] = true);
        });
    }

    // Concatenate filename and path with the correct amount of slashes.
    function createFilePath(fileName, filePath, sep = pathSep) {
        var pathSlash = filePath.endsWith(sep);
        var fileSlash = fileName.startsWith(sep);
        var path;
        if (pathSlash || fileSlash) {
            if (pathSlash && fileSlash) {
                path = filePath.substring(0, filePath.length - 1) + fileName;
            } else {
                path = filePath + fileName;
            }
        } else {
            path = filePath + sep + fileName;
        }
        return path;
    }

    // Fetch a MaterialX definition file using the fetch browser API.
    function fetchXml(fileName, searchPaths) {
        var i = 0;
        function fetchHandler() {
            var filePath = createFilePath(fileName, searchPaths[i++]);
            return fetch(filePath).then(function (response) {
                if (response.status === 200) {
                    return response.text().then(function (data) {
                        var url = new URL(response.url);
                        var filePath = url.pathname.substring(1);
                        filePath = filePath.replace(new RegExp(pathSep, "g"), wasmPathSep);
                        return {
                            data: data,                         // file content
                            filePath: filePath,                 // file path relative to root
                            fullPath: url.origin + url.pathname,// the full url
                        };
                    });
                } else if (i < searchPaths.length) {
                    return fetchHandler();
                } else {
                    throw new Error("MaterialX file not found: " + fileName);
                }
            });
        }

        return fetchHandler();
    }

    // Fetch a MaterialX definition file using Node's fs API.
    function loadXml(fileName, searchPaths) {
        var i = 0;
        function loadHandler() {
            var filePath = createFilePath(fileName, searchPaths[i++]);
            filePath = nodePath.resolve(filePath);
            return new Promise(function (resolve, reject) {
                nodeFs.readFile(filePath, "utf8", function (err, data) {
                    if (err) {
                        if (i < searchPaths.length) {
                            resolve(loadHandler());
                        } else {
                            reject(new Error("MaterialX file not found: " + fileName));
                        }
                    } else {
                        var parsedPath = nodePath.parse(filePath);
                        var path = filePath.substring(parsedPath.root.length);
                        var sep = pathSep === "\\" ? "\\\\" : pathSep;
                        path = path.replace(new RegExp(sep, "g"), wasmPathSep);
                        resolve({
                            data: data,         // file content
                            filePath: path,     // file path relative to root
                            fullPath: filePath, // the full absolute path

                        });
                    }
                });
            });
        }

        return loadHandler();
    }

    // Concatenate searchPath, MATERIALX_SEARCH_PATH_ENV_VAR and './' and clean up the list.
    function prepareSearchPaths(searchPath) {
        var newSearchPath = searchPath.concat(PATH_LIST_SEPARATOR,
            Module.getEnviron(Module.MATERIALX_SEARCH_PATH_ENV_VAR));
        var searchPaths = ["." + pathSep].concat(newSearchPath.split(PATH_LIST_SEPARATOR));
        var i = searchPaths.length - 1;
        while (i) {
            if (searchPaths[i].trim() === "") {
                searchPaths.splice(i, 1);
            }
            --i;
        }
        return removeDuplicates(searchPaths);
    }

    // Modify absolute paths so that they work in the WASM file system.
    function makeWasmAbsolute(paths, wasmRootFolder) {
        var pathList = paths;
        if (typeof paths === 'string') {
            pathList = paths.split(PATH_LIST_SEPARATOR);
        }
        // Rewrite absolute paths to be absolute in the WASM FS.
        for (var i = 0; i < pathList.length; ++i) {
            var path = pathList[i];
            if (ENVIRONMENT_IS_NODE) {
                if (nodePath.isAbsolute(path)) {
                    var parsed = nodePath.parse(path);
                    path = wasmRootFolder + wasmPathSep + parsed.dir.substring(parsed.root.length);
                }
                var sep = pathSep === "\\" ? "\\\\" : pathSep;
                path.replace(new RegExp(sep, "g"), wasmPathSep);
            } else if (ENVIRONMENT_IS_WEB) {
                var link = document.createElement("a");
                link.href = path;
                if (link.origin + link.pathname + link.search + link.hash === path) {
                    path = wasmRootFolder + link.pathname;
                }
            } else {
                throw new Error("Unknown environment!");
            }
            pathList[i] = path;
        }
        if (typeof paths === 'string') {
            return pathList.join(Module.PATH_LIST_SEPARATOR);
        } else {
            return pathList;
        }
    }

    // Tweak the user-provided search path so that it works in the WASM FS.
    function getWasmSearchPath(searchPath, wasmRootFolder) {
        var wasmSearchPath = searchPath.split(PATH_LIST_SEPARATOR);
        makeWasmAbsolute(wasmSearchPath, wasmRootFolder);
        wasmSearchPath.push(getWasmCwd(wasmRootFolder));
        wasmSearchPath = removeDuplicates(wasmSearchPath);
        wasmSearchPath = wasmSearchPath.join(Module.PATH_LIST_SEPARATOR);
        return wasmSearchPath;
    }

    // Parse a file to get includes.
    function getIncludes(file) {
        var includeRegex = /<xi:include href="(.*)"\s*\/>/g;
        var matches = file.matchAll(includeRegex);
        var includes = [];
        for (var match of matches) {
            includes.push(match[1]);
        }
        return includes;
    }

    // Load a file depending on the environment.
    function loadFile(fileToLoad, searchPaths) {
        var promise;
        if (ENVIRONMENT_IS_WEB) {
            promise = fetchXml(fileToLoad, searchPaths);
        } else if (ENVIRONMENT_IS_NODE) {
            promise = loadXml(fileToLoad, searchPaths);
        } else {
            throw new Error("Unknown environment!");
        }
        return promise;
    }

    // Track folders and files that have been created in the WASM file system, to delete them again later.
    function trackPath(path, filesUploaded, isFile = false) {
        if (isFile) {
        // Remember full file path
            if (!filesUploaded.files) {
                filesUploaded.files = [];
            }
            filesUploaded.files.push(path);
        } else {
            // Remember paths in inverse order of creation, to delete them again later.
            if (!filesUploaded.folders) {
                filesUploaded.folders = [];
            }
            filesUploaded.folders.splice(0, 0, path);
        }
    }

    // Store a file in the WASM file system. File is expected to be an absolute path, starting with wasmRootFolder.
    function createInWasm(file, data, filesUploaded, wasmRootFolder, isFile = true) {
        // Create folders if necessary.
        var folders;
        if (isFile) {
           folders = file.substring(1, file.lastIndexOf(wasmPathSep)).split(wasmPathSep);
        } else {
            folders = file.substring(wasmRootFolder.length).split(wasmPathSep);
        }
        var folder = wasmRootFolder;
        // Skipping 0 because it's the root folder, which we already created.
        for (var i = 1; i < folders.length; ++i) {
            folder += wasmPathSep + folders[i];
            var dirExists;
            try {
                var stat = FS.stat(folder);
                dirExists = FS.isDir(stat.mode);
            } catch (e) {
                dirExists = false;
            }
            if (!dirExists) {
                try {
                    FS.mkdir(folder);
                    trackPath(folder, filesUploaded);
                } catch (e) {
                    throw new Error("Failed to create folder in WASM FS.");
                }
            }
        }
        // Store the file.
        if (isFile) {
            try {
                FS.writeFile(file, data);
                trackPath(file, filesUploaded, true);
            } catch (e) {
                throw new Error("Failed to store file in WASM FS.");
            }
        }
    }

    // Returns the current working directory, in the WASM FS.
    function getWasmCwd(wasmRootFolder) {
        if (ENVIRONMENT_IS_NODE) {
            var cwd = nodeProcess.cwd();
            var parsed = nodePath.parse(cwd);
            var wasmCwd = wasmRootFolder + wasmPathSep + cwd.substring(parsed.root.length);
            var sep = pathSep === "\\" ? "\\\\" : pathSep;
            return wasmCwd.replace(new RegExp(sep, "g"), wasmPathSep);
        } else if (ENVIRONMENT_IS_WEB) {
            var cwd = window.location.pathname;
            cwd = cwd.substring(0, cwd.lastIndexOf(pathSep));
            return createFilePath(cwd, wasmRootFolder, wasmPathSep);
        } else {
            throw new Error("Unknown environment!");
        }
    }

    // Store a file to disk or download it in the browser
    function storeFileToDisk(fileName, content) {
        if (ENVIRONMENT_IS_NODE) {
            // Write file to local file system
            try {
                nodeFs.writeFileSync(fileName, content);
            } catch (e) {
                throw new Error("Failed to write file '" + fileName + "': " + e.message);
            }
        } else if (ENVIRONMENT_IS_WEB) {
            // Only take the name of the file (fileName might be a path)
            var pos = fileName.lastIndexOf(pathSep);
            fileName = fileName.substring(pos > -1 ? pos + 1 : 0);
            // Download file in the browser
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(content));
            element.setAttribute('download', fileName);
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }
    }

    onModuleReady(function () {
        // Determine environment and load dependencies as required.
        ENVIRONMENT_IS_WEB = typeof window === "object";
        ENVIRONMENT_IS_NODE =
            typeof process === "object" &&
            typeof process.versions === "object" &&
            typeof process.versions.node === "string";
        if (ENVIRONMENT_IS_WEB) {
            pathSep = "/";
        }
        if (ENVIRONMENT_IS_NODE) {
            nodeFs = require("fs");
            nodePath = require("path");
            nodeProcess = require("process");
            pathSep = nodePath.sep;
        }

        /**
         * Internal method that is called from the public methods to carry out the actual work.
         * There are several reasons for having a custom implementation in JS:
         * - We want to support both NodeJS and browsers, so we need different implementations for reading files.
         * - The C++ implementation assumes files to be stored in a filesystem. That works for NodeJS, but not for
         *   browsers, where files will usually be fetched via HTTP.
         * - Fetching files via HTTP is an asynchronous operation, but the C++ implementation is synchronous.
         * 
         * The approach taken is as follows:
         * - Determine the environment (NodeJS vs browser)
         * - Resolve and load files according to that environment
         * - Upload the files to the virtual WebAssembly file system, preserving the file system structure they were
         *   found in
         * - Rewrite absolute search paths to work in that file system
         * - Call the native (C++) function so that it reads the files from the virtual FS and constructs the document
         * - Delete the temporary files from the WASM FS
         */ 
        function _readFromXmlString(doc, str, searchPath, readOptions, filesLoaded = [], initialFilePath = "") {
            // Set WASM root folder
            var wasmRootFolder = "/readFromXml" + (callId++ % MAX_CALL_ID);

            // Prepare search paths
            var searchPaths = prepareSearchPaths(searchPath);

            // Prepare temporary folder in WASM file system
            try {
                FS.mkdir(wasmRootFolder);
            } catch (e) {
                throw new Error("Failed to create folder in WASM FS.");
            }

            // Parse includes
            var includes = [];
            if (!readOptions || readOptions.readXIncludes) {
                includes = getIncludes(str);
            }

            // Keep track of files uploaded to the WASM file system
            var filesUploaded = {
                files: [],
                folders: []
            };

            // Upload the initial file (string) to the WASM FS
            var wasmCwd = getWasmCwd(wasmRootFolder);
            var initialFileName = wasmCwd + "/ChosenToHopefullyNotClashWithAnyOtherFile123";
            if (initialFilePath) {
                var sep = pathSep === "\\" ? "\\\\" : pathSep;
                initialFileName = initialFilePath.replace(new RegExp(sep, "g"), wasmPathSep);
                initialFileName = createFilePath(initialFileName, wasmRootFolder, wasmPathSep);
                // initialFilePath being set means the user called readFromXmlFile, which might have resolved the
                // initial file outside of cwd, so we have to create cwd in the wasm fs explicitly, since we cd into it
                // later.
                createInWasm(wasmCwd, null, filesUploaded, wasmRootFolder, false);
            }
            createInWasm(initialFileName, str, filesUploaded, wasmRootFolder);

            // Load all files recursively (depth first) and store them in the WASM FS.
            function loadFiles(filesLoadedList, fileList, pathsList) {
                var promises = [Promise.resolve()];
                for (var fileToLoad of fileList) {
                    // Some lists need to be copied, to track their entries per branch
                    var filesLoadedCopy = filesLoadedList.slice();
                    var searchPathsCopy = pathsList.slice();

                    // Load the current file
                    var promise = loadFile(fileToLoad, searchPathsCopy).then(function (result) {
                        // Check for cycles
                        if (filesLoadedCopy.includes(result.fullPath)) {
                            throw new Error("Cycle detected!\n" + filesLoadedCopy.join("\n-> ") + "\n-> " +
                                result.fullPath);
                        }
                        filesLoadedCopy.push(result.fullPath);

                        // Update searchPaths
                        var pos = result.fullPath.lastIndexOf(pathSep);
                        var path = result.fullPath.substring(0, pos > -1 ? pos : 0);
                        if (!searchPathsCopy.includes(path)) {
                            searchPathsCopy.splice(0, 0, path);
                        }

                        // Check for includes
                        var includes = getIncludes(result.data);

                        // Upload to WASM FS
                        var wasmPath = createFilePath(result.filePath, wasmRootFolder, wasmPathSep);
                        if (!filesUploaded.files.includes(wasmPath)) {
                            createInWasm(wasmPath, result.data, filesUploaded, wasmRootFolder);
                        }

                        // Recursively load next file
                        return loadFiles(filesLoadedCopy, includes, searchPathsCopy);
                    });

                    promises.push(promise);
                }

                return Promise.all(promises);
            }

            // Wait for all files to be loaded and invoke the native function
            return loadFiles(filesLoaded, includes, searchPaths).then(function () {
                // Prepare search paths for the native function call
                var wasmSearchPath = getWasmSearchPath(searchPath, wasmRootFolder);

                // Set the current working directory, to make relative search paths work
                FS.chdir(wasmCwd);

                // Invoke native function
                try {
                    // Make sure the search path environment variable works in the WASM FS.
                    var searchPathEnv = Module.getEnviron(Module.MATERIALX_SEARCH_PATH_ENV_VAR);
                    if (searchPathEnv) {
                        var wasmSearchPathEnv = makeWasmAbsolute(searchPathEnv, wasmRootFolder);
                        Module.setEnviron(Module.MATERIALX_SEARCH_PATH_ENV_VAR, wasmSearchPathEnv);
                    }
                    Module._readFromXmlFile(doc, initialFileName, wasmSearchPath, readOptions);
                    if (searchPathEnv) {
                        Module.setEnviron(Module.MATERIALX_SEARCH_PATH_ENV_VAR, searchPathEnv);
                    }
                } catch (errPtr) {
                    throw new Error("Failed to read MaterialX files from WASM FS: " + Module.getExceptionMessage(errPtr));
                }

                // Purge WASM FS directory
                try {
                    for (var file of filesUploaded.files) {
                        FS.unlink(file);
                    }
                    FS.chdir("/"); // root
                    for (var folder of filesUploaded.folders) {
                        FS.rmdir(folder);
                    }
                    FS.rmdir(wasmRootFolder);
                } catch (e) {
                    throw new Error("Failed to delete temporary files from WASM FS.");
                }
            });
        };

        // Register the 'bindings'
        // Read a document from a string.
        Module.readFromXmlString = function (doc, str, searchPath = "", readOptions = null) {
            if (arguments.length < 2 || arguments.length > 4) {
                throw new Error("Function readFromXmlString called with an invalid number of arguments (" +
                    arguments.length + ") - expects 2 to 4!");
            }

            // Simply forward the call to the internal method
            return _readFromXmlString(doc, str, searchPath, readOptions);
        };

        // Read a document from file.
        Module.readFromXmlFile = function (doc, fileName, searchPath = "", readOptions = null) {
            if (arguments.length < 2 || arguments.length > 4) {
                throw new Error("Function readFromXmlFile called with an invalid number of arguments (" +
                    arguments.length + ") - expects 2 to 4!");
            }

            var searchPaths = prepareSearchPaths(searchPath);

            // Load initial file and pass the content to _readFromXmlString
            return loadFile(fileName, searchPaths).then(function (result) {
                // Keep track of the loaded file
                var filesLoaded = [result.fullPath];

                // Add file path to the search path
                var pos = result.fullPath.lastIndexOf(pathSep);
                var path = result.fullPath.substring(0, pos > -1 ? pos : 0);
                searchPath = searchPath.concat(PATH_LIST_SEPARATOR, path);

                // Pass file content to internal method
                return _readFromXmlString(doc, result.data, searchPath, readOptions, filesLoaded, result.filePath);
            });
        };

        // Write a document to a file.
        Module.writeToXmlFile = function(doc, fileName, writeOptions = null) {
            if (arguments.length < 2 || arguments.length > 3) {
                throw new Error("Function writeToXmlFile called with an invalid number of arguments (" +
                    arguments.length + ") - expects 2 to 3!");
            }

            var file = Module.writeToXmlString(doc, writeOptions);

            storeFileToDisk(fileName, file);
        };

        // Export a document to a file.
        Module.exportToXmlFile = function(doc, fileName, exportOptions = null) {
            if (arguments.length < 2 || arguments.length > 3) {
                throw new Error("Function exportToXmlFile called with an invalid number of arguments (" +
                    arguments.length + ") - expects 2 to 3!");
            }

            var file = Module.exportToXmlString(doc, exportOptions);

            storeFileToDisk(fileName, file);
        };
    });
})();
