var Loader = /** @constructor */ function() {

	this.env = null;

	this.init = function(loadPromise, basePath, config) {
		var me = this;
		return new Promise(function(resolve, reject) {
			var cfg = config || {};
			cfg['locateFile'] = Utils.createLocateRewrite(basePath);
			cfg['instantiateWasm'] = Utils.createInstantiatePromise(loadPromise);
			loadPromise = null;
			Godot(cfg).then(function(module) {
				me.env = module;
				resolve();
			});
		});
	}

	this.start = function(preloadedFiles, args) {
		var me = this;
		return new Promise(function(resolve, reject) {
			if (!me.env) {
				reject(new Error('The engine must be initialized before it can be started'));
			}
			preloadedFiles.forEach(function(file) {
				Utils.copyToFS(me.env['FS'], file.path, file.buffer);
			});
			preloadedFiles.length = 0; // Clear memory
			me.env['callMain'](args);
			resolve();
		});
	}
};
