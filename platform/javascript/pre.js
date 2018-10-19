var Engine = {
	RuntimeEnvironment: function(Module, exposedLibs) {
		exposedLibs['PATH'] = PATH;
		exposedLibs['FS'] = FS;
		return Module;
	},
};
