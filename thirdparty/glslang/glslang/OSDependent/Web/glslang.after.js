export default (() => {
    const initialize = () => {
        return new Promise(resolve => {
            Module({
                locateFile() {
                    const i = import.meta.url.lastIndexOf('/')
                    return import.meta.url.substring(0, i) + '/glslang.wasm';
                },
                onRuntimeInitialized() {
                    resolve({
                        compileGLSLZeroCopy: this.compileGLSLZeroCopy,
                        compileGLSL: this.compileGLSL,
                    });
                },
            });
        });
    };

    let instance;
    return () => {
        if (!instance) {
            instance = initialize();
        }
        return instance;
    };
})();
