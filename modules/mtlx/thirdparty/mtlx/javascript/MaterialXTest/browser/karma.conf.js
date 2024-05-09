module.exports = function (config)
{
    config.set({
        basePath: '../', // base is the javascript folder
        files: [
            { pattern: '_build/JsMaterialXGenShader.js', watched: false, included: true, served: true },
            { pattern: '_build/JsMaterialXGenShader.wasm', watched: false, included: false, served: true },
            { pattern: '_build/JsMaterialXGenShader.data', watched: false, included: false, served: true, nocache: true },
            { pattern: 'browser/*.spec.js', watched: true, included: true, served: true },
        ],
        mime: {
            'application/wasm': ['wasm'],
            'application/octet-stream; charset=UTF-8': ['data'],
        },
        proxies: {
            '/JsMaterialXGenShader.data': '/base/_build/JsMaterialXGenShader.data',
        },
        reporters: ['mocha'],
        client: {
            mocha: {
                reporter: 'html'
            }
        },
        browsers: ['Chrome'],
        port: 8080,
        autoWatch: true,
        concurrency: Infinity,
        // logLevel: config.LOG_DEBUG,
        frameworks: ['mocha', 'chai'],
        plugins: [
            'karma-chai',
            'karma-chrome-launcher',
            'karma-mocha',
            'karma-mocha-reporter',
        ],
    });
};
