// 2dog: library builds have no C main, but Emscripten requires this exported runtime symbol.
const callMain = function (_args) { // eslint-disable-line no-unused-vars
	throw new Error('"callMain" is not implemented.');
};
