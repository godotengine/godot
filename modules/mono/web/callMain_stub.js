// When we are passing "callMain" to <EmccExportedRuntimeMethod /> it checks if "callMain" exists,
// but doesn't generate "callMain" itself. I guess emscripten is hard wired to detect
// main and only generates "callMain" if it detects it. Or it is some C# customization.
const callMain = function (_args) { // eslint-disable-line no-unused-vars
	throw new Error('"callMain" is not implemented.');
};
