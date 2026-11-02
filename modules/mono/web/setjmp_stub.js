// 2dog: satisfy setjmp imports omitted by the .NET wasm toolchain.
const SetJmpStub = {
	__wasm_setjmp_sig: 'vpip',
	__wasm_setjmp: function (_env, _label, _func_invocation_id) { },

	__wasm_setjmp_test_sig: 'ipp',
	__wasm_setjmp_test: function (_env, _func_invocation_id) {
		return 0;
	},
};

addToLibrary(SetJmpStub);
