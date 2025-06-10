export default {
	multipass: true,
	precision: 2,
	js2svg: {
		eol: "lf",
		finalNewline: true,
	},
	plugins: [
		{
			name: "preset-default",
			params: {
				overrides: {
					removeHiddenElems: false,
					convertPathData: false,
				},
			},
		},
		"convertStyleToAttrs",
		"removeScriptElement",
		"removeStyleElement",
		"reusePaths",
		"sortAttrs",
	],
};
