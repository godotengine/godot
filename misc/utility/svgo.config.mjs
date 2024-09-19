export default {
	multipass: true,
	precision: 2,
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
