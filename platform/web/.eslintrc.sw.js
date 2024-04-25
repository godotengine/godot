module.exports = {
	extends: ["./.eslintrc.js"],
	rules: {
		"no-restricted-globals": 0,
	},
	globals: {
		onClientMessage: true,
		___GODOT_ENSURE_CROSSORIGIN_ISOLATION_HEADERS___: true,
		___GODOT_CACHE___: true,
		___GODOT_OPT_CACHE___: true,
	},
};
