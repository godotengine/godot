import globals from "globals";
import pluginJs from "@eslint/js";
import pluginReference from "eslint-plugin-html";

export default [
	pluginJs.configs.recommended,
	// pluginJs.configs.all,

	{
		rules: {
			"indent": ["error", "tab"],
			"curly": ["error", "all"],
			"quote-props": ["error", "consistent"],
			"no-self-assign": "off",
			"no-unused-vars": ["error", { "args": "none", "caughtErrors": "none" }],
			"no-console": "error",
			"no-eval": "error",
			"no-shadow": "error",
			"strict": ["error", "safe"],
			"no-alert": "error",
		},
	},

	{
		files: ["platform/web/js/engine/**/*.js"],
		languageOptions: {
			globals: {
				...globals.browser,
				"Features": true,
				"Godot": true,
				"InternalConfig": true,
				"Preloader": true,
			},
		},
	},

	{
		files: ["platform/web/js/jsdoc2rst/**/*.js"],
		languageOptions: {
			globals: globals.node,
		},
	},

	{
		files: [
			"platform/web/js/libs/**/*.js",
			"modules/**/*.js",
		],
		languageOptions: {
			globals: {
				...globals.browser,
				"autoAddDeps": true,
				"Browser": true,
				"ERRNO_CODES": true,
				"FS": true,
				"GL": true,
				"GodotConfig": true,
				"GodotEventListeners": true,
				"GodotFS": true,
				"GodotOS": true,
				"GodotRuntime": true,
				"HEAP32": true,
				"HEAP8": true,
				"HEAPF32": true,
				"HEAPU8": true,
				"IDBFS": true,
				"IDHandler": true,
				"LibraryManager": true,
				"mergeInto": true,
				"XRWebGLLayer": true,
			},
		},
	},

	{
		files: ["misc/dist/html/**/*.js"],
		languageOptions: {
			globals: {
				...globals.browser,
				"___GODOT_CACHE___": true,
				"___GODOT_ENSURE_CROSSORIGIN_ISOLATION_HEADERS___": true,
				"___GODOT_OPT_CACHE___": true,
				"onClientMessage": true,
			},
		},
	},

	{
		files: ["misc/dist/html/**/*.html"],
		plugins: {
			"eslint-plugin-html": pluginReference,
		},
		languageOptions: {
			globals: {
				...globals.browser,
				"Engine": true,
				"$GODOT_CONFIG": true,
				"$GODOT_PROJECT_NAME": true,
				"$GODOT_THREADS_ENABLED": true,
				"___GODOT_THREADS_ENABLED___": true,
			}
		},
		rules: {
			"no-console": "off",
			"no-alert": "off",
		},
	},

	{
		ignores: [
			"platform/web/eslint.config.mjs",
			"**/*.externs.js",
		],
	},
];
