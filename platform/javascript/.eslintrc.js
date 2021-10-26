module.exports = {
	"env": {
		"browser": true,
		"es2021": true,
	},
	"extends": [
		"airbnb-base",
	],
	"parserOptions": {
		"ecmaVersion": 12,
	},
	"ignorePatterns": "*.externs.js",
	"rules": {
		"func-names": "off",
		// Use tabs for consistency with the C++ codebase.
		"indent": ["error", "tab"],
		"max-len": "off",
		"no-else-return": ["error", {allowElseIf: true}],
		"curly": ["error", "all"],
		"brace-style": ["error", "1tbs", { "allowSingleLine": false }],
		"no-bitwise": "off",
		"no-continue": "off",
		"no-self-assign": "off",
		"no-tabs": "off",
		"no-param-reassign": ["error", { "props": false }],
		"no-plusplus": "off",
		"no-unused-vars": ["error", { "args": "none" }],
		"prefer-destructuring": "off",
		"prefer-rest-params": "off",
		"prefer-spread": "off",
		"camelcase": "off",
		"no-underscore-dangle": "off",
		"max-classes-per-file": "off",
		"prefer-arrow-callback": "off",
		// Messes up with copyright headers in source files.
		"spaced-comment": "off",
		// Completely breaks emscripten libraries.
		"object-shorthand": "off",
		// Closure compiler (exported properties)
		"quote-props": ["error", "consistent"],
		"dot-notation": "off",
		// No comma dangle for functions (it's madness, and ES2017)
		"comma-dangle": ["error", {
			"arrays": "always-multiline",
			"objects": "always-multiline",
			"imports": "always-multiline",
			"exports": "always-multiline",
			"functions": "never"
		}],
	}
};
