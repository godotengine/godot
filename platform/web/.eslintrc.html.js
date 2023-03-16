module.exports = {
	"plugins": [
		"html",
		"@html-eslint",
	],
	"parser": "@html-eslint/parser",
	"extends": ["plugin:@html-eslint/recommended", "./.eslintrc.js"],
	"rules": {
		"no-alert": "off",
		"no-console": "off",
		"@html-eslint/require-closing-tags": ["error", { "selfClosing": "never" }],
		"@html-eslint/indent": ["error", "tab"],
	},
	"globals": {
		"Godot": true,
		"Engine": true,
		"$GODOT_CONFIG": true,
	},
};
