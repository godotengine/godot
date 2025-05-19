import { argv, chdir, exit } from 'node:process';
import { basename, dirname, extname, join, resolve as pathResolve, relative } from 'node:path';
import { readdir, rm } from 'node:fs/promises';

import { ESLint } from 'eslint';
import { exec } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const currentFilePath = fileURLToPath(import.meta.url);
const entryPointPath = pathResolve(argv[1]);
const isThisFileBeingRunDirectly = currentFilePath.includes(entryPointPath);

const root = pathResolve(dirname(currentFilePath), '..', '..');

/**
 * @param {string} name
 * @returns {string}
 */
export function rootName(name) {
	return name.split('.')[0];
}

/**
 * @param {string} name
 * @returns {string}
 */
export function withoutExtName(name) {
	const ext = extname(name);
	const withoutExt = name.substring(0, name.length - ext.length);
	return withoutExt;
}

/**
 * @param {string} file
 * @param {object} settings
 * @param {boolean?} settings.verbose
 * @param {boolean?} settings.keepPreprocessed
 */
export async function lintEmscriptenLibraryFile(file, settings) {
	const { verbose = false, keepPreprocessed = false } = settings;
	if (verbose) {
		console.log(`Linting emscripten library file "${file}"`);
	}

	const libraryDirectory = dirname(file);
	let directoryFiles;
	try {
		directoryFiles = await readdir(libraryDirectory);
	} catch (err) {
		console.error(err);
		exit(1);
	}

	const fileExt = extname(file);
	const fileWithoutExt = withoutExtName(basename(file));
	const fileRoot = rootName(basename(file));
	const settingFiles = [];

	for (const directoryFile of directoryFiles) {
		const directoryFileRoot = rootName(directoryFile);
		if (directoryFileRoot !== fileRoot) {
			continue;
		}
		if (
			directoryFile === `${directoryFileRoot}.lint_settings.json`
			|| (directoryFile.startsWith(`${directoryFileRoot}.lint_settings.`) && directoryFile.endsWith('.json'))
		) {
			settingFiles.push(join(libraryDirectory, directoryFile));
		}
	}

	if (settingFiles.length === 0) {
		if (verbose) {
			console.log('Did not find any settings file. Linting file as is.');
		}
		const results = await runESLint(file);
		processESLintResultsAndExit(results);
	}

	for (const settingFile of settingFiles) {
		const baseSettingFile = basename(settingFile);
		try {
			const settingMiddlePart = baseSettingFile.substring(
				fileRoot.length,
				baseSettingFile.length - '.json'.length
			);
			const preprocessed = join(libraryDirectory, `${fileWithoutExt}.preprocessed${settingMiddlePart}${fileExt}`);

			await new Promise((resolve, _reject) => {
				const preprocess_script = join('misc', 'scripts', 'preprocess_emscripten_library.mjs');
				const command = ['node', preprocess_script, settingFile, file, '-o', preprocessed].join(' ');
				if (verbose) {
					console.log(`Executing: ${command}`);
				}
				exec(command, (error, _stdout, stderr) => {
					if (error != null) {
						console.error(stderr);
						exit(1);
					}
					resolve();
				});
			});

			const results = await runESLint(preprocessed);
			const success = processESLintResults(results);

			if (!keepPreprocessed) {
				await rm(preprocessed);
			}

			if (!success) {
				exit(1);
			}
		} catch (err) {
			console.error(err);
			exit(1);
		}
	}

	exit(0);
}

/**
 * @param {string | string[]} patterns
 * @returns {Promise<ESLint.LintResult[]>}
 */
export async function runESLint(patterns) {
	const eslint = new ESLint({
		cwd: root,
		fix: true,
		warnIgnored: false,
		overrideConfigFile: 'eslint.config.mts',
	});

	const results = await eslint.lintFiles(patterns);
	return results;
}

/**
 * @param {ESLint.LintResult[]} results
 * @returns {boolean}
 */
export function processESLintResults(results) {
	if (results.length === 0) {
		return true;
	}
	let hasError = false;
	for (const result of results) {
		const localHasError = result.errorCount > 0 || result.fatalErrorCount > 0;
		if (!localHasError) {
			continue;
		}
		hasError = true;

		const relativeFilePath = relative(root, result.filePath);
		for (const resultMessage of result.messages) {
			let error = '\n';
			const suggestions = resultMessage.suggestions ?? [];
			error += `${relativeFilePath}:${resultMessage.line}:${resultMessage.column}
${resultMessage.message}
`;
			for (const suggestion of suggestions) {
				error += `Suggestion: ${suggestion.desc}\n`;
			}
			console.error(error);
		}
	}
	return !hasError;
}

/**
 * @param {ESLint.LintResult[]} results
 * @returns {never}
 */
function processESLintResultsAndExit(results) {
	exit(processESLintResults(results) ? 0 : 1);
}

/**
 * @param {string} fileName
 * @param {object} options
 * @param {boolean?} options.verbose
 * @param {boolean?} options.keepPreprocessed
 * @returns {Promise<void>}
 */
export async function lint(fileName, options = {}) {
	const { verbose = false, keepPreprocessed = false } = options;

	const relativeFileName = relative(root, fileName);
	if (relativeFileName.startsWith('..')) {
		throw new Error(`"${fileName}" is outside project root.`);
	}

	if (!existsSync(fileName)) {
		throw new Error(`"${fileName}" doesn't exist.`);
	}

	if (relativeFileName.startsWith(join('platform', 'web', 'js', 'libs'))) {
		lintEmscriptenLibraryFile(fileName, { verbose, keepPreprocessed });
		return;
	}

	const results = await runESLint(fileName, { verbose });
	processESLintResultsAndExit(results);
}

export function chdirToGodotRoot() {
	chdir(root);
}

async function main() {
	const args = parseArgs({
		options: {
			'verbose': {
				short: 'V',
				type: 'boolean',
				default: false,
			},
			'keep-preprocessed': {
				short: 'k',
				type: 'boolean',
				default: false,
			},
			'file': {
				type: 'string',
			},
		},
		strict: true,
		allowPositionals: true,
	});

	await lint(args.positionals[0], {
		verbose: args.values.verbose,
		keepPreprocessed: args.values['keep-preprocessed'],
	});
}

// Only run if the script is run directly from Node.
// (i.e. not imported as a module.)
if (isThisFileBeingRunDirectly) {
	try {
		chdirToGodotRoot();
		await main();
	} catch (err) {
		console.error(err);
		exit(1);
	}
}
