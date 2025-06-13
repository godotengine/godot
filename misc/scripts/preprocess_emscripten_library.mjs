import { basename, dirname, join, resolve } from 'node:path';
import { exit, platform } from 'node:process';
import { mkdtemp, readFile, rm, rmdir, writeFile } from 'node:fs/promises';
import { execSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { tmpdir } from 'node:os';

const currentFilePath = fileURLToPath(import.meta.url);
const entryPointPath = resolve(process.argv[1]);
const isThisFileBeingRunDirectly = currentFilePath.includes(entryPointPath);

/** @type {import("node:util").ParseArgsOptionsConfig} */
const argOptions = {
	'help': {
		type: 'boolean',
		default: false,
		short: 'h',
		description: 'Summons help',
	},
	'verbose': {
		type: 'boolean',
		default: false,
		short: 'V',
		description: 'Enable verbosity',
	},
	'output': {
		type: 'string',
		default: '',
		short: 'o',
		description: 'Write to a file instead of stdout',
	},
	'emscripten-settings': {
		type: 'string',
	},
	'file': {
		type: 'string',
	},
};

/**
 * @param {string} command
 * @returns {string | null}
 */
export function which(command) {
	try {
		/** @type {string} */
		let result;
		if (platform === 'win32') {
			result = execSync(`where ${command}`).toString().trim().split('\r\n')[0];
		} else {
			result = execSync(`which ${command}`).toString().trim();
		}
		return result;
	} catch (err) {
		return null;
	}
}

/**
 * @param {ReturnType<typeof parseArgs>} args
 * @returns {void}
 */
function processArgs(args) {
	const processBasename = basename(resolve(import.meta.url));

	const printArgName = (name, short = null) => {
		let printValue = '';
		if (short != null) {
			printValue += `-${short}, `;
		}
		printValue += `--${name}`;
		return printValue;
	};

	// Help.
	if (args.values.help) {
		const positionals = Object.entries(argOptions).filter(([_key, value]) => !('default' in value));
		let command = processBasename;
		for (const [key] of positionals) {
			command += ` <${key}>`;
		}
		console.log(command);

		const nonPositionals = Object.entries(argOptions).filter(([_key, value]) => 'default' in value);
		for (const [key, value] of nonPositionals) {
			console.log(`${printArgName(key, value.short)}: ${value.description}`);
		}
		exit(0);
	}

	// Check positional args.
	const numberOfPositionalArgs = Object.entries(argOptions).filter(([_key, value]) => !('default' in value)).length;
	if (args.positionals.length !== numberOfPositionalArgs) {
		if (args.positionals.length === 0) {
			console.error('Positional argument <emscripten-settings> missing.');
		} else if (args.positionals.length === 1) {
			console.error('Positional argument <file> missing.');
		} else {
			console.error('Too many positional arguments.');
		}
		exit(1);
	}
}

/**
 * @param {object} options
 * @param {string} options.settings
 * @param {string} options.file
 * @param {string} options.emscriptenPath
 * @param {string} options.emscriptenSettings
 * @param {string?} options.output
 * @returns {Promise<void>}
 */
export async function parseFile(options) {
	const { file, emscriptenPath, emscriptenSettings, output = '' } = options;

	if (!existsSync(file)) {
		console.error(`"${file}" doesn't exist.`);
		exit(1);
	}

	const fileContents = await readFile(file, { encoding: 'utf-8' });

	const { loadDefaultSettings, addToCompileTimeContext } = await import(resolve(emscriptenPath, 'src/utility.mjs'));
	loadDefaultSettings();
	const { processMacros, preprocess } = await import(resolve(emscriptenPath, 'src/parseTools.mjs'));

	const settingsContents = await readFile(emscriptenSettings, { encoding: 'utf-8' });
	const settingsJson = JSON.parse(settingsContents);
	addToCompileTimeContext(settingsJson);

	const processedMacrosFileContents = processMacros(fileContents, file);

	const temporaryDir = await mkdtemp(join(tmpdir(), 'preprocess_emscripten'));
	const preprocessedFile = join(temporaryDir, 'lib.preprocessed.js');
	await writeFile(preprocessedFile, processedMacrosFileContents, { encoding: 'utf-8' });
	let preprocessedFileContents;
	try {
		preprocessedFileContents = preprocess(preprocessedFile);
	} catch (err) {
		console.error(err);
		console.error(
			`\nEmscripten's \`preprocess()\` failed. It often means that there's a definition missing in "${emscriptenSettings}".`
		);
		exit(1);
	}

	await rm(preprocessedFile);
	await rmdir(temporaryDir);

	if (output.length === 0) {
		console.log(preprocessedFileContents);
	} else {
		await writeFile(output, preprocessedFileContents, {
			encoding: 'utf-8',
		});
	}
	exit(0);
}

/**
 * @returns {Promise<void>}
 */
async function main() {
	const emccPath = which('emcc');
	if (emccPath == null) {
		throw new Error('Did not find `emcc`. Has emscripten added to the path?');
	}
	const emscriptenPath = dirname(emccPath);

	const args = parseArgs({
		options: argOptions,
		strict: true,
		allowPositionals: true,
	});
	processArgs(args);

	let output = args.values.output;
	if (output.length > 0) {
		output = resolve(output);
	}
	await parseFile({
		emscriptenPath,
		emscriptenSettings: args.positionals[0],
		file: args.positionals[1],
		output,
	});
}

// Only run if the script is run directly from Node.
// (i.e. not imported as a module.)
if (isThisFileBeingRunDirectly) {
	try {
		await main();
	} catch (err) {
		console.error(err);
		exit(1);
	}
}
