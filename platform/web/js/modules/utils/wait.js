/**
 * @file Wait utilities.
 */

/**
 * Waits the specified amount of time then returns.
 * @async
 * @param {number} pNumber
 * @param {string} [pUnit="s"]
 * @returns {Promise<void>}
 * @throws
 */
export function wait(pNumber, pUnit = 's') {
	if (typeof pNumber != 'number') {
		throw new TypeError('pNumber is not a number.');
	}
	if (typeof pUnit != 'string') {
		throw new TypeError('pUnit is not a string.');
	}

	let waitTime = 0;

	switch (pUnit.toLowerCase()) {
	case 's':
	case 'second':
	case 'seconds':
		waitTime = pNumber * 1000;
		break;

	case 'ms':
	case 'millisecond':
	case 'milliseconds':
		waitTime = pNumber;
		break;

	default:
		throw new Error(`Unknown pUnit (${pUnit})`);
	}

	return new Promise((resolve) => {
		setTimeout(() => resolve(), waitTime);
	});
}

export default {
	wait,
};
