/**
 * @file Simple concurrency queue manager.
 * @example
 * // This will limit the number of simultaneous fetch calls to 2.
 * const manager = ConcurrencyQueueManager({ limit: 2 });
 * for (let i; i < 100; i++) {
 *   manager.queue(async () => {
 *     const targetUrl = new URL('https://example.org/');
 *     targetUrl.searchParams.set("i", i);
 *     return await fetch(targetUrl);
 *   });
 * }
 */

const defaultValues = Object.freeze({
	concurrencyLimit: 5,
});

/**
 * @template T
 * @typedef {() => Promise<T>} PromiseWrapper<T>
 * @typedef {{
 *   symbol: Symbol,
 *   promiseWrapper: PromiseWrapper<T>,
 * }} ConcurrencyQueueManagerItem<T>
 */

/**
 * @typedef {{
 *   limit: number,
 * }} ConcurrencyQueueManagerOptions
 */

export class ConcurrencyQueueManager {
	/** @type {number} */
	#limit = defaultValues.concurrencyLimit;
	/** @type {ConcurrencyQueueManagerItem[]} */
	#queue = [];
	/** @type {ConcurrencyQueueManagerItem[]} */
	#active = [];

	#eventTarget = new EventTarget();

	/**
	 * @constructor
	 * @param {ConcurrencyQueueManagerOptions} [pOptions={}]
	 */
	constructor(pOptions = {}) {
		const { limit = defaultValues.concurrencyLimit } = pOptions;
		this.#limit = limit;
	}

	get limit() {
		return this.#limit;
	}

	/**
	 * Adds a function that returns a promise to the queue. Will return after the promise is executed.
	 * @async
	 * @template T
	 * @param {PromiseWrapper<T>} pPromiseWrapper
	 * @returns {Promise<T>}
	 * @throws
	 */
	async queue(pPromiseWrapper) {
		if (pPromiseWrapper == null) {
			throw new ReferenceError('pPromiseWrapper is null.');
		}

		const queueItem = await this.#waitForActiveQueueItem(pPromiseWrapper);
		try {
			return await queueItem.promiseWrapper();
		} catch (error) {
			const newError = new Error('ConcurrencyQueueManager detected an error in a managed promise.');
			newError.cause = error;
			throw error;
		} finally {
			const queueIndex = this.#active.indexOf(queueItem);
			this.#active.splice(queueIndex, 1);
			while (this.#queue.length > 0 && this.#active.length < this.#limit) {
				const concurrencyQueueItem = this.#queue[0];
				this.#queue.splice(0, 1);
				this.#active.push(queueItem);
				this.#eventTarget.dispatchEvent(new CustomEvent('queuenext', { detail: concurrencyQueueItem }));
			}
		}
	}

	/**
	 * Waits for the queue to be available.
	 * @template T
	 * @param {PromiseWrapper<T>} pPromiseWrapper
	 * @returns {Promise<ConcurrencyQueueManagerItem<T>>}
	 * @throws
	 */
	#waitForActiveQueueItem(pPromiseWrapper) {
		if (pPromiseWrapper == null) {
			throw new ReferenceError('pPromiseWrapper is null.');
		}

		const symbol = Symbol('ConcurrencyQueueItemId');
		/** @type {ConcurrencyQueueManagerItem} */
		const queueItem = {
			symbol,
			promiseWrapper: pPromiseWrapper,
		};

		if (this.#active.length < this.#limit) {
			this.#active.push(queueItem);
			return queueItem;
		}

		this.#queue.push(queueItem);
		return new Promise((pResolve, _pReject) => {
			/** @type {(event: CustomEvent) => void} */
			const onQueueNext = (pEvent) => {
				if (pEvent?.detail?.symbol !== symbol) {
					return;
				}
				this.#eventTarget.removeEventListener('queuenext', onQueueNext);
				pResolve(pEvent.detail);
			};
			this.#eventTarget.addEventListener('queuenext', onQueueNext);
		});
	}
}

export default {
	ConcurrencyQueueManager,
};
