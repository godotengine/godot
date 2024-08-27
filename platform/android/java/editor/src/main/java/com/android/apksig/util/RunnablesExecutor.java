/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.apksig.util;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Phaser;
import java.util.concurrent.ThreadPoolExecutor;

public interface RunnablesExecutor {
    static final RunnablesExecutor SINGLE_THREADED = p -> p.createRunnable().run();

    static final RunnablesExecutor MULTI_THREADED = new RunnablesExecutor() {
        private final int PARALLELISM = Math.min(32, Runtime.getRuntime().availableProcessors());
        private final int QUEUE_SIZE = 4;

        @Override
        public void execute(RunnablesProvider provider) {
            final ExecutorService mExecutor =
                    new ThreadPoolExecutor(PARALLELISM, PARALLELISM,
                            0L, MILLISECONDS,
                            new ArrayBlockingQueue<>(QUEUE_SIZE),
                            new ThreadPoolExecutor.CallerRunsPolicy());

            Phaser tasks = new Phaser(1);

            for (int i = 0; i < PARALLELISM; ++i) {
                Runnable task = () -> {
                    Runnable r = provider.createRunnable();
                    r.run();
                    tasks.arriveAndDeregister();
                };
                tasks.register();
                mExecutor.execute(task);
            }

            // Waiting for the tasks to complete.
            tasks.arriveAndAwaitAdvance();

            mExecutor.shutdownNow();
        }
    };

    void execute(RunnablesProvider provider);
}
