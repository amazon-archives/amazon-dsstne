/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

package com.amazon.dsstne.knn;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import lombok.RequiredArgsConstructor;
import lombok.Value;

/**
 * Batches the {@link #add(Object) add}s and invokes the worker once the batchSize is hit
 * or the timeout is hit (whichever comes first).
 *
 * @param <I> type of input
 * @param <O> type of output
 */
public class TimedBatchExecutor<I, O> {
    private static final long NOW = 0L;

    /**
     * Defines the batch work. The inputs contain the inputs in the batch.
     * The outputs is empty, where one should put the output for each input.
     * The inputs and outputs should be aligned by index. That is, the first element of
     * outputs should be the output for the first element of the inputs and so on.
     */
    public interface Work<I, O> {
        /**
         * Invokes this batch work.
         */
        void invoke(final List<I> inputs, final List<O> outputs);
    }

    private final String label;

    private final int batchSize;
    private final long timeout;

    private final BlockingQueue<BatchElement<I, O>> batchQueue;
    private final ScheduledExecutorService daemon;
    private final BatchProcessor batchProcessor;
    private final Lock queueLock;
    private final Lock processLock;

    public TimedBatchExecutor(final String label, final int batchSize, final long timeout, final Work<I, O> work) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize: " + batchSize + " should be > 0");
        }

        if (timeout <= 0) {
            throw new IllegalArgumentException("timeout: " + timeout + " should be > 0 ms");
        }

        this.label = label;
        this.batchSize = batchSize;
        this.timeout = timeout;

        this.daemon = Executors.newSingleThreadScheduledExecutor();
        this.batchQueue = new ArrayBlockingQueue<>(batchSize, true);
        this.batchProcessor = new BatchProcessor(work);
        this.queueLock = new ReentrantLock();
        this.processLock = new ReentrantLock();
    }

    /**
     * Pair of input and its respective output future.
     */
    @Value
    private static class BatchElement<I, O> {
        private final I input;
        private final CompletableFuture<O> output;
    }

    private ScheduledFuture<?> timeoutFlusher;

    /**
     * Adds the input to the batch to be processed when the batch size is reached
     * or the configured timeout milliseconds have elapsed. This method blocks until
     * the output is computed.
     */
    public O add(final I input) {
        // wait for the batch processor to return with results
        try {
            return addAsync(input).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error waiting for output for input: " + input);
        }
    }

    /**
     * Adds the input to the batch to be processed when the batch size is reached
     * or the configured timeout milliseconds have elapsed. This method returns immediately
     * after adding the input to the batch. Callers must wait on the returned future for
     * the result.
     */
    public CompletableFuture<O> addAsync(final I input) {
        try {
            CompletableFuture<O> output = new CompletableFuture<>();

            queueLock.lock();
            try {
                batchQueue.put(new BatchElement<>(input, output));

                boolean firstTask = batchQueue.size() == 1;
                if (firstTask) {
                    timeoutFlusher = daemon.schedule(batchProcessor, this.timeout, TimeUnit.MILLISECONDS);
                } else {
                    boolean batchFull = batchQueue.remainingCapacity() == 0;
                    if (batchFull) {
                        if (timeoutFlusher != null) {
                            timeoutFlusher.cancel(false);
                            timeoutFlusher = null;
                        }
                        daemon.schedule(batchProcessor, NOW, TimeUnit.MILLISECONDS);
                    }
                }
            } finally {
                queueLock.unlock();
            }

            return output;
        } catch (InterruptedException e) {
            throw new RuntimeException("Error queueing input: " + input + " to the current batch");
        }
    }

    /**
     * Wraps {@link Work} as a runnable. Responsible for draining the batch queue
     * and calling the {@link Work#invoke} function. The {@linkplain #run} function
     * is only thread-safe if the {@linkplain Work#invoke invoke} function is.
     */
    @RequiredArgsConstructor
    public class BatchProcessor implements Runnable {
        private final Work<I, O> work;

        @Override
        public void run() {
            processLock.lock();
            try {
                List<BatchElement<I, O>> batch = new ArrayList<>(batchSize);
                int numInputs = batchQueue.drainTo(batch, batchSize);
                if (numInputs == 0) { // no work
                    return;
                }

                List<I> inputs = new ArrayList<>(batchSize);
                List<O> outputs = new ArrayList<>(batchSize);

                for (int i = 0; i < batch.size(); i++) {
                    BatchElement<I, O> batchElement = batch.get(i);
                    inputs.add(batchElement.getInput());
                }

                invokeWork(inputs, outputs);

                if (inputs.size() != outputs.size()) {
                    throw new RuntimeException(
                            "Num inputs: " + inputs.size() + " does not match num outputs: " + outputs.size());
                }

                for (int i = 0; i < outputs.size(); i++) {
                    batch.get(i).getOutput().complete(outputs.get(i));
                }
            } catch (Throwable e) {
                e.printStackTrace();
            } finally {
                processLock.unlock();
            }
        }

        private void invokeWork(final List<I> inputs, final List<O> outputs) {
            work.invoke(inputs, outputs);
        }
    }
}
