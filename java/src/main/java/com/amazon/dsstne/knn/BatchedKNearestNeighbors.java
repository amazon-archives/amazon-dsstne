package com.amazon.dsstne.knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import com.amazon.dsstne.knn.TimedBatchExecutor.Work;

import lombok.Value;

/**
 * A thread-safe implementation of {@link KNearestNeighbors} that batches
 * the {@link KNearestNeighbors#findKnn(int, Vector)} calls.
 * <p>
 * The timeout defines the maximum wait time for the batch to fill.
 * If the batch does not fill by the designated timeout, then a call to findKnn
 * is dispatched with what is there.
 */
public class BatchedKNearestNeighbors extends AbstractKNearestNeighbors {

    private final KNearestNeighborsCuda knnCuda;
    private final TimedBatchExecutor<KnnInput, NearestNeighbors> batchExecutor;

    private final float[] batchInputs; // input vectors
    private final float[] batchScores; // output scores
    private final String[] batchIds; // output ids (keys)

    /**
     * POJO representing knn input.
     */
    @Value
    static class KnnInput {
        private final int k;
        private final Vector vector;
    }

    BatchedKNearestNeighbors(final KNearestNeighborsCuda knnCuda, final long timeout) {
        this(null, knnCuda, timeout);
    }

    public BatchedKNearestNeighbors(final String label, final KNearestNeighborsCuda knnCuda, final long timeout) {
        super(label, knnCuda.getMaxK(), knnCuda.getFeatureSize(), knnCuda.getBatchSize());
        this.knnCuda = knnCuda;

        /*
         * pre-allocate the batch input/output arrays
         * this is what makes this class by itself NOT thread-safe
         * however, when used in the context of TimedBatchExecutor,
         * the executor ensures that only one batch is being worked on
         * at any given point in time.
         */
        int batchSize = getBatchSize();
        int featureSize = getFeatureSize();
        int maxK = getMaxK();
        this.batchInputs = new float[batchSize * featureSize];
        this.batchScores = new float[batchSize * maxK];
        this.batchIds = new String[batchSize * maxK];

        this.batchExecutor = new TimedBatchExecutor<>(label, batchSize, timeout, new FindKnnWork());
    }

    /**
     * Work that calls knnCuda implementation in batches.
     */
    class FindKnnWork implements Work<KnnInput, NearestNeighbors> {

        @Override
        public void invoke(final List<KnnInput> inputs, final List<NearestNeighbors> outputs) {
            int featureSize = getFeatureSize();
            int maxK = getMaxK();
            int activeBatchSize = inputs.size(); // inputs that are actually set in the batch

            for (int offset = 0; offset < inputs.size(); offset++) {
                KnnInput input = inputs.get(offset);
                List<Float> coordinates = input.vector.getCoordinates();
                for (int i = 0; i < coordinates.size(); i++) {
                    batchInputs[offset * featureSize + i] = coordinates.get(i);
                }
            }

            if (inputs.size() < getBatchSize()) {
                // zero out the rest of the vector
                Arrays.fill(batchInputs, inputs.size() * featureSize, batchInputs.length, 0f);
            }

            knnCuda.findKnn(batchInputs, activeBatchSize, batchScores, batchIds);

            for (int offset = 0; offset < inputs.size(); offset++) {
                int k = inputs.get(offset).getK();
                int inputIdx = inputs.get(offset).getVector().getIndex();

                List<Neighbor> nearestNeighbors = new ArrayList<>(k);
                for (int i = 0; i < k; i++) {
                    float score = batchScores[maxK * offset + i];
                    String id = batchIds[maxK * offset + i];
                    Neighbor neighbor = Neighbor.builder().withId(id).withScore(score).build();
                    nearestNeighbors.add(neighbor);
                }
                outputs.add(NearestNeighbors.builder().withIndex(inputIdx).withNeighbors(nearestNeighbors).build());
            }
        }
    }

    @Override
    public NearestNeighbors findKnn(final int k, final Vector vector) {
        validateK(k);
        validateFeatureSize(vector);
        return batchExecutor.add(new KnnInput(k, vector));
    }

    @Override
    public List<NearestNeighbors> findKnnBatch(final int k, final List<Vector> inputBatch) {
        validateK(k);
        validateFeatureSize(inputBatch);

        List<CompletableFuture<NearestNeighbors>> futures = new ArrayList<>(inputBatch.size());
        for (Vector vector : inputBatch) {
            futures.add(batchExecutor.addAsync(new KnnInput(k, vector)));
        }

        List<NearestNeighbors> results = new ArrayList<>(inputBatch.size());
        for (CompletableFuture<NearestNeighbors> future : futures) {
            try {
                results.add(future.get());
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Error waiting for knn result", e);
            }
        }
        return results;
    }
}
