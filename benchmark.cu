#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Cache sizes for H100 (accurate)
#define L1_CACHE_SIZE (256 * 1024)  // 256KB L1 cache per SM
#define L2_CACHE_SIZE (50 * 1024 * 1024) // 50MB L2 cache total

// Test sizes - adjusted for H100's larger L1 cache
#define FREQ_DATA_SIZE (32 * 1024)  // 32KB - data that should stay in L1 cache
#define STREAM_DATA_SIZE (40 * 1024 * 1024) // 40MB - streaming data

// Number of iterations
#define WARM_UP_ITERATIONS 5
#define BENCHMARK_ITERATIONS 20

// Access frequency for hot data
#define ACCESS_FREQUENCY 50

// Enum for different load types
enum LoadType {
    STANDARD_LOAD,
    SPECIALIZED_LOAD
};

// Kernel that demonstrates cache pollution effects
template<LoadType streamingLoadType>
__global__ void cachePollutionKernel(
    float* frequentlyAccessed,  // Data that should stay in L1 cache
    const float* streamingData, // Large data to stream through once
    float* results,             // Output for frequently accessed data
    int freqSize,               // Size of frequently accessed data
    int streamSize,             // Size of streaming data
    int accessFrequency)        // How many times to access the frequent data
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread-local accumulator
    float sum = 0.0f;
    
    // First, access the frequently accessed data and read it into registers
    // This should get the data into L1 cache
    for (int i = tid; i < freqSize; i += blockDim.x * gridDim.x) {
        sum += frequentlyAccessed[i];
    }
    
    // Now stream through the large data array once
    // This could potentially evict the frequently accessed data from L1 cache
    for (int i = tid; i < streamSize; i += blockDim.x * gridDim.x) {
        float val;
        
        if (streamingLoadType == STANDARD_LOAD) {
            // Standard load for streaming data
            val = streamingData[i];
        }
        else {
            #if __CUDA_ARCH__ >= 900
                asm volatile("ld.global.nc.L1::no_allocate.L2::256B.f32 %0, [%1];" : "=f"(val) : "l"(&streamingData[i]));
            #else
                val = streamingData[i];
            #endif
        }
        
        // Do something with the value so it doesn't get optimized away
        sum += val * 0.0001f;
    }
    
    // Now access the frequent data multiple times
    // If it was evicted from L1 cache, this will be slower
    for (int freq = 0; freq < accessFrequency; freq++) {
        for (int i = tid; i < freqSize; i += blockDim.x * gridDim.x) {
            sum += frequentlyAccessed[i] * 1.01f;
        }
    }
    
    // Store the result
    if (tid < freqSize) {
        results[tid] = sum;
    }
}

// Function to run the benchmark with a specific load type
float runBenchmark(LoadType loadType, const char* benchmarkName) {
    float *d_freqData, *d_streamData, *d_results;
    float totalTime = 0.0f;
    cudaEvent_t start, stop;
    
    // Allocate device memory
    cudaMalloc(&d_freqData, FREQ_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_streamData, STREAM_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_results, FREQ_DATA_SIZE * sizeof(float));
    
    // Initialize data
    cudaMemset(d_freqData, 0xAA, FREQ_DATA_SIZE * sizeof(float));
    cudaMemset(d_streamData, 0xBB, STREAM_DATA_SIZE * sizeof(float));
    
    // Create timing events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Set up kernel launch parameters
    dim3 blockSize(256);
    dim3 gridSize(512); // Use many blocks to increase parallelism
    
    // Warm-up runs
    printf("Warming up %s kernel (%d iterations)...\n", benchmarkName, WARM_UP_ITERATIONS);
    
    for (int i = 0; i < WARM_UP_ITERATIONS; i++) {
        if (loadType == STANDARD_LOAD) {
            cachePollutionKernel<STANDARD_LOAD><<<gridSize, blockSize>>>(
                d_freqData, d_streamData, d_results, 
                FREQ_DATA_SIZE, STREAM_DATA_SIZE, ACCESS_FREQUENCY);
        }
        else {
            cachePollutionKernel<SPECIALIZED_LOAD><<<gridSize, blockSize>>>(
                d_freqData, d_streamData, d_results, 
                FREQ_DATA_SIZE, STREAM_DATA_SIZE, ACCESS_FREQUENCY);
        }
    }
    
    cudaDeviceSynchronize();
    
    // Main benchmark runs
    printf("Running %s benchmark (%d iterations)...\n", benchmarkName, BENCHMARK_ITERATIONS);
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        float milliseconds = 0.0f;
        cudaDeviceSynchronize(); // Ensure GPU is idle
        
        // Record start time
        cudaEventRecord(start);
        
        // Launch appropriate kernel
        if (loadType == STANDARD_LOAD) {
            cachePollutionKernel<STANDARD_LOAD><<<gridSize, blockSize>>>(
                d_freqData, d_streamData, d_results, 
                FREQ_DATA_SIZE, STREAM_DATA_SIZE, ACCESS_FREQUENCY);
        }
        else {
            cachePollutionKernel<SPECIALIZED_LOAD><<<gridSize, blockSize>>>(
                d_freqData, d_streamData, d_results, 
                FREQ_DATA_SIZE, STREAM_DATA_SIZE, ACCESS_FREQUENCY);
        }
        
        // Record end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in %s kernel: %s\n", benchmarkName, cudaGetErrorString(err));
            continue;
        }
        
        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
        
        printf("  Iteration %2d: %.3f ms\n", i+1, milliseconds);
    }
    
    // Calculate average time
    float avgTime = totalTime / BENCHMARK_ITERATIONS;
    
    // Clean up
    cudaFree(d_freqData);
    cudaFree(d_streamData);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avgTime;
}

// Function to verify the results from both implementations are equivalent
bool verifyResults() {
    float *d_freqData, *d_streamData;
    float *d_standardResults, *d_specializedResults;
    float *h_standardResults, *h_specializedResults;
    bool resultsMatch = true;
    
    // Allocate device memory
    cudaMalloc(&d_freqData, FREQ_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_streamData, STREAM_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_standardResults, FREQ_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_specializedResults, FREQ_DATA_SIZE * sizeof(float));
    
    // Allocate host memory for results
    h_standardResults = (float*)malloc(FREQ_DATA_SIZE * sizeof(float));
    h_specializedResults = (float*)malloc(FREQ_DATA_SIZE * sizeof(float));
    
    // Initialize data with fixed values for reproducibility
    cudaMemset(d_freqData, 0x42, FREQ_DATA_SIZE * sizeof(float));
    cudaMemset(d_streamData, 0x43, STREAM_DATA_SIZE * sizeof(float));
    
    // Set up kernel launch parameters
    dim3 blockSize(256);
    dim3 gridSize(512);
    
    // Run both kernels once
    cachePollutionKernel<STANDARD_LOAD><<<gridSize, blockSize>>>(
        d_freqData, d_streamData, d_standardResults, 
        FREQ_DATA_SIZE, STREAM_DATA_SIZE, 1); // Just one access for verification
    
    cachePollutionKernel<SPECIALIZED_LOAD><<<gridSize, blockSize>>>(
        d_freqData, d_streamData, d_specializedResults, 
        FREQ_DATA_SIZE, STREAM_DATA_SIZE, 1); // Just one access for verification
    
    // Copy results back to host
    cudaMemcpy(h_standardResults, d_standardResults, FREQ_DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_specializedResults, d_specializedResults, FREQ_DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results (allow for small floating-point differences)
    const float epsilon = 1e-5f;
    for (int i = 0; i < FREQ_DATA_SIZE; i++) {
        float diff = fabs(h_standardResults[i] - h_specializedResults[i]);
        if (diff > epsilon) {
            printf("Results mismatch at index %d: Standard=%.6f, Specialized=%.6f\n",
                   i, h_standardResults[i], h_specializedResults[i]);
            resultsMatch = false;
            break;
        }
    }
    
    // Clean up
    cudaFree(d_freqData);
    cudaFree(d_streamData);
    cudaFree(d_standardResults);
    cudaFree(d_specializedResults);
    free(h_standardResults);
    free(h_specializedResults);
    
    return resultsMatch;
}

int main() {
    // Check CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    
    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n=== CACHE POLLUTION BENCHMARK ===\n\n");
    
    printf("GPU: %s (Compute Capability %d.%d)\n", 
           prop.name, prop.major, prop.minor);
    
    printf("L1 Cache per SM: ~256KB\n");  // Updated for H100
    printf("L2 Cache Total: ~50MB\n");    // Updated for H100
    printf("Benchmark Data Sizes:\n");
    printf("  - Frequently accessed data: %d KB\n", FREQ_DATA_SIZE / 1024);
    printf("  - Streaming data: %d MB\n", STREAM_DATA_SIZE / (1024 * 1024));
    printf("  - Access frequency for hot data: %d times\n\n", ACCESS_FREQUENCY);
    
    // Verify that both implementations produce the same results
    printf("Verifying both implementations produce equivalent results...\n");
    bool resultsMatch = verifyResults();
    if (!resultsMatch) {
        printf("ERROR: Results don't match! This indicates a problem with the implementation.\n");
        return 1;
    }
    printf("Results verified: Both implementations produce equivalent results.\n\n");
    
    // Run standard load benchmark
    float standardTime = runBenchmark(STANDARD_LOAD, "Standard Load");
    
    // Run specialized load benchmark
    float specializedTime = runBenchmark(SPECIALIZED_LOAD, "Specialized Load");
    
    // Print results
    printf("\n=== BENCHMARK RESULTS ===\n");
    printf("Standard Load:     %.3f ms\n", standardTime);
    printf("Specialized Load:  %.3f ms\n", specializedTime);
    
    double speedup = standardTime / specializedTime;
    printf("Speedup:           %.3fx\n", speedup);
    
    if (speedup > 1.05) {
        printf("\nRESULT: Specialized load is FASTER (%.1f%% improvement)\n", 
               (speedup - 1.0) * 100.0);
    } else if (speedup < 0.95) {
        printf("\nRESULT: Specialized load is SLOWER (%.1f%% slower)\n", 
               (1.0 - speedup) * 100.0);
    } else {
        printf("\nRESULT: Performance is EQUIVALENT (within 5%%)\n");
    }
    
    printf("\n");
    printf("This benchmark demonstrates cache pollution effects:\n");
    printf("1. First, we load small data that should ideally stay in L1 cache\n");
    printf("2. Then, we stream through large data that could evict the small data\n");
    printf("   - Standard loads may pollute L1 cache with streaming data\n");
    printf("   - Specialized loads with L1::no_allocate bypass L1 cache\n");
    printf("3. Finally, we access the small data again multiple times\n");
    printf("\n");
    printf("If specialized loads are faster, it indicates they're preserving \n");
    printf("the important data in L1 cache by not polluting it with streaming data.\n");
    
    return 0;
}
