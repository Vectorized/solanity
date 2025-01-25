#include <vector>
#include <random>
#include <chrono>

#include <iostream>
#include <ctime>

#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Types ----------------------------------------------------------------- */

typedef struct {
    unsigned char** states;
} config;

/* -- Prototypes, Because C++ ----------------------------------------------- */

void            vanity_setup(config& vanity);
void            vanity_run(config& vanity);
void __global__ vanity_scan(unsigned char* state, int* keys_found, int* gpu, int* execution_count);
void __device__ b58enc(char* b58, const uint8_t* data);

/* -- Entry Point ----------------------------------------------------------- */

int main(int argc, char const* argv[]) {
    ed25519_set_verbose(true);

    config vanity;
    vanity_setup(vanity);
    vanity_run(vanity);
}

std::string getTimeStr(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

/* -- Vanity Step Functions ------------------------------------------------- */

void vanity_setup(config &vanity) {
    printf("GPU: Initializing Memory\n");
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);
    vanity.states = (unsigned char **) malloc(gpuCount * sizeof(unsigned char *));

    // Create random states so kernels have access to random generators
    // while running in the GPU.
    for (int i = 0; i < gpuCount; ++i) {
        cudaSetDevice(i);

        // Fetch Device Properties
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, i);

        // Calculate Occupancy
        int blockSize       = 0,
            minGridSize     = 0,
            maxActiveBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

        // Output Device Details
        // 
        // Our kernels currently don't take advantage of data locality
        // or how warp execution works, so each thread can be thought
        // of as a totally independent thread of execution (bad). On
        // the bright side, this means we can really easily calculate
        // maximum occupancy for a GPU because we don't have to care
        // about building blocks well. Essentially we're trading away
        // GPU SIMD ability for standard parallelism, which CPUs are
        // better at and GPUs suck at.
        //
        // Next Weekend Project: ^ Fix this.
        printf("GPU: %d (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
            i,
            device.name,
            blockSize,
            minGridSize,
            maxActiveBlocks,
            device.warpSize,
            device.multiProcessorCount,
            device.maxThreadsPerBlock,
            device.maxThreadsDim[0],
            device.maxThreadsDim[1],
            device.maxThreadsDim[2],
            device.maxGridSize[0],
            device.maxGridSize[1],
            device.maxGridSize[2]
        );

        unsigned int n = maxActiveBlocks * blockSize * 32;
        unsigned char *rseed = (unsigned char *) malloc(n);

        std::random_device rd;
        std::uniform_int_distribution<int> dist(0, 255);
        for (unsigned int j = 0; j < n; ++j) {
            rseed[j] = static_cast<unsigned char>(dist(rd));
        }

        cudaMalloc((void **)&(vanity.states[i]), n);
        cudaMemcpy(vanity.states[i], rseed, n, cudaMemcpyHostToDevice); 
    }

    printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity) {
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    unsigned long long int  executions_total = 0; 
    unsigned long long int  executions_this_iteration; 
    int  executions_this_gpu; 
    int* dev_executions_this_gpu[100];

    int  keys_found_total = 0;
    int  keys_found_this_iteration;
    int* dev_keys_found[100]; // not more than 100 GPUs ok!

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        executions_this_iteration=0;

        // Run on all GPUs
        for (int g = 0; g < gpuCount; ++g) {
            cudaSetDevice(g);
            // Calculate Occupancy
            int blockSize       = 0,
                minGridSize     = 0,
                maxActiveBlocks = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

            int* dev_g;
            cudaMalloc((void**)&dev_g, sizeof(int));
            cudaMemcpy( dev_g, &g, sizeof(int), cudaMemcpyHostToDevice ); 

            cudaMalloc((void**)&dev_keys_found[g], sizeof(int));        
            cudaMalloc((void**)&dev_executions_this_gpu[g], sizeof(int));       

            vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], dev_g, dev_executions_this_gpu[g]);

        }

        // Synchronize while we wait for kernels to complete. I do not
        // actually know if this will sync against all GPUs, it might
        // just sync with the last `i`, but they should all complete
        // roughly at the same time and worst case it will just stack
        // up kernels in the queue to run.
        cudaDeviceSynchronize();
        auto finish = std::chrono::high_resolution_clock::now();

        for (int g = 0; g < gpuCount; ++g) {
            cudaMemcpy( &keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost ); 
            keys_found_total += keys_found_this_iteration; 
            //printf("GPU %d found %d keys\n",g,keys_found_this_iteration);

            cudaMemcpy( &executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost ); 
            executions_this_iteration += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
            executions_total += executions_this_gpu * ATTEMPTS_PER_EXECUTION; 
            //printf("GPU %d executions: %d\n",g,executions_this_gpu);
        }

        // Print out performance Summary
        std::chrono::duration<double> elapsed = finish - start;
        printf("%s Iteration %d Attempts: %llu in %f at %fcps - Total Attempts %llu - keys found %d\n",
            getTimeStr().c_str(),
            i+1,
            executions_this_iteration,
            elapsed.count(),
            executions_this_iteration / elapsed.count(),
            executions_total,
            keys_found_total
        );

        if ( keys_found_total >= STOP_AFTER_KEYS_FOUND ) {
            printf("Enough keys found, Done! \n");
            exit(0);    
        }   
    }

    printf("Iterations complete, Done!\n");
}

/* -- CUDA Vanity Functions ------------------------------------------------- */

void __global__ vanity_scan(unsigned char* state, int* keys_found, int* gpu, int* exec_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x +
        (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
        (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;

    atomicAdd(exec_count, 1);

    // SMITH - should really be passed in, but hey ho
    int prefix_letter_counts[MAX_PATTERNS];
    for (unsigned int n = 0; n < sizeof(prefixes) / sizeof(prefixes[0]); ++n) {
        if (MAX_PATTERNS == n) {
            printf("NEVER SPEAK TO ME OR MY SON AGAIN");
            return;
        }
        int letter_count = 0;
        for (; prefixes[n][letter_count] != 0; letter_count++);
        prefix_letter_counts[n] = letter_count;
    }
    char prefix_ignore_case_char_masks[64];
    for (int i = 0; i < 64; i++) prefix_ignore_case_char_masks[i] = 0xff;
    for (int i = 0; prefix_ignore_case_mask[i] != 0; i++) {
        prefix_ignore_case_char_masks[i] ^= (prefix_ignore_case_mask[i] == 64) << 5;
    }
    
    // Local Kernel State
    ge_p3 A;
    unsigned int seed_limbs[8] = {0};
    unsigned char seed[32]     = {0};
    unsigned char publick[32]  = {0};
    unsigned char privatek[64] = {0};
    char key[256]              = {0};

    memcpy(seed_limbs, state + id * 32, 32);

    // Generate Random Key Data
    sha512_context md;

    // I've unrolled all the MD5 calls and special cased them to 32 byte
    // inputs, which eliminates a lot of branching. This is a pretty poor
    // way to optimize GPU code though.
    //
    // A better approach would be to split this application into two
    // different kernels, one that is warp-efficient for SHA512 generation,
    // and another that is warp efficient for bignum division to more
    // efficiently scan for prefixes. Right now bs58enc cuts performance
    // from 16M keys on my machine per second to 4M.
    for (int attempts = 0; attempts < ATTEMPTS_PER_EXECUTION; ++attempts) {
        memcpy(seed, seed_limbs, 32);
        // sha512_init Inlined
        md.curlen   = 0;
        md.length   = 0;
        md.state[0] = UINT64_C(0x6a09e667f3bcc908);
        md.state[1] = UINT64_C(0xbb67ae8584caa73b);
        md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
        md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
        md.state[4] = UINT64_C(0x510e527fade682d1);
        md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
        md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
        md.state[7] = UINT64_C(0x5be0cd19137e2179);

        // sha512_update inlined
        // 
        // All `if` statements from this function are eliminated if we
        // will only ever hash a 32 byte seed input. So inlining this
        // has a drastic speed improvement on GPUs.
        //
        // This means:
        //   * Normally we iterate for each 128 bytes of input, but we are always < 128. So no iteration.
        //   * We can eliminate a MIN(inlen, (128 - md.curlen)) comparison, specialize to 32, branch prediction improvement.
        //   * We can eliminate the in/inlen tracking as we will never subtract while under 128
        //   * As a result, the only thing update does is copy the bytes into the buffer.
        memcpy(md.buf + md.curlen, seed, 32);
        md.curlen += 32;

        // sha512_final inlined
        // 
        // As update was effectively elimiated, the only time we do
        // sha512_compress now is in the finalize function. We can also
        // optimize this:
        //
        // This means:
        //   * We don't need to care about the curlen > 112 check. Eliminating a branch.
        //   * We only need to run one round of sha512_compress, so we can inline it entirely as we don't need to unroll.
        md.length += md.curlen * UINT64_C(8);
        md.buf[md.curlen++] = (unsigned char)0x80;

        #pragma unroll
        while (md.curlen < 120) {
            md.buf[md.curlen++] = (unsigned char)0;
        }

        STORE64H(md.length, md.buf+120);

        // Inline sha512_compress
        uint64_t S[8], W[80], t0, t1;
        int i;

        /* Copy state into S */
        #pragma unroll
        for (i = 0; i < 8; i++) {
            S[i] = md.state[i];
        }

        /* Copy the state into 1024-bits into W[0..15] */
        #pragma unroll
        for (i = 0; i < 16; i++) {
            LOAD64H(W[i], md.buf + (8*i));
        }

        /* Fill W[16..79] */
        #pragma unroll
        for (i = 16; i < 80; i++) {
            W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
        }

        /* Compress */
        #define RND(a,b,c,d,e,f,g,h,i) \
        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
        t1 = Sigma0(a) + Maj(a, b, c);\
        d += t0; \
        h  = t0 + t1;

        #pragma unroll
        for (i = 0; i < 80; i += 8) {
            RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],i+0);
            RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],i+1);
            RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],i+2);
            RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],i+3);
            RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],i+4);
            RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],i+5);
            RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],i+6);
            RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],i+7);
        }

        #undef RND

        /* Feedback */
        #pragma unroll
        for (i = 0; i < 8; i++) {
            md.state[i] = md.state[i] + S[i];
        }

        // We can now output our finalized bytes into the output buffer.
        #pragma unroll
        for (i = 0; i < 8; i++) {
            STORE64H(md.state[i], privatek+(8*i));
        }

        // Code Until here runs at 87_000_000H/s.

        // ed25519 Hash Clamping
        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        // ed25519 curve multiplication to extract a public key.
        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(publick, &A);

        // Code Until here runs at 87_000_000H/s still!

        b58enc(key, publick);

        // Code Until here runs at 22_000_000H/s. b58enc badly needs optimization.

        // We don't have access to strncmp/strlen here, I don't know
        // what the efficient way of doing this on a GPU is, so I'll
        // start with a dumb loop. There seem to be implementations out
        // there of bignunm division done in parallel as a CUDA kernel
        // so it might make sense to write a new parallel kernel to do
        // this.

        #define CONDITIONAL_CASE_CHAR_EQ(a, b, j) ((prefix_ignore_case_char_masks[j] & (a[j] ^ b[j])) == 0)

        for (int i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {

            for (int j = 0; j < prefix_letter_counts[i]; ++j) {
                // it doesn't match this prefix, no need to continue
                if (!CONDITIONAL_CASE_CHAR_EQ(prefixes[i], key, j) &&!(prefixes[i][j] == '?')) {
                    break;
                }

                // we got to the end of the prefix pattern, it matched!
                if (j == ( prefix_letter_counts[i] - 1)) {
                    atomicAdd(keys_found, 1);
                    //size_t pkeysize = 256;
                    //b58enc(pkey, &pkeysize, seed, 32);
                                       
                    // SMITH    
                    // The 'key' variable is the public key in base58 'address' format
                    // We display the seed in hex

                    // Solana stores the keyfile as seed (first 32 bytes)
                    // followed by public key (last 32 bytes)
                    // as an array of decimal numbers in json format

                    printf("GPU %d MATCH %s - ", *gpu, key);
                    for (int n = 0; n < sizeof(seed); n++) { 
                        printf("%02x", (unsigned char)seed[n]); 
                    }
                    printf("\n");
                    
                    printf("[");
                    for (int n = 0; n < sizeof(seed); n++) { 
                        printf("%d,", (unsigned char)seed[n]);
                    }
                    for (int n = 0; n < sizeof(publick); n++) {
                        if (n + 1 == sizeof(publick)) {   
                            printf("%d",publick[n]);
                        } else {
                            printf("%d,",publick[n]);
                        }
                    }
                    printf("]\n");

                    /*
                    printf("Public: ");
                                        for(int n=0; n<sizeof(publick); n++) { printf("%d ",publick[n]); }
                    printf("\n");
                    printf("Private: ");
                                        for(int n=0; n<sizeof(privatek); n++) { printf("%d ",privatek[n]); }
                    printf("\n");
                    printf("Seed: ");
                                        for(int n=0; n<sizeof(seed); n++) { printf("%d ",seed[n]); }
                    printf("\n");
                    */

                    break;
                }
            }
        }

        // Code Until here runs at 22_000_000H/s. So the above is fast enough.

        // Increment Seed.
        for (int i = 0; i < 8; ++i) {
            if (++seed_limbs[i] != 0) break;
        }
    }

    // Copy Random State so that future calls of this kernel/thread/block
    // don't repeat their sequences.
    memcpy(state + id * 32, seed_limbs, 32);
}

// Modified from https://github.com/firedancer-io/firedancer/tree/main/src/ballet/base58
void __device__ b58enc(
    char    *b58,
    const uint8_t *data
) {
    #define BINARY_SZ 8
    #define INTERMEDIATE_SZ 9    

    unsigned int binary[BINARY_SZ];
    memcpy(binary, data, 32);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        binary[i] = __byte_perm(binary[i], 0, 0x0123);
    }
    
    #define R1_DIV 656356768UL
    #define RAW58_SZ (INTERMEDIATE_SZ * 5)
    
    unsigned int in_leading_0s = __clz(binary[0]) >> 3;
    if (in_leading_0s == 4) {
        for (; in_leading_0s < 32; in_leading_0s++) if (data[in_leading_0s]) break;    
    }
    
    unsigned long intermediate[INTERMEDIATE_SZ] = {0};
    
    intermediate[1] += (unsigned long) binary[0] * (unsigned long) 513735UL;
    intermediate[2] += (unsigned long) binary[0] * (unsigned long) 77223048UL;
    intermediate[3] += (unsigned long) binary[0] * (unsigned long) 437087610UL;
    intermediate[4] += (unsigned long) binary[0] * (unsigned long) 300156666UL;
    intermediate[5] += (unsigned long) binary[0] * (unsigned long) 605448490UL;
    intermediate[6] += (unsigned long) binary[0] * (unsigned long) 214625350UL;
    intermediate[7] += (unsigned long) binary[0] * (unsigned long) 141436834UL;
    intermediate[8] += (unsigned long) binary[0] * (unsigned long) 379377856UL;
    intermediate[2] += (unsigned long) binary[1] * (unsigned long) 78508UL;
    intermediate[3] += (unsigned long) binary[1] * (unsigned long) 646269101UL;
    intermediate[4] += (unsigned long) binary[1] * (unsigned long) 118408823UL;
    intermediate[5] += (unsigned long) binary[1] * (unsigned long) 91512303UL;
    intermediate[6] += (unsigned long) binary[1] * (unsigned long) 209184527UL;
    intermediate[7] += (unsigned long) binary[1] * (unsigned long) 413102373UL;
    intermediate[8] += (unsigned long) binary[1] * (unsigned long) 153715680UL;
    intermediate[3] += (unsigned long) binary[2] * (unsigned long) 11997UL;
    intermediate[4] += (unsigned long) binary[2] * (unsigned long) 486083817UL;
    intermediate[5] += (unsigned long) binary[2] * (unsigned long) 3737691UL;
    intermediate[6] += (unsigned long) binary[2] * (unsigned long) 294005210UL;
    intermediate[7] += (unsigned long) binary[2] * (unsigned long) 247894721UL;
    intermediate[8] += (unsigned long) binary[2] * (unsigned long) 289024608UL;
    intermediate[4] += (unsigned long) binary[3] * (unsigned long) 1833UL;
    intermediate[5] += (unsigned long) binary[3] * (unsigned long) 324463681UL;
    intermediate[6] += (unsigned long) binary[3] * (unsigned long) 385795061UL;
    intermediate[7] += (unsigned long) binary[3] * (unsigned long) 551597588UL;
    intermediate[8] += (unsigned long) binary[3] * (unsigned long) 21339008UL;
    intermediate[5] += (unsigned long) binary[4] * (unsigned long) 280UL;
    intermediate[6] += (unsigned long) binary[4] * (unsigned long) 127692781UL;
    intermediate[7] += (unsigned long) binary[4] * (unsigned long) 389432875UL;
    intermediate[8] += (unsigned long) binary[4] * (unsigned long) 357132832UL;
    intermediate[6] += (unsigned long) binary[5] * (unsigned long) 42UL;
    intermediate[7] += (unsigned long) binary[5] * (unsigned long) 537767569UL;
    intermediate[8] += (unsigned long) binary[5] * (unsigned long) 410450016UL;
    intermediate[7] += (unsigned long) binary[6] * (unsigned long) 6UL;
    intermediate[8] += (unsigned long) binary[6] * (unsigned long) 356826688UL;
    intermediate[8] += (unsigned long) binary[7] * (unsigned long) 1UL;
    
    intermediate[7] += intermediate[8] / R1_DIV;
    intermediate[8] %= R1_DIV;
    intermediate[6] += intermediate[7] / R1_DIV;
    intermediate[7] %= R1_DIV;
    intermediate[5] += intermediate[6] / R1_DIV;
    intermediate[6] %= R1_DIV;
    intermediate[4] += intermediate[5] / R1_DIV;
    intermediate[5] %= R1_DIV;
    intermediate[3] += intermediate[4] / R1_DIV;
    intermediate[4] %= R1_DIV;
    intermediate[2] += intermediate[3] / R1_DIV;
    intermediate[3] %= R1_DIV;
    intermediate[1] += intermediate[2] / R1_DIV;
    intermediate[2] %= R1_DIV;
    intermediate[0] += intermediate[1] / R1_DIV;
    intermediate[1] %= R1_DIV;
    
    uint_fast8_t raw_base58[64];
    
    #pragma unroll
    for (int i = 0; i < INTERMEDIATE_SZ; ++i) {
        raw_base58[5 * i + 4] = ((((unsigned int) intermediate[i]) / 1U       ) % 58U);
        raw_base58[5 * i + 3] = ((((unsigned int) intermediate[i]) / 58U      ) % 58U);
        raw_base58[5 * i + 2] = ((((unsigned int) intermediate[i]) / 3364U    ) % 58U);
        raw_base58[5 * i + 1] = ((((unsigned int) intermediate[i]) / 195112U  ) % 58U);
        raw_base58[5 * i + 0] = ( ((unsigned int) intermediate[i]) / 11316496U);    
    }
    unsigned int raw_leading_0s;
    memcpy(&raw_leading_0s, raw_base58, 4);
    raw_leading_0s = __clz(__byte_perm(raw_leading_0s, 0, 0x0123)) >> 3;

    if (raw_leading_0s == 4) {
        for (; raw_leading_0s < RAW58_SZ; raw_leading_0s++) if (raw_base58[raw_leading_0s]) break;    
    }

    unsigned int skip = raw_leading_0s - in_leading_0s;
    
    static uint_fast8_t const b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (int i = 0; i < RAW58_SZ - skip; i++) b58[i] = b58digits_ordered[raw_base58[skip + i]];
    b58[RAW58_SZ - skip] = 0;
}
