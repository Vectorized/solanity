#ifndef VANITY_CONFIG
#define VANITY_CONFIG

import * as fs from 'fs';

static int const MAX_ITERATIONS = 1000000000;
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
__device__ const int ATTEMPTS_PER_EXECUTION = 100000;

__device__ const int MAX_PATTERNS = 10;

// exact matches at the beginning of the address, letter ? is wildcard
__device__ static char const *prefixes[] = {
	// "De1eg",
	// "De1ega",
	// "De1egat",
	"De1egate"
};

// "_" to denote exact case
// "@" to denote case insensitive
__device__ static char const *prefix_ignore_case_mask = "@@@@@@@@";

async function handleResult(result: any, publicKey: string) {
    // Create filename using the public key
    const filename = `${publicKey}.json`;
    
    try {
        // Convert result to JSON string with pretty formatting
        const jsonData = JSON.stringify(result, null, 2);
        
        // Write to file asynchronously
        await fs.promises.writeFile(filename, jsonData);
        console.log(`Results saved to ${filename}`);
    } catch (error) {
        console.error('Error saving results:', error);
    }
}
#endif
