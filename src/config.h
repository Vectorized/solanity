#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 1000000000;
static int const STOP_AFTER_KEYS_FOUND = 1;

// how many times a gpu thread generates a public key in one go
__device__ const int ATTEMPTS_PER_EXECUTION = 100000;

__device__ const int MAX_PATTERNS = 10;

// exact matches at the beginning of the address, letter ? is wildcard
__device__ static char const *prefixes[] = {
	"De1eg",
	"de1egate",
	"De1egate",
	"de1egatexyz",
	"De1egatexyz",
	"De1egateXYZ"
};

#endif
