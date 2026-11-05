// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#include "Rand_gen.h"
#include <random>

namespace cc_approxbc_lib {

Rand_gen::Rand_gen( uint64_t seed )
{
    this->engine.seed(seed);
}

Rand_gen::~Rand_gen()
{
}

// Returns a random number using std::mt19937_64
uint64_t Rand_gen::get() {
    return engine();
}

// Return a random number between 0 and m-1 (included).
// Uses rejection sampling to avoid modulo bias.
uint64_t Rand_gen::get_max( uint64_t m ) {
    uint64_t r;
    uint64_t threshold = UINT64_MAX - UINT64_MAX % m;
    while( (r = get()) > threshold ) {}
    return r % m;
}

} // namespace cc_approxbc_lib
