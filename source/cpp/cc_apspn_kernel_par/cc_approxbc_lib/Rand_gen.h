// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#ifndef RAND_GEN_H
#define RAND_GEN_H

#include <cstdint>
#include <random>

namespace cc_approxbc_lib {

class Rand_gen
{
    public:
        Rand_gen(const uint64_t seed);
        virtual ~Rand_gen();
        uint64_t get();
        uint64_t get_max( const uint64_t i );
    protected:
    private:
        std::mt19937_64 engine;
};

} // namespace cc_approxbc_lib

#endif // RAND_GEN_H
