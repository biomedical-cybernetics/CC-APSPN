// Derived from SILVAN by L. Pellegrina and F. Vandin
// (https://github.com/leonardopellegrina/SILVAN), which builds on KADABRA by
// M. Borassi and E. Natale. Licensed under the Apache License 2.0; see
// THIRD_PARTY_NOTICES.md in the CC-APSPN release. Modified for CC-APSPN.
#ifndef DUMB_LIST_H
#define DUMB_LIST_H

#include <cstdint>

namespace cc_approxbc_lib {

class Ranking_list
{
    public:
        void print_list();
        double get_value( const uint32_t i ) const {return values[i];}
        Ranking_list( const uint32_t k);
        virtual ~Ranking_list();
        void put(const uint32_t element, const double new_value);
        uint32_t get(int i) { return elements[i]; }
        const uint32_t k;
    protected:
    private:
        uint32_t *elements;
        double *values;
};

} // namespace cc_approxbc_lib

#endif // DUMB_LIST_H
