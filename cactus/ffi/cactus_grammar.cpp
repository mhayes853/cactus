#include "cactus_ffi.h"
#include "cactus_utils.h"

using namespace cactus::ffi;

struct CactusGrammarHandle {

};

extern "C" {

cactus_grammar_t cactus_grammar_init_gbnf(const char* ebnf, const char* start_symbol) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_json() {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_empty() {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_json_schema(
    const char* json_schema,
    bool any_whitespace,
    int indent,
    const char*** separators,
    size_t separators_count,
    bool strict_mode,
    int max_whitespace_count
) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_regex(const char* regex) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_structural_tag(const char* structural_tag_json) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_union(cactus_grammar_t* grammars, size_t num_grammars) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_concatenate(cactus_grammar_t* grammars, size_t num_grammars) {
    return nullptr;
}

void cactus_grammar_destroy(cactus_grammar_t grammar) {
}

}
