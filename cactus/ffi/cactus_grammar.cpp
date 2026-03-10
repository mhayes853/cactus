#include "cactus_ffi.h"
#include "cactus_utils.h"

using namespace cactus::ffi;

struct CactusGrammarHandle {

};

struct CactusCompiledGrammarHandle {

};

extern "C" {

cactus_grammar_t cactus_grammar_init_ebnf(const char* ebnf, const char* start_symbol) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_json() {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_json_schema(const char* json_schema) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_init_regex(const char* regex) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_union(cactus_grammar_t g1, cactus_grammar_t g2) {
    return nullptr;
}

cactus_grammar_t cactus_grammar_concatenate(cactus_grammar_t g1, cactus_grammar_t g2) {
    return nullptr;
}

void cactus_grammar_destroy(cactus_grammar_t grammar) {
}

cactus_compiled_grammar_t cactus_grammar_compile(cactus_grammar_t grammar) {
    return nullptr;
}

void cactus_grammar_destroy_compiled(cactus_compiled_grammar_t compiled_grammar) {
}

}
