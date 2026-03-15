#include "grammar.h"
#include <memory>

namespace cactus {
namespace grammar {

Grammar::Grammar() : grammar(nullptr) {}

Grammar Grammar::json() {
    Grammar g;
    g.grammar = std::make_shared<xgrammar::Grammar>(xgrammar::Grammar::BuiltinJSONGrammar());
    return g;
}

std::shared_ptr<xgrammar::Grammar> Grammar::handle() const {
    return grammar;
}

}
}
