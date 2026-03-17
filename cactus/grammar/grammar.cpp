#include "grammar.h"
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace cactus {
namespace grammar {

Grammar::Grammar() : grammar(nullptr) {}

Grammar::Grammar(xgrammar::Grammar raw_grammar)
    : grammar(std::make_shared<xgrammar::Grammar>(std::move(raw_grammar))) {}

Grammar Grammar::gbnf(const std::string& gbnf, const std::string& start_symbol) {
    return Grammar(xgrammar::Grammar::FromEBNF(gbnf, start_symbol));
}

Grammar Grammar::json() {
    return Grammar(xgrammar::Grammar::BuiltinJSONGrammar());
}

Grammar Grammar::json_schema(
    const std::string& json_schema,
    bool any_whitespace,
    int indent,
    std::pair<std::string, std::string> separators,
    bool strict_mode,
    int max_whitespace_count
) {
    return Grammar(xgrammar::Grammar::FromJSONSchema(
        json_schema,
        any_whitespace,
        indent,
        separators,
        strict_mode,
        max_whitespace_count >= 0 ? std::optional<int>(max_whitespace_count) : std::nullopt
    ));
}

Grammar Grammar::regex(const std::string& regex) {
    return Grammar(xgrammar::Grammar::FromRegex(regex));
}

Grammar Grammar::structural_tag(const std::string& structural_tag_json) {
    auto result = xgrammar::Grammar::FromStructuralTag(structural_tag_json);
    if (std::holds_alternative<xgrammar::Grammar>(result)) {
        return Grammar(std::get<xgrammar::Grammar>(std::move(result)));
    }
    throw std::get<xgrammar::StructuralTagError>(result);
}

Grammar Grammar::unite(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) {
            continue;
        }
        handles.push_back(*grammar.handle());
    }

    if (handles.empty()) {
        return Grammar();
    }
    return Grammar(xgrammar::Grammar::Union(handles));
}

Grammar Grammar::concatenate(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) {
            continue;
        }
        handles.push_back(*grammar.handle());
    }

    if (handles.empty()) {
        return Grammar();
    }
    return Grammar(xgrammar::Grammar::Concat(handles));
}

bool Grammar::is_empty() const {
    return grammar == nullptr;
}

std::shared_ptr<xgrammar::Grammar> Grammar::handle() const {
    return grammar;
}

}
}
