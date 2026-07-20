#pragma once
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// A tiny, dependency-free JSON reader — just enough for model-spec files. Kept
// internal to the engine. Preserves object key order (handy for round-tripping).
namespace synapse::json {

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool bool_v = false;
  double num_v = 0.0;
  std::string str_v;
  std::vector<Value> arr_v;
  std::vector<std::pair<std::string, Value>> obj_v;

  bool is_object() const { return type == Type::Object; }
  bool is_array() const { return type == Type::Array; }

  // Object lookup; nullptr if absent or not an object.
  const Value* find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& kv : obj_v)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  // Required field; throws with context if missing.
  const Value& at(const std::string& key) const {
    if (const Value* v = find(key)) return *v;
    throw std::runtime_error("JSON: missing required key '" + key + "'");
  }
  double as_number() const {
    if (type != Type::Number) throw std::runtime_error("JSON: expected a number");
    return num_v;
  }
  std::string string_or(const std::string& key, const std::string& fallback) const {
    const Value* v = find(key);
    return (v && v->type == Type::String) ? v->str_v : fallback;
  }
};

Value parse(const std::string& text);  // throws std::runtime_error on malformed input

}  // namespace synapse::json
