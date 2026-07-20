#include "json.hpp"

#include <cctype>

namespace synapse::json {
namespace {

// A small recursive-descent parser. Not a conformance-chasing implementation, but
// correct for well-formed JSON of the kind our model specs use.
struct Parser {
  const std::string& s;
  size_t i = 0;
  explicit Parser(const std::string& str) : s(str) {}

  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("JSON parse error at offset " + std::to_string(i) + ": " + msg);
  }
  void skip_ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  char peek() { skip_ws(); return i < s.size() ? s[i] : '\0'; }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++i;
  }

  Value parse_value() {
    char c = peek();
    switch (c) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': {
        Value v;
        v.type = Value::Type::String;
        v.str_v = parse_string();
        return v;
      }
      case 't':
      case 'f': return parse_bool();
      case 'n': return parse_null();
      default:
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        fail("unexpected character");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) fail("bad escape");
      char e = s[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          if (i + 4 > s.size()) fail("bad \\u escape");
          int code = std::stoi(s.substr(i, 4), nullptr, 16);
          i += 4;
          if (code < 0x80) {
            out += static_cast<char>(code);
          } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
          }
          break;
        }
        default: fail("invalid escape");
      }
    }
    return out;
  }

  Value parse_number() {
    skip_ws();
    size_t start = i;
    if (peek() == '-') ++i;
    while (i < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' ||
            s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
      ++i;
    }
    Value v;
    v.type = Value::Type::Number;
    v.num_v = std::stod(s.substr(start, i - start));
    return v;
  }

  Value parse_bool() {
    Value v;
    v.type = Value::Type::Bool;
    if (s.compare(i, 4, "true") == 0) {
      v.bool_v = true;
      i += 4;
    } else if (s.compare(i, 5, "false") == 0) {
      v.bool_v = false;
      i += 5;
    } else {
      fail("invalid literal");
    }
    return v;
  }

  Value parse_null() {
    if (s.compare(i, 4, "null") != 0) fail("invalid literal");
    i += 4;
    return Value{};
  }

  Value parse_array() {
    expect('[');
    Value v;
    v.type = Value::Type::Array;
    if (peek() == ']') {
      ++i;
      return v;
    }
    while (true) {
      v.arr_v.push_back(parse_value());
      char c = peek();
      if (c == ',') {
        ++i;
        continue;
      }
      expect(']');
      break;
    }
    return v;
  }

  Value parse_object() {
    expect('{');
    Value v;
    v.type = Value::Type::Object;
    if (peek() == '}') {
      ++i;
      return v;
    }
    while (true) {
      if (peek() != '"') fail("expected object key");
      std::string key = parse_string();
      expect(':');
      v.obj_v.emplace_back(std::move(key), parse_value());
      char c = peek();
      if (c == ',') {
        ++i;
        continue;
      }
      expect('}');
      break;
    }
    return v;
  }
};

}  // namespace

Value parse(const std::string& text) {
  Parser p(text);
  Value v = p.parse_value();
  if (p.peek() != '\0') p.fail("trailing characters after JSON value");
  return v;
}

}  // namespace synapse::json
