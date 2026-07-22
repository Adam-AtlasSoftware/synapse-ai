#include "protocol.hpp"

#include <cstdio>
#include <sstream>

namespace synapse::proto {
namespace {

// %.6g keeps the stream small while staying well past display precision.
void put_float(std::ostringstream& os, float v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(v));
  os << buf;
}

void put_floats(std::ostringstream& os, const std::vector<float>& v) {
  os << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) os << ',';
    put_float(os, v[i]);
  }
  os << ']';
}

}  // namespace

std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string to_json(const Topology& t) {
  std::ostringstream os;
  os << "{\"name\":\"" << escape(t.name) << "\",\"input_dim\":" << t.input_dim << ",\"layers\":[";
  for (size_t k = 0; k < t.layers.size(); ++k) {
    const LayerInfo& L = t.layers[k];
    if (k) os << ',';
    os << "{\"name\":\"" << escape(L.name) << "\",\"type\":\"" << escape(L.type)
       << "\",\"input_dim\":" << L.input_dim << ",\"output_dim\":" << L.output_dim
       << ",\"activation\":\"" << escape(L.activation) << "\"}";
  }
  os << "]}";
  return os.str();
}

std::string to_json(const StepSnapshot& s) {
  std::ostringstream os;
  os << "{\"step\":" << s.step << ",\"epoch\":" << s.epoch << ",\"loss\":";
  put_float(os, s.loss);
  os << ",\"phase\":\"" << escape(s.phase) << "\",\"active_layer\":" << s.active_layer
     << ",\"tensors\":[";
  for (size_t i = 0; i < s.tensors.size(); ++i) {
    const TensorView& tv = s.tensors[i];
    if (i) os << ',';
    os << "{\"name\":\"" << escape(tv.name) << "\",\"rows\":" << tv.rows
       << ",\"cols\":" << tv.cols << ",\"data\":";
    put_floats(os, tv.data);
    os << '}';
  }
  os << "]}";
  return os.str();
}

std::string event_topology(const Topology& t) {
  return "{\"ev\":\"topology\",\"topology\":" + to_json(t) + "}";
}

std::string event_step(const StepSnapshot& s) {
  return "{\"ev\":\"step\",\"step\":" + to_json(s) + "}";
}

std::string event_error(long id, const std::string& message) {
  return "{\"ev\":\"error\",\"id\":" + std::to_string(id) + ",\"message\":\"" + escape(message) +
         "\"}";
}

}  // namespace synapse::proto
