#include "synapse/model_spec.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "synapse/activation.hpp"
#include "json.hpp"

namespace synapse {

Topology parse_topology_json(const std::string& text) {
  json::Value root = json::parse(text);
  if (!root.is_object()) throw std::runtime_error("model spec must be a JSON object");

  Topology t;
  t.name = root.string_or("name", "model");
  t.input_dim = static_cast<int>(root.at("input_dim").as_number());
  if (t.input_dim <= 0) throw std::runtime_error("input_dim must be positive");

  const json::Value& layers = root.at("layers");
  if (!layers.is_array()) throw std::runtime_error("'layers' must be an array");

  int prev = t.input_dim;
  int index = 0;
  for (const json::Value& lv : layers.arr_v) {
    LayerInfo li;
    li.type = lv.string_or("type", "dense");
    const int units = static_cast<int>(lv.at("units").as_number());
    if (units <= 0) throw std::runtime_error("layer 'units' must be positive");
    li.output_dim = units;
    li.input_dim = prev;
    std::string act = lv.string_or("activation", "sigmoid");
    std::transform(act.begin(), act.end(), act.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    li.activation = activation_exists(act) ? act : "sigmoid";  // unknown → safe default
    li.name = lv.string_or("name", "L" + std::to_string(index));
    t.layers.push_back(std::move(li));
    prev = units;
    ++index;
  }
  if (t.layers.empty()) throw std::runtime_error("model must have at least one layer");
  return t;
}

Topology load_topology_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open model spec: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_topology_json(ss.str());
}

std::string topology_to_json(const Topology& t) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"name\": \"" << t.name << "\",\n";
  os << "  \"input_dim\": " << t.input_dim << ",\n";
  os << "  \"layers\": [\n";
  for (size_t k = 0; k < t.layers.size(); ++k) {
    const LayerInfo& L = t.layers[k];
    os << "    { \"type\": \"" << L.type << "\", \"units\": " << L.output_dim
       << ", \"activation\": \"" << L.activation << "\" }"
       << (k + 1 < t.layers.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return os.str();
}

void save_topology_file(const Topology& t, const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write model spec: " + path);
  out << topology_to_json(t);
}

namespace {
std::vector<float> read_float_array(const json::Value& v) {
  std::vector<float> out;
  if (v.is_array())
    for (const json::Value& x : v.arr_v) out.push_back(static_cast<float>(x.as_number()));
  return out;
}
}  // namespace

Dataset parse_dataset_json(const std::string& text) {
  Dataset ds;
  json::Value root = json::parse(text);
  const json::Value* d = root.find("dataset");
  if (!d || !d->is_array()) return ds;  // no training data is fine
  for (const json::Value& s : d->arr_v) {
    Sample sample;
    if (const json::Value* in = s.find("input")) sample.input = read_float_array(*in);
    if (const json::Value* tg = s.find("target")) sample.target = read_float_array(*tg);
    ds.samples.push_back(std::move(sample));
  }
  return ds;
}

Dataset load_dataset_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open dataset: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_dataset_json(ss.str());
}

}  // namespace synapse
