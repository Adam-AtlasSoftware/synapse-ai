#include "synapse/telemetry.hpp"

#include <algorithm>
#include <cctype>

namespace synapse {

std::string to_string(Activation a) {
  switch (a) {
    case Activation::Linear: return "linear";
    case Activation::Sigmoid: return "sigmoid";
    case Activation::ReLU: return "relu";
    case Activation::Tanh: return "tanh";
    case Activation::Softmax: return "softmax";
  }
  return "sigmoid";
}

Activation activation_from_string(const std::string& s) {
  std::string k = s;
  std::transform(k.begin(), k.end(), k.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (k == "linear" || k == "none" || k == "identity") return Activation::Linear;
  if (k == "sigmoid" || k == "logistic") return Activation::Sigmoid;
  if (k == "relu") return Activation::ReLU;
  if (k == "tanh") return Activation::Tanh;
  if (k == "softmax") return Activation::Softmax;
  return Activation::Sigmoid;  // safe default
}

}  // namespace synapse
