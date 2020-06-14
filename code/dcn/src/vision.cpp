
#include "deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward", &deform_conv_backward, "deform_conv_backward");
}
