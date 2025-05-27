#pragma once

#include <vector>

namespace mem {
int gather_chw(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<int>& in_shape,
               const std::vector<int>& indices, int axis);
}