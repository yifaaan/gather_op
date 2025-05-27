#pragma once

#include <vector>

namespace rvv {
int gather_hwc(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<int>& in_shape_hwc,
               const std::vector<int>& indices, int axis_chw,
               int align_channels);
}

namespace mem {
int gather_hwc(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<int>& in_shape_hwc,
               const std::vector<int>& indices, int axis_chw,
               int align_channels);
}
