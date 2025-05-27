#include "gather_chw.h"

#include <cstring>
#include <numeric>

namespace mem {
int gather_chw(std::vector<float>& output, const std::vector<float>& input,
               const std::vector<int>& in_shape,
               const std::vector<int>& indices, int axis) {
  size_t outer_count = std::accumulate(
      in_shape.begin(), in_shape.begin() + axis, 1, std::multiplies<size_t>{});
  size_t indices_count = indices.size();
  size_t block_size =
      std::accumulate(in_shape.begin() + axis + 1, in_shape.end(), 1,
                      std::multiplies<size_t>{});
  int output_size = input.size() * indices_count / in_shape[axis];
  output.resize(output_size);
  auto* in_ptr = input.data();
  auto* out_ptr = output.data();

  for (size_t o = 0; o < outer_count; ++o) {
    for (size_t i = 0; i < indices_count; ++i) {
      auto* o_ptr = out_ptr + i * block_size;
      auto indices_ptr =
          indices[i] >= 0 ? indices[i] : indices[i] + in_shape[axis];
      memcpy(o_ptr, in_ptr + (indices_ptr * block_size),
             block_size * sizeof(float));
    }
    in_ptr += in_shape[axis] * block_size;
    out_ptr += indices_count * block_size;
  }
  return 0;
}
}  // namespace mem