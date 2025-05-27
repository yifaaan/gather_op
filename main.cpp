#include <riscv_vector.h>  // 包含 RISC-V Vector intrinsics 的头文件
#include <stdio.h>

#include "op.h"

int main() {
  // std::cout << "memcpy:" << '\n';
  // TestGatherHWC(false, {64, 1, 128}, {1, 128}, "64_1_128__1_128/input.txt",
  // "64_1_128__1_128/indices.txt", "64_1_128__1_128/output.txt", 0,
  // "64_1_128__1_128/true_output.txt"); TestGatherHWC(false, {64, 1, 256}, {1,
  // 256}, "64_1_256__1_256/input.txt", "64_1_256__1_256/indices.txt",
  // "64_1_256__1_256/output.txt", 0, "64_1_256__1_256/true_output.txt");
  // TestGatherHWC(false, {60, 4, 64}, {1}, "60_4_64_1/input.txt",
  // "60_4_64_1/indices.txt", "60_4_64_1/output.txt", 0,
  // "60_4_64_1/true_output.txt"); TestGatherHWC(false, {128, 64, 16}, {32},
  // "128_64_16__32/input.txt", "128_64_16__32/indices.txt",
  // "128_64_16__32/output.txt", 1, "128_64_16__32/true_output.txt");
  // TestGatherHWC(false, {32, 2, 208}, {1, 200}, "32_2_208__1_200/input.txt",
  // "32_2_208__1_200/indices.txt", "32_2_208__1_200/output.txt", 0,
  // "32_2_208__1_200/true_output.txt");

  // TestGatherHWC(false, {3, 8, 4, 16}, {1}, "3_8_4_16__1/input.txt",
  // "3_8_4_16__1/indices.txt", "3_8_4_16__1/output.txt", 0,
  // "3_8_4_16__1/true_output.txt"); TestGatherHWC(false, {1, 4, 4, 16}, {1},
  // "1_4_4_16__1/input.txt", "1_4_4_16__1/indices.txt",
  // "1_4_4_16__1/output.txt", 2, "1_4_4_16__1/true_output.txt");
  // TestGatherHWC(false, {1, 4, 4, 16}, {7, 2}, "1_4_4_16__7_2/input.txt",
  // "1_4_4_16__7_2/indices.txt", "1_4_4_16__7_2/output.txt", 2,
  // "1_4_4_16__7_2/true_output.txt"); TestGatherHWC(false, {1, 8, 8, 160}, {3,
  // 2}, "1_8_8_160__3_2/input.txt", "1_8_8_160__3_2/indices.txt",
  // "1_8_8_160__3_2/output.txt", 2, "1_8_8_160__3_2/true_output.txt");

  // TestGatherHWC(false, {1, 16, 6, 4, 96}, {1}, "1_16_6_4_96__1/input.txt",
  // "1_16_6_4_96__1/indices.txt", "1_16_6_4_96__1/output.txt", 3,
  // "1_16_6_4_96__1/true_output.txt"); TestGatherHWC(false, {4, 1, 8, 2, 160},
  // {1}, "4_1_8_2_160__1/input.txt", "4_1_8_2_160__1/indices.txt",
  // "4_1_8_2_160__1/output.txt", 4, "4_1_8_2_160__1/true_output.txt");
  // TestGatherHWC(false, {1, 2, 4, 8, 160}, {3, 5},
  // "1_2_4_8_160__3_5/input.txt", "1_2_4_8_160__3_5/indices.txt",
  // "1_2_4_8_160__3_5/output.txt", 4, "1_2_4_8_160__3_5/true_output.txt");
  // TestGatherHWC(false, {1, 12, 2, 8, 208}, {3, 2},
  // "1_12_2_8_208__3_2/input.txt", "1_12_2_8_208__3_2/indices.txt",
  // "1_12_2_8_208__3_2/output.txt", 4, "1_12_2_8_208__3_2/true_output.txt");

  // std::cout << "\nrvv:" << '\n';
  // TestGatherHWC(true, {64, 1, 128}, {1, 128}, "64_1_128__1_128/input.txt",
  // "64_1_128__1_128/indices.txt", "64_1_128__1_128/output.txt", 0,
  // "64_1_128__1_128/true_output.txt"); TestGatherHWC(true, {64, 1, 256}, {1,
  // 256}, "64_1_256__1_256/input.txt", "64_1_256__1_256/indices.txt",
  // "64_1_256__1_256/output.txt", 0, "64_1_256__1_256/true_output.txt");
  // TestGatherHWC(true, {60, 4, 64}, {1}, "60_4_64_1/input.txt",
  // "60_4_64_1/indices.txt", "60_4_64_1/output.txt", 0,
  // "60_4_64_1/true_output.txt"); TestGatherHWC(true, {128, 64, 16}, {32},
  // "128_64_16__32/input.txt", "128_64_16__32/indices.txt",
  // "128_64_16__32/output.txt", 1, "128_64_16__32/true_output.txt");
  // TestGatherHWC(true, {32, 2, 208}, {1, 200}, "32_2_208__1_200/input.txt",
  // "32_2_208__1_200/indices.txt", "32_2_208__1_200/output.txt", 0,
  // "32_2_208__1_200/true_output.txt");

  // TestGatherHWC(true, {3, 8, 4, 16}, {1}, "3_8_4_16__1/input.txt",
  // "3_8_4_16__1/indices.txt", "3_8_4_16__1/output.txt", 0,
  // "3_8_4_16__1/true_output.txt"); TestGatherHWC(true, {1, 4, 4, 16}, {1},
  // "1_4_4_16__1/input.txt", "1_4_4_16__1/indices.txt",
  // "1_4_4_16__1/output.txt", 2, "1_4_4_16__1/true_output.txt");
  // TestGatherHWC(true, {1, 4, 4, 16}, {7, 2}, "1_4_4_16__7_2/input.txt",
  // "1_4_4_16__7_2/indices.txt", "1_4_4_16__7_2/output.txt", 2,
  // "1_4_4_16__7_2/true_output.txt"); TestGatherHWC(true, {1, 8, 8, 160}, {3,
  // 2}, "1_8_8_160__3_2/input.txt", "1_8_8_160__3_2/indices.txt",
  // "1_8_8_160__3_2/output.txt", 2, "1_8_8_160__3_2/true_output.txt");

  // TestGatherHWC(true, {1, 16, 6, 4, 96}, {1}, "1_16_6_4_96__1/input.txt",
  // "1_16_6_4_96__1/indices.txt", "1_16_6_4_96__1/output.txt", 3,
  // "1_16_6_4_96__1/true_output.txt"); TestGatherHWC(true, {4, 1, 8, 2, 160},
  // {1}, "4_1_8_2_160__1/input.txt", "4_1_8_2_160__1/indices.txt",
  // "4_1_8_2_160__1/output.txt", 4, "4_1_8_2_160__1/true_output.txt");
  // TestGatherHWC(true, {1, 2, 4, 8, 160}, {3, 5},
  // "1_2_4_8_160__3_5/input.txt", "1_2_4_8_160__3_5/indices.txt",
  // "1_2_4_8_160__3_5/output.txt", 4, "1_2_4_8_160__3_5/true_output.txt");
  // TestGatherHWC(true, {1, 12, 2, 8, 208}, {3, 2},
  // "1_12_2_8_208__3_2/input.txt", "1_12_2_8_208__3_2/indices.txt",
  // "1_12_2_8_208__3_2/output.txt", 4, "1_12_2_8_208__3_2/true_output.txt");

  // std::cout << "\nchw_gather_memcpy:\n";
  // TestGatherCHW(false, {60, 60, 4}, {1}, "chw/60_60_4__1/input.txt",
  // "chw/60_60_4__1/indices.txt", "chw/60_60_4__1/output.txt", 0,
  // "chw/60_60_4__1/true_output.txt"); TestGatherCHW(false, {1, 128, 64}, {32},
  // "chw/1_128_64__32/input.txt", "chw/1_128_64__32/indices.txt",
  // "chw/1_128_64__32/output.txt", 1, "chw/1_128_64__32/true_output.txt");
  // TestGatherCHW(false, {200, 32, 2}, {1, 200},
  // "chw/200_32_2__1_200/input.txt", "chw/200_32_2__1_200/indices.txt",
  // "chw/200_32_2__1_200/output.txt", 0,
  // "chw/200_32_2__1_200/true_output.txt");

  // std::cout << "\nchw_gather_rvv:\n";
  // TestGatherCHW(true, {60, 60, 4}, {1}, "chw/60_60_4__1/input.txt",
  // "chw/60_60_4__1/indices.txt", "chw/60_60_4__1/output.txt", 0,
  // "chw/60_60_4__1/true_output.txt"); TestGatherCHW(true, {1, 128, 64}, {32},
  // "chw/1_128_64__32/input.txt", "chw/1_128_64__32/indices.txt",
  // "chw/1_128_64__32/output.txt", 1, "chw/1_128_64__32/true_output.txt");
  // TestGatherCHW(true, {200, 32, 2}, {1, 200},
  // "chw/200_32_2__1_200/input.txt", "chw/200_32_2__1_200/indices.txt",
  // "chw/200_32_2__1_200/output.txt", 0,
  // "chw/200_32_2__1_200/true_output.txt");

  // std::cout << "Test time:\n";
  // TestGatherTime(0);
  // TestGatherTime(1);
  // TestGatherTime(2);
  TestGatherCHW(true, {1, 128, 64}, {32}, "chw/1_128_64__32/input.txt",
                "chw/1_128_64__32/indices.txt", "chw/1_128_64__32/output.txt",
                1);
  return 0;
}