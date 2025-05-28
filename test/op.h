#ifndef OP_H
#define OP_H

#include <sys/time.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

float *readFile(const char *path, int len);
int *readFileINT(const char *path, int len);
void outputFile_line(const char *path, const vector<float> &output);
void outputFile_line_int(const char *path, const vector<int> &output);
void outputFile2d_line(const char *path, const vector<vector<int>> &output);
/// 返回{convert_to_chw_time, calculate_time, convert_to_hwc_time, all_time}
std::vector<float> TestGatherCHW(bool is_rvv, std::vector<int> in_shape,
                                 std::vector<int> indices_shape,
                                 const char *input_path,
                                 const char *indices_path,
                                 const char *output_path, int axis,
                                 int align_channels);

float TestGatherHWC(bool is_rvv, std::vector<int> in_shape,
                    std::vector<int> indices_shape, const char *input_path,
                    const char *indices_path, const char *output_path, int axis,
                    int align_channels);
#endif