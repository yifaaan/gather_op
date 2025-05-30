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

float* readFile(const char* path, int len);
int* readFileINT(const char* path, int len);
void outputFile_line(const char* path, const vector<float>& output);
void outputFile_line_int(const char* path, const vector<int>& output);
void outputFile2d_line(const char* path, const vector<vector<int>>& output);
void TestGatherCHW(bool is_rvv, std::vector<int> in_shape,
                   std::vector<int> indices_shape, const char* input_path,
                   const char* indices_path, const char* output_path, int axis);
#endif