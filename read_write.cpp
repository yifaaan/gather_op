#include "op.h"

float* readFile(const char* path, int len) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    printf("无法打开文件: %s\n", path);
    return nullptr;
  }

  float* dataBuf = (float*)malloc(len * sizeof(float));
  if (!dataBuf) {
    fclose(fp);  // 内存分配失败时关闭文件
    return nullptr;
  }

  for (int i = 0; i < len; i++) {
    if (fscanf(fp, "%f", &dataBuf[i]) != 1) {
      free(dataBuf);
      fclose(fp);  // 读取失败时关闭文件
      return nullptr;
    }
  }

  fclose(fp);  // 正确关闭文件
  return dataBuf;
}

int* readFileINT(const char* path, int len) {
  FILE* fp = fopen(path, "r");
  if (fp == NULL)
    printf("cannot open file\n");
  else {
    int* dataBuf = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
      fscanf(fp, "%d", dataBuf + i);
    }
    return dataBuf;
  }
}

void outputFile_line(const char* path, const vector<float>& output) {
  FILE* fp = fopen(path, "w+");
  if (fp == NULL) {
    printf("cannot open file for writing\n");
    return;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    fprintf(fp, "%.6f\n", output[i]);
  }

  fclose(fp);
  // printf("Data written to %s\n", path);
}

void outputFile_line_int(const char* path, const vector<int>& output) {
  FILE* fp = fopen(path, "w+");
  if (fp == NULL) {
    printf("无法打开文件%s进行写入\n", path);  // 提示具体无法打开的文件路径
    return;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    fprintf(fp, "%d\n", output[i]);  // 按照整型格式进行输出
  }

  fclose(fp);
}

void outputFile2d_line(const char* path,
                       const vector<vector<int>>& output) {  // smh
  FILE* fp = fopen(path, "w");
  if (fp == NULL) {
    printf("cannot open file for writing\n");
    return;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    for (size_t j = 0; j < output[i].size(); ++j) {
      fprintf(fp, "%d\n", output[i][j]);
    }
  }

  fclose(fp);
  printf("Data written to %s\n", path);
}