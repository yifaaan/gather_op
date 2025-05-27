# 指令编译器和选项
CC = riscv64-unknown-linux-musl-g++
# 定义k230 SDK的根目录
k230sdk = ../../k230_sdk

CFLAGS= -w -g -mcmodel=medany -march=rv64gcv -mabi=lp64d -O2\
		-T $(k230sdk)/src/big/mpp/userapps/sample/linker_scripts/riscv64/link.lds \
		-L$(k230sdk)-main/src/big/rt-smart/userapps/sdk/rt-thread/lib/ \
 		-Wl,--whole-archive -lrtthread -Wl,--no-whole-archive -n --static
 		
 		
CLIBS=  -L$(k230sdk)/src/big/rt-smart/userapps/sdk/lib/risc-v/rv64/ \
		-L$(k230sdk)/src/big/rt-smart/userapps/sdk/rt-thread/lib/risc-v/rv64/ \
		-Wl,--start-group -lrtthread -Wl,--end-group 
		


# 目标文件
TARGET=./a.elf

SRC_DIRS = ../postprocess  ../include/

SRCS = $(wildcard $(addsuffix /*.c, $(SRC_DIRS)))	./main.cpp ./gather.cpp ./gather_chw.cpp ./convert.cpp ./read_write.cpp ./test_gather_chw.cpp

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(CLIBS) -o $(TARGET) -lm -g 

clean:
	rm -f $(TARGET) 
