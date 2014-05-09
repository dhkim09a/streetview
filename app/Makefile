
.PHONY: all clean

include config.mk

CCBIN = gcc
CC = nvcc -ccbin $(CCBIN)

CFLAGS = -Wall -I../openSURF/src -I../dbmaker -g \
		 $(shell pkg-config --cflags opencv)
LDFLAGS = $(shell pkg-config --libs opencv) \
		  ../openSURF/libsurf.so

CFLAGS += $(addprefix -D,$(CONFIG))

CPU_PROG = cpu.elf
GPU_PROG = gpu.elf

COMMON_OBJ = main.o
CPU_OBJ = cpu_search.o
GPU_OBJ = gpu_search.o

all: $(GPU_PROG)

$(CPU_PROG): $(COMMON_OBJ) $(CPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

$(GPU_PROG): $(COMMON_OBJ) $(GPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

%.o: %.cu
	$(CC) $(addprefix -Xcompiler ,$(CFLAGS)) -c $<

clean:
	rm -f *.o
	rm -f $(CPU_PROG) $(GPU_PROG)
