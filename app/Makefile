
.PHONY: all clean

CONFIG_MK = config.mk

include $(CONFIG_MK)

CCBIN = gcc
CC = nvcc -ccbin $(CCBIN)

CFLAGS = -O2 -Wall -I../openSURF/src -I../dbmaker -g \
		 $(shell pkg-config --cflags opencv)
CFLAGS += -Wno-unused-function
LDFLAGS = $(shell pkg-config --libs opencv) \
		  ../openSURF/libsurf.so

CFLAGS += $(addprefix -D,$(CONFIG))

CFLAG_CPU = -Dsearch=searchCPU
CFLAG_GPU = -Dsearch=searchGPU
CFLAG_HYB = -Dsearch=searchHyb

CPU_PROG = cpu.elf
GPU_PROG = gpu.elf
HYB_PROG = hyb.elf

SHARED_OBJ = db_loader.o
SRC_SHARED_OBJ = main.o
CPU_OBJ = cpu_search.o
GPU_OBJ = gpu_search.o
HYB_OBJ = hyb_search.o $(CPU_OBJ) $(GPU_OBJ)

CPU_SHARED_OBJ = $(addprefix cpu_,$(SRC_SHARED_OBJ))
GPU_SHARED_OBJ = $(addprefix gpu_,$(SRC_SHARED_OBJ))
HYB_SHARED_OBJ = $(addprefix hyb_,$(SRC_SHARED_OBJ))

all: $(HYB_PROG) $(CPU_PROG) $(GPU_PROG)

$(CPU_PROG): $(SHARED_OBJ) $(CPU_SHARED_OBJ) $(CPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

$(GPU_PROG): $(SHARED_OBJ) $(GPU_SHARED_OBJ) $(GPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

$(HYB_PROG): $(SHARED_OBJ) $(HYB_SHARED_OBJ) $(HYB_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

%.o: %.cu
	$(CC) -O0 --use_fast_math $(addprefix -Xcompiler ,$(CFLAGS)) -c $<

%.o: %.cpp
	$(CCBIN) -mavx $(CFLAGS) -c $<

$(CPU_SHARED_OBJ): $(addsuffix .cpp, $(basename $(SRC_SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_CPU) $(CFLAGS) -c $< -o $@

$(GPU_SHARED_OBJ): $(addsuffix .cpp, $(basename $(SRC_SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_GPU) $(CFLAGS) -c $< -o $@

$(HYB_SHARED_OBJ): $(addsuffix .cpp, $(basename $(SRC_SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_HYB) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f *.elf

