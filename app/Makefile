
.PHONY: all clean

include config.mk

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

SHARED_OBJ = main.o
CPU_OBJ = cpu_search.o
GPU_OBJ = gpu_search.o
HYB_OBJ = hyb_search.o $(CPU_OBJ) $(GPU_OBJ)


all: $(HYB_PROG) $(CPU_PROG) $(GPU_PROG)

$(CPU_PROG): $(addprefix cpu_,$(SHARED_OBJ)) $(CPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

$(GPU_PROG): $(addprefix gpu_,$(SHARED_OBJ)) $(GPU_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

$(HYB_PROG): $(addprefix hyb_,$(SHARED_OBJ)) $(HYB_OBJ)
	$(CC) -o $@ $^ $(addprefix -Xlinker ,$(LDFLAGS))

%.o: %.cu
	$(CC) -O2 --use_fast_math $(addprefix -Xcompiler ,$(CFLAGS)) -c $<

%.o: %.cpp
	$(CCBIN) -mavx $(CFLAGS) -c $<

$(addprefix cpu_,$(SHARED_OBJ)): $(addsuffix .cpp, $(basename $(SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_CPU) $(CFLAGS) -c $< -o $@

$(addprefix gpu_,$(SHARED_OBJ)): $(addsuffix .cpp, $(basename $(SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_GPU) $(CFLAGS) -c $< -o $@

$(addprefix hyb_,$(SHARED_OBJ)): $(addsuffix .cpp, $(basename $(SHARED_OBJ)))
	$(CCBIN) -mavx $(CFLAG_HYB) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(CPU_PROG) $(GPU_PROG)

