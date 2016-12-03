
#REG per a GPU thread
REG = 64

# Essential macros
CONFIG = \
	REG=$(REG) \
	MEM_LIMIT=2000000000 \
	CPU_CHUNK_SIZE=1000000 \
	GPU_CHUNK_SIZE=100000000 \
	NUMCPU=10 \
	NUMGPU=0 \
	MATCH_THRESH=1.0 \
	RESIZE=500 \
	PORT=8090 \
	BACKLOG=20 \
	NET_IMG_SIZE_LIMIT=10000000 \
	TAGLEN=32 \

#	CPU_CHUNK_SIZE=30000000 \

# Optional macros
CONFIG += \
#	AVX \
	PROFILE \
#	PROFILE_CUDA \
