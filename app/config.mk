
# Essential macros
CONFIG = \
	MEM_LIMIT=1500000000 \
	DB_LIMIT=10000000000 \
	NUMCPU=8 \
	CHUNK_SIZE=100000000 \
	MATCH_THRESH=1.0

# Optional macros
CONFIG += \
	AVX \
	PROFILE \
#	PROFILE_CUDA \
