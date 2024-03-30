CC := gcc
CFLAGS_COMMON := -Wall -Wextra -pedantic
CFLAGS := $(CFLAGS_COMMON) -O3
CFLAGS_DB := $(CFLAGS_COMMON) -ggdb

OUT := ntt.exe

default:
	$(CC) ntt.c -o $(OUT) $(CFLAGS)
	./$(OUT)

db:
	$(CC) ntt.c -o $(OUT) $(CFLAGS_DB)
	gdb $(OUT)