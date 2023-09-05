CC = gcc
CFLAGS_COMMON = -Wall -Wextra -Wpedantic
CFLAGS = $(CFLAGS_COMMON) -O3
CFLAGS_DB = $(CFLAGS_COMMON) -ggdb

norm:
	$(CC) ntt.c -o ntt $(CFLAGS)
	./ntt

db:
	$(CC) ntt.c -o ntt $(CFLAGS_DB)
	gdb ntt