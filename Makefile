CC = gcc
LIBS =
FLAGS = -g -Wall
EXECS = make_matrix prog4_shared

all: $(EXECS)

prog4_shared: prog4_shared.c
	$(CC) $(FLAGS) -fopenmp -o $@ $? $(LIBS)

make_matrix: make_matrix.c
	$(CC) $(FLAGS) -o $@ $? $(LIBS) -lm

clean:
	$(RM) $(EXECS)

