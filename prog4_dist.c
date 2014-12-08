/******************************************************************************
File: prog4_dist.c

Purpose: Perform LU Decomposition on a matrix input by the user.

Compiling: mpicc -g -Wall -std=c99 -lm -o prog4_dist prog4_dist.c 

Usage: mpiexec -n <number of processes> -hostfile <list of hosts> ./prog4_dist

Notes: IMPORTANT NOTE, Redirected i/o is very strongly recommended for using
this program and an example input file should be shipped with this file.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void printMatrix(unsigned int code, unsigned int ROWS, double *arr);
void subtractRow(double *original, double* toChange, double multiplier, unsigned int SIZE);
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, int *INDEXES, double *SCALARS);
void makeMatrices(unsigned int size, double *U, double *L, unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank);

int main(int argc, char * argv[])
{
	double 		*input;
	unsigned int 	size;
	double 		*L;
	double 		*U;
	int		*indexes;
	double		*scalars;
	int		comm_sz;
	int		my_rank;
	double 		t_start, t_length;
	unsigned int	start, end;
	unsigned int 	block;
	double 		*localU;
	double		*localL;	

	MPI_Init(NULL, NULL);

	//get comm size and rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	//get user input and set up the original states of the matrices
	if(my_rank == 0)
	{
		printf("Enter number of rows for the square matrix: ");
		scanf("%u", &size);
	
		if(size % comm_sz != 0)
		{
			printf("Number of processes must divide number of rows.\n");
			size = -1;
		}
		else
		{
			//matrices as contiguous memory
			input = malloc(sizeof(double) * size * size);
			L = malloc(sizeof(double) * size * size);
			U = malloc(sizeof(double) * size * size);
			indexes = malloc(sizeof(unsigned int) * size);
			scalars = malloc(sizeof(double)*(size-1)*(size)/2);
		
			makeInput(size, input, L, U, indexes, scalars);
			t_start = MPI_Wtime();
		}
	}

	//send size to all processes
	MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	if (size == -1)
	{
		MPI_Finalize();
		return 0;
	}
	//get start and end rows
	block = size / comm_sz;
	start = my_rank * size / comm_sz;
	end = (my_rank+1) * size / comm_sz - 1;

	//allocate memory for local rows
	localU = malloc(sizeof(double) * block * size);
	localL = malloc(sizeof(double) * block * size);

	//distribute matrix
	MPI_Scatter(input, block*size, MPI_DOUBLE, localU, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(L, block*size, MPI_DOUBLE, localL, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//make the L and U matrices
	makeMatrices(size, localU, localL, block, start, end, my_rank);
	
	//gather results
	MPI_Gather(localU, block*size, MPI_DOUBLE,  U, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Gather(localL, block*size, MPI_DOUBLE,  L, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	
	//print results
	if(my_rank == 0)
	{
		t_length = MPI_Wtime() - t_start;
		printf("The LU Decomposition took %lf seconds\n", t_length);
		printMatrix(3, size, input);
		printMatrix(1, size, L);
		printMatrix(2, size, U);
	}
	
	//free all the arrays
	if(my_rank == 0)
	{
		free(scalars);
		free(indexes);
		free(input);
		free(L);
		free(U);
	}
	free(localU);
	free(localL);

	MPI_Finalize();
	return 0;
}

/******************************************************************************
Gets the input of the matrix information from the user (redirect i/o) is
recommended. L is initialized to the identity matrix of the appropriate
dimensions. U is initialized to a copy of the INPUT matrix so that the INPUT
matrix is preserved for output.
******************************************************************************/
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, int *INDEXES, double *SCALARS)
{
	unsigned int i, j;

	for(i = 0; i < SIZE; i++)
	{
		for(j = 0; j < SIZE; j++)
		{
			scanf("%lf", &INPUT[i*SIZE+j]);
			U[i*SIZE+j] = INPUT[i*SIZE+j];
			if(i == j)
			{
				L[i*SIZE+j] = 1;
			}
			else
			{
				L[i*SIZE+j] = 0;
			}
		}
		if(i == 0 || i == 1)
		{
			INDEXES[i] = -1;
		}
		else
		{
			INDEXES[i] = INDEXES[i-1] + i - 1;
		}
	}
}

/******************************************************************************
Handles the making of the U matrix. The matrix starts as a copy of the INPUT
matrix and does Guassian Row Elimination up to the point where it becomes a
Right Upper Matrix. Multipliers for rows are stored in the SCALARS array by the
scaleURow method for later use by the makeLMatrix() method.
******************************************************************************/
void makeMatrices(unsigned int size, double *U, double *L, unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank)
{
	unsigned int i, j;
	int owner;
	double scalar;
	double *rowI;
	double *rowJ;

	//allocate rows
	rowI = malloc(size * sizeof(double));

	for(i = 0; i < size; i++)
	{
		//send necessary row to all processes
		owner = i / block;
		if(my_rank == owner)
		{
			for (j=0; j < size; j++)
				rowI[j] = U[(i-start)*size+j];
		}	
		MPI_Bcast(rowI, size, MPI_DOUBLE, owner, MPI_COMM_WORLD);

		for(j = 0; j < block; j++)
		{
			if(j+start > i)
			{
				rowJ = &U[j*size];
				if(rowJ[i] == 0 || rowI[i] == 0)
				{
					L[j*size+i] = 0;
				}
				else
				{
					scalar = rowJ[i] / rowI[i];
					L[j*size+i] = scalar;
					subtractRow(rowI, rowJ, scalar, size);
				}	
			}		
		}
	}
}

/******************************************************************************
Prints the matrix specified by the code passed into it. 1 prints L, 2 prints U,
3 prints INPUT.
******************************************************************************/
void printMatrix(unsigned int code, unsigned int ROWS, double *arr)
{
	unsigned int i, j;

	switch (code)
	{
		case 1:
		{
			printf("**********L Matrix**********\n");
			break;
		}
		case 2:
		{
			printf("**********U Matrix**********\n");
			break;
		}
		case 3:
		{
			printf("**********Input Matrix**********\n");
			break;
		}
		/*case 4:
		{
			double sum = 0;
			printf("**********Multiply Matrix**********\n");
			for(k = 0; k < ROWS; k++)
			{
				for(i = 0; i < ROWS; i++)
				{
					for(j = 0; j < ROWS; j++)
					{
						sum = sum + L[k][j] * U[j][i];
					}
					printf("%0.2lf    ", sum);
					sum = 0;
				}
				printf("\n");
			}
			break;
		}*/
	}
	for(i = 0; i < ROWS; i++)
	{
		for(j = 0; j < ROWS; j++)
		{
			printf("%lf    ", arr[i*ROWS+j]);
		}
		printf("\n");
	}
	return;
}

/******************************************************************************
This function subtracts one row from another while scaling the row that
corresponds to the "oroginal" indexer. It returns a new row rather than
assigning to the original row in an effort to shorten critical sections.
******************************************************************************/
void subtractRow(double* original, double* toChange, double multiplier, unsigned int SIZE)
{
	unsigned int i;

	for(i = 0; i < SIZE; i++)
	{
		toChange[i] = toChange[i] - original[i] * multiplier;
	}

	//return toReturn;
}
