/******************************************************************************
File: prog4_dist_row.c

Purpose: Perform LU Decomposition on a matrix input by the user, based on
	 row-wise partitioning.

Compiling: mpicc -g -Wall -std=c99 -lm -o prog4_dist_row prog4_dist_row.c 

Usage: mpiexec -n <number of processes> -hostfile <list of hosts> ./prog4_dist_row

Notes: IMPORTANT NOTE, Redirected i/o is very strongly recommended for using
this program and an example input file should be shipped with this file.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void printMatrix(unsigned int code, unsigned int ROWS, double *arr);
void subtractRow(double *original, double* toChange, double multiplier, unsigned int SIZE);
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, double *P);
void makeMatrices(unsigned int size, double *U, double *L, double *P,  unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank, unsigned int comm_sz);
void swap(double *a, double *b, int size);
void multiplyMatrices(double* P, double *L, double *U, int size);

int main(int argc, char * argv[])
{
	double 		*input;
	unsigned int 	size;
	double 		*L;
	double 		*U;
	double 		*P;
	int		comm_sz;
	int		my_rank;
	double 		t_start, t_length;
	unsigned int	start, end;
	unsigned int 	block;
	double 		*localU;
	double		*localL;	
	double 		*localP;

	MPI_Init(NULL, NULL);

	//get comm size and rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	//get user input and set up the original states of the matrices
	if(my_rank == 0)
	{
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
			P = malloc(sizeof(double) * size * size);
		
			makeInput(size, input, L, U, P);
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
	localP = malloc(sizeof(double) * block * size);

	//distribute matrix
	MPI_Scatter(input, block*size, MPI_DOUBLE, localU, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(L, block*size, MPI_DOUBLE, localL, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(P, block*size, MPI_DOUBLE, localP, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//make the L, U and P matrices
	makeMatrices(size, localU, localL, localP, block, start, end, my_rank, comm_sz);
	
	//gather results
	MPI_Gather(localU, block*size, MPI_DOUBLE, U, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Gather(localL, block*size, MPI_DOUBLE, L, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Gather(localP, block*size, MPI_DOUBLE, P, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

	//print results
	if(my_rank == 0)
	{
		t_length = MPI_Wtime() - t_start;
		printf("The LU Decomposition took %lf seconds\n", t_length);
		
		#ifdef DEBUG
		printMatrix(3, size, input);
		printMatrix(1, size, L);
		printMatrix(2, size, U);
		printMatrix(4, size, P);
		multiplyMatrices(P, L, U, size);
		#endif
	}
	
	//free all the arrays
	if(my_rank == 0)
	{
		free(input);
		free(L);
		free(U);
		free(P);
	}
	free(localU);
	free(localL);
	free(localP);

	MPI_Finalize();
	return 0;
}

/******************************************************************************
Gets the input of the matrix information from the user (redirect i/o) is
recommended. L and P are initialized to the identity matrix of the appropriate
dimensions. U is initialized to a copy of the INPUT matrix so that the INPUT
matrix is preserved for output.
******************************************************************************/
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, double *P)
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
				P[i*SIZE+j] = 1;
			}
			else
			{
				L[i*SIZE+j] = 0;
				P[i*SIZE+j] = 0;
			}
		}
	}
}

/******************************************************************************
Handles the making of the U, L, and P matrices. U starts as a copy of the INPUT
matrix and does Guassian Row Elimination up to the point where it becomes a
Right Upper Matrix. Multipliers for rows are stored in L to produce a Left
Lower Matrix. Anytime a row swap is performed based on partial pivoting,
the rows in P are swapped to produce a Permutation Matrix.

Each process owns a section of rows of the input matrix. When a row swap is 
required, the owners of those rows communicate the rows to be swapped. The
row to eliminate is sent to all processes, where it is subtracted from the 
other rows. This was very slow.
******************************************************************************/
void makeMatrices(unsigned int size, double *U, double *L, double *P, unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank, unsigned int comm_sz)
{
	unsigned int i, j;
	int owner;
	double scalar;
	double *rowI;
	double *rowJ;
	double localMax;
	int localMaxRow;
	double *potential;
	int *potentialRows;
	double globalMax;
	int globalRow;
	int globalOwner;

	//allocate rows
	rowI = malloc(size * sizeof(double));

	//allocate memory for finding global max
	if (my_rank == 0)
	{
		potential = malloc(comm_sz * sizeof(double));
		potentialRows = malloc(comm_sz * sizeof(int));
	}

	for(i = 0; i < size-1; i++)
	{
		//find max to put in diagonal for row swap
		//first, local max
		localMax = -1;
		localMaxRow = -1;
		for(j = 0; j < block; j++)
		{
			if(j+start >= i && abs(U[j*size+i]) > localMax)
			{
				localMax = abs(U[j*size+i]);
				localMaxRow = j + start;
			}
		}

		//send to process 0	
		MPI_Gather(&localMax, 1, MPI_DOUBLE, potential, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&localMaxRow, 1, MPI_INT, potentialRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//find row with max for diagonal
		if(my_rank == 0)
		{
			globalMax = -1;
			for(j = 0; j < comm_sz; j++)
			{
				if(potential[j] > globalMax)
				{
					globalMax = potential[j];
					globalRow = potentialRows[j];
				}
			}
		}	
		//send location of new max row to all processes
		MPI_Bcast(&globalRow, 1, MPI_INT, 0, MPI_COMM_WORLD);	

		//swap rows if necessary
		owner = i / block;
		globalOwner = globalRow / block;
		//swap between processes
		if(i != globalRow && owner != globalOwner)
		{
			if(my_rank == owner)
			{
				double *temp = malloc(sizeof(double)*size*3);
				swap(temp, &U[(i-start)*size], size);
				swap(&temp[size], &L[(i-start)*size], size);
				swap(&temp[2*size], &P[(i-start)*size], size);
				MPI_Sendrecv_replace(temp, size*3, MPI_DOUBLE, globalOwner, 0, globalOwner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
				swap(temp, &U[(i-start)*size], size);
				swap(&temp[size], &L[(i-start)*size], size);
				swap(&temp[2*size], &P[(i-start)*size], size);
				L[(i-start)*size+globalRow] = 0;
				L[(i-start)*size+i] = 1;
			}
			else if(my_rank == globalOwner)
			{
				double *temp = malloc(sizeof(double)*size*3);
				swap(temp, &U[(globalRow-start)*size], size);
				swap(&temp[size], &L[(globalRow-start)*size], size);
				swap(&temp[2*size], &P[(globalRow-start)*size], size);
				MPI_Sendrecv_replace(temp, size*3, MPI_DOUBLE, owner, 0, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
				swap(temp, &U[(globalRow-start)*size], size);
				swap(&temp[size], &L[(globalRow-start)*size], size);
				swap(&temp[2*size], &P[(globalRow-start)*size], size);
				L[(globalRow-start)*size+i] = 0;
				L[(globalRow-start)*size+globalRow] = 1;
			}		
		}
		//swap in same process
		else if(i != globalRow && my_rank == owner)
		{
			//swap rows in U
			swap(&U[(i-start)*size], &U[(globalRow-start)*size], size);
			//swap rows in P
			swap(&P[(i-start)*size], &P[(globalRow-start)*size], size);
			//swap below diagonal in L
			swap(&L[(i-start)*size], &L[(globalRow-start)*size], size);
			L[(i-start)*size+globalRow] = 0;
			L[(i-start)*size+i] = 1;
			L[(globalRow-start)*size+i] = 0;
			L[(globalRow-start)*size+globalRow] = 1;
		}

		//send necessary row to all processes
		if(my_rank == owner)
		{
			for (j=0; j < size; j++)
				rowI[j] = U[(i-start)*size+j];
		}	
		MPI_Bcast(rowI, size, MPI_DOUBLE, owner, MPI_COMM_WORLD);

		//perform row elimination, put multipliers in L
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
Swap two rows of a matrix, making sure that individual elements switch places
to maintain contiguous memory.
******************************************************************************/
void swap(double *a, double *b, int size)
{
	int i;
	double temp;

	for(i = 0; i < size; i++)
	{
		temp = a[i];
		a[i] = b[i];
		b[i] = temp;
	}
}

/******************************************************************************
Prints the matrix specified by the code passed into it. 1 prints L, 2 prints U,
3 prints INPUT, 4 prints P, and 5 prints the multiplied result of P'LU.
******************************************************************************/
void printMatrix(unsigned int code, unsigned int ROWS, double *arr)
{
	unsigned int i, j;

	switch (code)
	{
		case 1:
		{
			printf("**********L Matrix**************\n");
			break;
		}
		case 2:
		{
			printf("**********U Matrix**************\n");
			break;
		}
		case 3:
		{
			printf("**********Input Matrix**********\n");
			break;
		}
		case 4:
		{
			printf("**********P Matrix**************\n");
			break;
		}
		case 5:
		{
			printf("**********Multiply Matrix**********\n");
			break;
		}
	}
	for(i = 0; i < ROWS; i++)
	{
		for(j = 0; j < ROWS; j++)
		{
			printf("%5.2lf    ", arr[i*ROWS+j]);
		}
		printf("\n");
	}
	return;
}

/******************************************************************************
This function subtracts one row from another while scaling the row that
corresponds to the "original" indexer. 
******************************************************************************/
void subtractRow(double* original, double* toChange, double multiplier, unsigned int SIZE)
{
	unsigned int i;

	for(i = 0; i < SIZE; i++)
	{
		toChange[i] = toChange[i] - original[i] * multiplier;
	}
}

/******************************************************************************
Multiply P'LU and display results to compare against original matrix A.
******************************************************************************/
void multiplyMatrices(double* P, double *L, double *U, int size)
{
	int i, j, k;
	double sum = 0;
	double *sumContainer = malloc(sizeof(double) * size * size);
	double *sumContainer2 = malloc(sizeof(double) * size * size);

	for(k = 0; k < size; k++)
	{
		for (i = 0; i < size; i++)
		{
			for(j = 0; j < size; j++)
				sum = sum + L[k*size+j] * U[j*size+i];
			sumContainer[k*size+i] = sum;
			sum = 0;
		}
	}

	for(k = 0; k < size; k++)
	{
		for (i = 0; i < size; i++)
		{
			for(j = 0; j < size; j++)
				sum = sum + P[j*size+k] * sumContainer[j*size+i];
			sumContainer2[k*size+i] = sum;
			sum = 0;
		}
	}

	printMatrix(5, size, sumContainer2);

	free(sumContainer);
	free(sumContainer2);
	return;	
}
