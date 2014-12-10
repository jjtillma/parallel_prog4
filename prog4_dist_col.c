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
#include <math.h>

void printMatrix(unsigned int code, unsigned int ROWS, double *arr);
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, double *P);
void makeMatrices(unsigned int size, double *U, double *L, double *P,  unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank, unsigned int comm_sz);
void swap(double *a, double *b);
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
	MPI_Scatter(U, block*size, MPI_DOUBLE, localU, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(L, block*size, MPI_DOUBLE, localL, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(P, block*size, MPI_DOUBLE, localP, block*size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//make the L and U matrices
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
recommended. L is initialized to the identity matrix of the appropriate
dimensions. U is initialized to a copy of the INPUT matrix so that the INPUT
matrix is preserved for output.
******************************************************************************/
void makeInput(unsigned int SIZE, double *INPUT, double *L, double *U, double *P)
{
	unsigned int 	i, j;
	double		temp;

	for(i = 0; i < SIZE; i++)
	{
		for(j = 0; j < SIZE; j++)
		{
			scanf("%lf", &temp);
			U[j*SIZE+i] = temp;
			INPUT[j*SIZE+i] = temp;
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
Handles the making of the U matrix. The matrix starts as a copy of the INPUT
matrix and does Guassian Row Elimination up to the point where it becomes a
Right Upper Matrix. Multipliers for rows are stored in the SCALARS array by the
scaleURow method for later use by the makeLMatrix() method.
******************************************************************************/
void makeMatrices(unsigned int size, double *U, double *L, double *P, unsigned int block, unsigned int start, unsigned int end, unsigned int my_rank, unsigned int comm_sz)
{
	unsigned int 	i, j, k;
	double		max;
	double 		maxRow;
	double 		*multipliers;

	//allocate memory for multipliers
	multipliers = malloc(sizeof(double) * size);

	for(i = 0; i < size-1; i++)
	{
		//find max to put in diagonal for row swap
		if (my_rank == i / block)
		{
			max = U[(i-start)*size];
			maxRow = -1;
			multipliers[0] = -1;
			for(j = i+1; j < size; j++)
			{
				if(abs(U[(i-start)*size+j]) > max)
				{
					max = abs(U[(i-start)*size+j]);
					maxRow = j;
				}
			}
		
			//swap if necessary
			if (maxRow != -1)
			{
				multipliers[0] = maxRow;
				for(j = 0; j < block; j++)
				{
					swap(&U[j*size+i], &U[j*size+(int)maxRow]);
					swap(&P[j*size+i], &P[j*size+(int)maxRow]);
					if (j+start < i)
						swap(&L[j*size+i], &L[j*size+(int)multipliers[0]]);
				}
			}

			//compute multipliers
			for (j = i+1; j < size; j++)
			{
				multipliers[j] = U[(i-start)*size+j] / U[(i-start)*size+i];
				L[(i-start)*size+j] = multipliers[j];
			}
		}
		//send multipliers and swap index
		MPI_Bcast(multipliers, size, MPI_DOUBLE, i/block, MPI_COMM_WORLD);	

		//swap rows if necessary
		if (my_rank != i/block && (int)multipliers[0] != -1)
		{
			for(j = 0; j < block; j++)
			{
				swap(&U[j*size+i], &U[j*size+(int)multipliers[0]]);
				swap(&P[j*size+i], &P[j*size+(int)multipliers[0]]);
				if (j+start < i)
					swap(&L[j*size+i], &L[j*size+(int)multipliers[0]]);
			}
		}	

		//subtract row
		for(j = 0; j < block; j++)
		{
			for(k = i+1; k < size; k++)
				U[j*size+k] = U[j*size+k] - multipliers[k] * U[j*size+i];
		}
	}
}

/******************************************************************************
Swap two rows of a matrix with pointer assignment.
******************************************************************************/
void swap(double *a, double *b)
{
	double temp = *a;
	*a = *b;
	*b = temp;
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
			printf("%5.2lf    ", arr[j*ROWS+i]);
		}
		printf("\n");
	}
	return;
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
				sum = sum + L[j*size+k] * U[i*size+j];
			sumContainer[i*size+k] = sum;
			sum = 0;
		}
	}

	for(k = 0; k < size; k++)
	{
		for (i = 0; i < size; i++)
		{
			for(j = 0; j < size; j++)
				sum = sum + P[k*size+j] * sumContainer[i*size+j];
			sumContainer2[i*size+k] = sum;
			sum = 0;
		}
	}

	printMatrix(5, size, sumContainer2);

	free(sumContainer);
	free(sumContainer2);
	return;	
}
