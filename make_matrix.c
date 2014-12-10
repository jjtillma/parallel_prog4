#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[])
{
	double i;
	unsigned int size;

	if(argc != 2)
	{
		printf("Use a single command argument that specifies one dimension of the matrix.\n");
		return 1;
	}


	size = atoi(argv[1]);
	srand(time(NULL));
	printf("%d\n", size);

	double stop = size*size;

	for(i = 1; i <= stop; i++)
	{
		printf("%d\n", rand() % 20 + 1);
	}

	return 0;
}
