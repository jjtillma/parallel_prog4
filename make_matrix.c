#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main()
{
	double i;
	double mill = 1000;

	srand(time(NULL));
	printf("1000\n");
	printf("1000\n");

	double stop = mill*mill;

	for(i = 1; i <= stop; i++)
	{
		printf("%d\n", rand() % 1000000);
	}

	return 0;
}
