#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#define ull unsigned long long 
#define min(a,b) (((a)<(b))?(a):(b))

ull r, R, k, size, pixels[8];

void* solve(void *attr){
	int id = *(int*)attr;

	ull unit = r/size, remain = r%size;
    ull start = unit*id+min(remain, id);
    ull len = unit+(id < remain);

	ull now = ceil(sqrtl(R-start*start));

	for(ull x = start; x < start+len; x++){
		ull t = R-x*x;
		while(t <= (now-1)*(now-1)) now -= 4;
		while(t > now*now) now++;
		pixels[id] += now;
	}

	pixels[id] %= k;

	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	r = atoll(argv[1]), R = r*r;
	k = atoll(argv[2]);
	ull res = 0;

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	size = CPU_COUNT(&cpuset);

	pthread_t T[8];
	int id[8];

	for(int i = 0; i < size; i++){
		id[i] = i;
		pthread_create(&T[i], NULL, solve, (void*)&id[i]);
	}

	for(int i = 0; i < size; i++){
		pthread_join(T[i], NULL);
		res = (res+pixels[i])%k;
	}

	printf("%llu\n", (4 * res) % k);

	return 0;
}
