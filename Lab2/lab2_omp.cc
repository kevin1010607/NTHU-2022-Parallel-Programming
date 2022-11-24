#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define ull unsigned long long 
#define min(a,b) (((a)<(b))?(a):(b))

int main(int argc, char** argv) {
	ull r = atoll(argv[1]), R = r*r;
	ull k = atoll(argv[2]);
	ull res = 0;

	int size = 4;
	ull unit = r/size, remain = r%size;

	#pragma omp parallel for num_threads(4) schedule(static, 1) reduction(+:res)
	for(int i = 0; i < 4; i++){
		ull start = unit*i+min(remain, i);
		ull len = unit+(i < remain);

		ull now = ceil(sqrtl(R-start*start));

		for(ull x = start; x < start+len; x++){
			ull t = R-x*x;
			while(t <= (now-1)*(now-1)) now -= 4;
			while(t > now*now) now++;
			res += now;
		}
		res %= k;
	}
	
	printf("%llu\n", (4 * res) % k);

	return 0;
}
