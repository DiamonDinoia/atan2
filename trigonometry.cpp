//============================================================================
// Name        : trigonometry.cpp
// Author      : Marco Barbone
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include <iostream>
#include <math.h>
#include <chrono>
#include <bitset>
using namespace std;
using namespace std::chrono;

// returns random numbers in range [0,1]
float floatRand() {
  return float(rand()) / (float(RAND_MAX));
}
// Nvidia implementation of Atan2 for Cuda
// source: https://developer.download.nvidia.com/cg/atan2.html
float nvidia_atan2(float y, float x){
  float t0, t1, t2, t3, t4;
  t3 = fabs(x);
  t1 = fabs(y);
  t0 = max(t3, t1);
  t1 = min(t3, t1);
  t3 = float(1) / t0;
  t3 = t1 * t3;
  t4 = t3 * t3;
  t0 =         - float(0.013480470);
  t0 = t0 * t4 + float(0.057477314);
  t0 = t0 * t4 - float(0.121239071);
  t0 = t0 * t4 + float(0.195635925);
  t0 = t0 * t4 - float(0.332994597);
  t0 = t0 * t4 + float(0.999995630);
  t3 = t0 * t3;
  t3 = (fabs(y) > fabs(x)) ? float(1.570796327) - t3 : t3;
  t3 = (x < 0) ?  float(3.141592654) - t3 : t3;
  t3 = (y < 0) ? -t3 : t3;
  return t3;
}

// atan2 implementation from root CERN
// source https://root.cern.ch/doc/v608/atan2_8h_source.html
float cern_fast_atan2f( float y, float x ) {
     // move in first octant
     float xx = std::fabs(x);
     float yy = std::fabs(y);
     float tmp (0.0f);
     if (yy>xx) {
         tmp = yy;
         yy=xx; xx=tmp;
         tmp =1.f;
     }
     // To avoid the fpe, we protect against /0.
     const float oneIfXXZero = (xx==0.f);
     float t=yy/(xx/*+oneIfXXZero*/);
     float z=t;
     if( t > 0.4142135623730950f ) // * tan pi/8
             z = (t-1.0f)/(t+1.0f);
     //printf("%e %e %e %e\n",yy,xx,t,z);
     float z2 = z * z;
 
     float ret =(((( 8.05374449538e-2f * z2
                     - 1.38776856032E-1f) * z2
                     + 1.99777106478E-1f) * z2
                     - 3.33329491539E-1f) * z2 * z
                     + z );
 
     // Here we put the result to 0 if xx was 0, if not nothing happens!
     ret*= (1.f - oneIfXXZero);
 
     // move back in place
     if (y==0.f) ret=0.f;
     if( t > 0.4142135623730950f ) ret += 0.78539816f;
     if (tmp!=0) ret = 1.57079637f - ret;
     if (x<0.f) ret = 3.14159274f - ret;
     if (y<0.f) ret = -ret;
 
     return ret;
 
 }
// implementation proposed by njuffa user on stackoverflow
// source https://stackoverflow.com/questions/46210708/atan2-approximation-with-11bits-in-mantissa-on-x86with-sse2-and-armwith-vfpv4
float fast_atan2f (float y, float x){
    float a, r, s, t, c, q, ax, ay, mx, mn;
    ax = fabsf (x);
    ay = fabsf (y);
    mx = fmaxf (ay, ax);
    mn = fminf (ay, ax);
    a = mn / mx;
    /* Minimax polynomial approximation to atan(a) on [0,1] */
    s = a * a;
    c = s * a;
    q = s * s;
    r =  0.024840285f * q + 0.18681418f;
    t = -0.094097948f * q - 0.33213072f;
    r = r * s + t;
    r = r * c + a;
    /* Map to full circle */
    if (ay > ax) r = 1.57079637f - r;
    if (x <   0) r = 3.14159274f - r;
    if (y <   0) r = -r;
    return r;
}



inline float approxAtan(float z){
	const float n1 = 0.97239411f;
	const float n2 = -0.19194795f;
	return 0.141499f * (z*z*z*z) - 0.343315f * (z*z*z) - 0.016224f * (z*z) + 1.003839f * z - 0.00015f ;
	// return (n1 + n2 * z * z) * z;
}

// Implenentation that exploits the Remez Method
// source: https://www.dsprelated.com/showarticle/1052.php
float approxAtan2(const float y, const float x){
	float ax = fabs(x);
	float ay = fabs(y);
	int invert = 0 - (ay > ax);
	float z = invert ? ax/ay : ay/ax;
	float th = approxAtan(z);
  // bitwise operations to avoid branches
  int *th_bits = (int*)(&th);
// th = invert ? 1.57079637f - th : th;

  float tmp = 1.57079637f - th;
  int *tmp_bits = (int*)(&tmp);
  *th_bits = (invert & *tmp_bits) | (~invert & *th_bits);
  
  th = x < 0 ? 3.14159274f - th : th;

  // tmp = 3.14159274f - th;
  // invert = 0 - (x < 0);
  // *th_bits = (invert & *tmp_bits) | (~invert & *th_bits);
  *th_bits |= (*th_bits & ~0) | (*th_bits & 1<<31);
  // th = invert ? 1.57079637f - th : th;
  // th = x < 0 ? 3.14159274f - th : th;
	// th = y >= 0 ? th : -th;
	return th;
}

const unsigned int N = 4096 * 4096;


void check_error(float* reference, float* result){
  auto error = 0.f;
  for (unsigned int i = 0; i < N; i++){
    error += fabs(result[i]-reference[i]);
  }
  cout << "AVG error " << error/N << endl;
}
//floatRandRange()


int main() {
	float* x = new float[N];
	float* y = new float[N];
	float* reference = new float[N];
  float* result = new float[N];
	float error = 0.f;
	const auto seed = 42;
	cout << "Test configuration" << endl;
	cout << "Number of iterations " << N << " seed " << seed << endl;
	srand(seed);

	// generating input data
	for (unsigned int i = 0; i < N; ++i) {
		x[i] = floatRand();
		y[i] = floatRand();
	}
	
  high_resolution_clock::time_point start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; ++i) {
		reference[i] = atan2(y[i], x[i]);
	}
	high_resolution_clock::time_point end = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(end - start);
    std::cout << "reference std implementation time: " << time_span.count() << std::endl;


  start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; ++i) {
		result[i] = nvidia_atan2(y[i], x[i]);
	}
	end = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(end - start);
  std::cout << "nvidia implementation time: " << time_span.count() << std::endl;
	check_error(reference, result);


  start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; ++i) {
		result[i] = cern_fast_atan2f(y[i], x[i]);
	}
	end = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(end - start);
  std::cout << "cern implementation time: " << time_span.count() << std::endl;
	check_error(reference, result);


  start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; ++i) {
		result[i] = fast_atan2f(y[i], x[i]);
	}
	end = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(end - start);
  std::cout << "fast implementation time: " << time_span.count() << std::endl;
	check_error(reference, result);


  start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; ++i) {
		result[i] = approxAtan2(y[i], x[i]);
	}
	end = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(end - start);
  std::cout << "approx implementation time: " << time_span.count() << std::endl;
	check_error(reference, result);
  
  return 0;
}
