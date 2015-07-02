#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include <cstdlib>

/**
 * Keeps normalizations functions which might get quite complicated in a single
 *  place for all RBM testing units to access.
 * 
 * Normalize and denormalize are designed such that
 *  x ≈ normalize(x,y); denormalize(x,y); x
**/

/**
 * Expects data to be normally distributed across [-1, 1] and transforms it
 *  into a uniform distribution across [0, 1]
**/
void normalize(float* data,unsigned size);

/**
 * Expects data to be uniformly distributed across [0, 1] and transforms it
 *  into a normal distribution across [-1, 1]
**/
void denormalize(float* data,unsigned size);

#endif
