#ifndef DENSETT_H
#define DENSETT_H

// NOTE: Tblis should be always included before core.h
#include <tblis/tblis.h>
#include "core.h"

std::vector<tblis::tensor<double>> TT_SVD_dense(tblis::tensor<double> tensor, int r_max, double eps);
std::vector<tblis::tensor<double>> TT_IDQR_dense_nocutoff(tblis::tensor<double> tensor, int r_max);
std::vector<tblis::tensor<double>> TT_IDPRRLDU_dense(tblis::tensor<double> tensor, int r_max, double eps);

#endif