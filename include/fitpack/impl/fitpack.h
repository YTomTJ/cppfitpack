#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <Eigen/Dense>

#if defined(DEBUG) || defined(_DEBUG)
#pragma comment(lib, "fitpackd.lib")
#else
#pragma comment(lib, "fitpack.lib")
#endif // defined(DEBUG) || defined(_DEBUG)

extern "C" {
void PARCUR(int *iopt, int *ipar, int *idim, int *m, double *u, int *mx, double *x, double *w,
    double *ub, double *ue, int *k, double *s, int *nest, int *n, double *t, int *nc, double *c,
    double *fp, double *wrk, int *lwrk, int *iwrk, int *ier);
void SPLEV(double *t, int *n, double *c, int *k, double *x, double *y, int *m, int *e, int *ier);
void SPLDER(double *t, int *n, double *c, int *k, int *nu, double *x, double *y, int *m, int *e,
    double *wrk, int *ier);
}

namespace fitpack {

#define CHECK_ERROR(ier)                                                                           \
    if (ier > 0) {                                                                                 \
        if (errmsg.find(ier) != errmsg.end()) {                                                    \
            throw std::runtime_error(errmsg.at(ier));                                              \
        } else if (pp.verbose > 0) {                                                               \
            std::cerr << "Unknow error: " << ier << std::endl;                                     \
        }                                                                                          \
    } else if (pp.verbose > 1) {                                                                   \
        std::cout << "Success: " << errmsg.at(ier) << std::endl;                                   \
    }

} // namespace fitpack
