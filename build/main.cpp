#include "fitpack/fitpack.hpp"

int main()
{
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(100, 3);
    std::cout << x << std::endl;

    fitpack::parcur_param pp;
    try {
        auto res = fitpack::parcur(x, pp);
        std::cout << res.t << std::endl;
        std::cout << res.c << std::endl;
        std::cout << res.fp << std::endl;

        fitpack::splev_param sp;
        sp.c = res.c;
        sp.t = res.t;
        sp.k = res.k;

        auto ss = fitpack::splev(res.u, sp);
        std::cout << ss.y << std::endl;

    } catch (std::exception e) {
        std::cout << e.what();
    }

    return 0;
}
