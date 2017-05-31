#ifndef MATRIX_H
#define MATRIX_H

class IMatrix {

public:
    IMatrix(int rows, int cols);
    IMatrix(IMatrix& mat);

    // Dot product
    virtual IMatrix dot(IMatrix& mat) = 0;

    // elementwise multiplication
    virtual IMatrix mul(IMatrix& mat) = 0;

    // elementwise addition
    virtual IMatrix add(IMatrix& mat) = 0;

    // elementwise subtraction
    virtual IMatrix sub(IMatrix& mat) = 0;

    // elementwise division
    virtual IMatrix div(IMatrix& mat) = 0;

    int rows() {
        return m_rows;
    }

    int cols() {
        return m_cols;
    }

protected:
    int m_rows;
    int m_cols;
};

#endif
