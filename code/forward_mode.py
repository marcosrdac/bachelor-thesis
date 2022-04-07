# forward_mode.py
#
# Author: Marcos Conceicao (marcosrdac@gmail.com)
# Date: 2021-07-26

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Var:
    '''
    Structure for naive implementation  of forward mode autodiff.

    Attributes:
        primal  node's primal value
        tangent node's tangent value
    '''
    primal: float
    tangent: float = 0

    def __repr__(self):
        '''How to represent this structure.'''
        class_name = self.__class__.__name__
        primal, tangent = self.primal, self.tangent
        return f"{class_name}({primal=:.4g}, {tangent=:.4g})"

    def __add__(a, b):
        '''Addition operation.'''
        return Var(
            a.primal + b.primal,
            a.tangent + b.tangent,
        )

    def __sub__(a, b):
        '''Subtraction operation.'''
        return Var(
            a.primal - b.primal,
            a.tangent - b.tangent,
        )

    def __mul__(a, b):
        '''Product operation.'''
        return Var(
            a.primal * b.primal,
            a.tangent * b.primal + a.primal * b.tangent,
        )

    def __truediv__(a, b):
        '''Division operation.'''
        return Var(a.primal / b.primal,
                   (a.tangent * b.primal - a.primal * b.tangent) / b.primal**2)

    def __pow__(a, b):
        '''Power operation.'''
        return Var(
            a.primal**b.primal,
            b.primal * a.primal**(b.primal - 1) * a.tangent +
            a.primal**b.primal * b.tangent,
        )

    def __abs__(a):
        '''Absolute value.'''
        return Var(
            abs(a.primal),
            a.tangent * math.copysign(1, a.primal),
        )


def sin(var):
    '''Sine operation.'''
    return Var(math.sin(var.primal), var.tangent * math.cos(var.primal))


def cos(var):
    '''Cosine operation.'''
    return Var(math.cos(var.primal), -var.tangent * math.sin(var.primal))


def exp(var):
    '''Exponentiation operation.'''
    return Var(math.exp(var.primal), var.tangent * math.exp(var.primal))


def ln(var):
    '''Natural logarithm operation.'''
    return Var(math.log(var.primal), var.tangent * 1 / var.primal)


if __name__ == '__main__':
    print('== Definitions')
    # constants
    ONE = Var(1, 0)  # init deriv = d(1)/da = 0
    # variables
    a = Var(2, 1)  # init deriv = da/da = 1
    b = Var(-1, 0)  # init deriv = db/da = 0
    print('-- Constants (tangent = dC/da = 0)')
    print(f'{ONE = }')
    print('-- Variables (tangent = dx/da)')
    print(f'{a = }')
    print(f'{b = }')
    print()

    print(
        '== Basic operation overloading')
    print('(result tangents are derivatives to a)')
    print(f'{a + b = }')
    print(f'{a - b = }')
    print(f'{a * b = }')
    print(f'{a / b = }')
    print(f'{a ** b = }')
    print(f'{abs(a) = }')
    print()

    print('== Some transcendental operations')
    print(f'{sin(a) = }')
    print(f'{cos(a) = }')
    print(f'{exp(a) = }')
    print(f'{ln(a) = }')
    print()

    print('== Composite operations')

    def f(x, y):
        return ONE / exp(cos(sin(x) + y))

    print('-- f(x,y) := 1/(exp(cos(sin(a) + b)))')
    print(f'{f(a, b) = }')
    print()

    print('== Partial derivatives')

    print('-- f(x) := [x[1] + x[0]*x[1], sin(x[1])]')

    def f(x):
        v_1 = x[0]
        v_2 = x[1]
        v_3 = v_1 * v_2
        v_4 = v_2 + v_3
        v_5 = sin(v_2)
        f = [v_4, v_5]
        return f

    print('-- Partial derivative relative to x_0 at x=[2,2]')
    x = [Var(2, 1), Var(2, 0)]
    print(f'{f(x) = }')
    print('--> Tangents:', [x_i.tangent for x_i in f(x)])
    print(4 * ' ' + 19 * '*')
    print()

    print('-- Jacobian vector product (J_f@r)')
    print(f'{x = }')
    r = Var(.1, 0), Var(-1, 0), Var(5, 0)
    print(f'{r = }')

    # make x's tangent = r
    x_with_r_as_tangent = tuple(
        Var(xi.primal, ri.primal) for xi, ri in zip(x, r))

    # run f(x)
    print(f'{f(x_with_r_as_tangent) = }')
    print('  ... pairs are (f(x)[i], J_f@r)[i])')
