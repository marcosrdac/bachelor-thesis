# reverse_mode.py
#
# Author: Marcos Conceicao (marcosrdac@gmail.com)
# Date: 2021-07-28

from dataclasses import dataclass
import math


@dataclass(frozen=True, eq=False)
class Var:
    '''
    Structure for naive implementation  of reverse mode autodiff.

    Attributes:
        primal  node's primal value
        parents_and_op_adjoints  tuple of tuples of a node's parents and
        associated adjoints for reverse chain rule application. The recurrent
        tuple structure is able to keep track of the paths drawn in the graph.
    '''
    primal: float
    parents_and_op_adjoints: tuple = ()

    def __add__(a, b):
        '''Addition operation.'''
        return Var(
            a.primal + b.primal,
            ((a, 1), (b, 1)),
        )

    def __mul__(a, b):
        '''Product operation.'''
        return Var(
            a.primal * b.primal,
            ((a, b.primal), (b, a.primal)),
        )

    def __neg__(a):
        '''Negation operation.'''
        return Var(
            -a.primal,
            ((a, -1), ),
        )

    def __inv__(a):
        '''Inversion operation.'''
        return Var(
            1 / a.primal,
            ((a, -1 / a.primal**2), ),
        )

    def __sub__(self, other):
        '''Subtraction operation.'''
        return self + other.__neg__()

    def __truediv__(self, other):
        '''Division operation.'''
        return self * other.__inv__()

    def __pow__(self, other):
        '''Power operation.'''
        primal = self.primal**other.primal
        return Var(primal, (
            (self, other.primal * self.primal**(other.primal - 1)),
            (other, self.primal**other.primal),
        ))

    def partials(self, last_adjoint=1):
        '''Function to perform second step of reverse autodiff and actually
        calculate derivatives to base inputs.'''
        partials = {}

        def get_adjoints(var, adjoint):
            for child, derivative in var.parents_and_op_adjoints:
                accumulated = adjoint * derivative
                partials[child] = partials.get(child, 0) + accumulated

                # graph propagation
                get_adjoints(child, accumulated)

        get_adjoints(self, last_adjoint)
        return partials


def sin(a):
    '''Sine operation.'''
    return Var(
        math.sin(a.primal),
        ((a, math.cos(a.primal)), ),
    )


def cos(a):
    '''Cosine operation.'''
    return Var(
        math.cos(a.primal),
        ((a, -math.sin(a.primal)), ),
    )


def exp(a):
    '''Exponentiation operation.'''
    result = math.exp(a.primal)
    return Var(
        result,
        ((a, result), ),
    )


def ln(a):
    '''Natural logarithm operation.'''
    return Var(
        math.log(a.primal),
        ((a, 1 / a.primal), ),
    )


if __name__ == '__main__':

    def f(x):
        v_1 = x[0]
        v_2 = x[1]
        v_3 = v_1 * v_2
        v_4 = v_2 + v_3
        v_5 = sin(v_2)
        f = [v_4, v_5]
        return f

    x = [Var(2), Var(2)]

    grads = tuple(fi.partials() for fi in f(x))

    print('-- Gradient of first output to each input')
    print(f'{grads[0].get(x[0]) = }')
    print(f'{grads[0].get(x[1]) = }')
    print()
    print('-- Gradient of second output to each input')
    print(f'{grads[1].get(x[0]) = }')  # None, as f_2 is not linked to x_0
    print(f'{grads[1].get(x[1]) = }')

    print('== Transpose jacobian product')
    print('-- First element')

    r = (3, .2)
    print(f'{r = }')
    outs = f(x)
    edited_outs = [
        out.partials(last_adjoint=r[i]) for i, out in enumerate(outs)
    ]
    print('--- First output')
    print(f'{edited_outs[0].get(x[0]) = }')
    print(f'{edited_outs[0].get(x[1]) = }')
    print('--- Second output')
    print(f'{edited_outs[1].get(x[0]) = }')
    print(f'{edited_outs[1].get(x[1]) = }')
