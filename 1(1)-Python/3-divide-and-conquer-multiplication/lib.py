from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    # 모듈로 연산에 사용할 상수 (main.py 에서 1000으로 설정)
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        # 생성자: 입력받은 2차원 리스트를 내부 행렬로 저장
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        # 모든 원소가 n인 shape 크기의 행렬 생성
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        # 주어진 크기의 0 행렬 생성
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        # 주어진 크기의 1 행렬 생성
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        # 단위행렬(I) 생성: 대각 원소만 1
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1  # __setitem__으로 MOD 적용 후 값 할당
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        # 행렬의 크기를 (행 개수, 열 개수)로 반환
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        # 깊은 복사를 통해 새로운 Matrix 객체 반환
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        # matrix[i][j] 읽기
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # matrix[i][j] 쓰기
        # MOD 연산을 적용하여 항상 0 <= 값 < MOD 유지
        i, j = key
        self.matrix[i][j] = value % Matrix.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        # 행렬 곱 연산: self @ matrix
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))
        # 행렬 곱셈: result[i][j] = sum(self[i][k] * matrix[k][j]) for k in range(m)
        # 3중 반복문으로 곱하고 더하기 (mod 연산은 __setitem__에서 처리)
        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:

        # 행렬 거듭제곱: 분할 정복(이진 지수법) 사용
        rows, cols = self.shape
        assert rows == cols, "정사각행렬에만 거듭제곱 적용 가능"

        # 초기 결과: 단위행렬
        result = Matrix.eye(rows)
        base = self.clone()  # 밑 행렬 복제
        exp = n

        # 지수를 이진법으로 처리
        while exp > 0:
            if exp & 1:
                # 현재 비트가 1이면 결과에 base 곱하기
                result = result @ base
            # base = base^2
            base = base @ base
            exp >>= 1  # 다음 비트로 이동

        return result

    def __repr__(self) -> str:
        # 행렬을 문자열로 표현: 각 행을 공백으로 구분해 줄바꿈으로 연결
        lines: list[str] = []
        for row in self.matrix:
            lines.append(" ".join(str(val) for val in row))
        return "\n".join(lines)

