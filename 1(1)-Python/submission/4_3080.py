from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None         # 이 노드가 표현하는 요소 (int 코드)
    children: list[int] = field(default_factory=list)
    is_end: bool = False             # 이 노드에서 단어가 끝나는지 여부


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        # 인덱스 0번이 항상 루트 노드입니다.
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        curr = 0
        for element in seq:
            node = self[curr]

            # ────── child_map lazy 생성 ──────
            # self[curr] 에 child_map 속성이 없으면
            # 현재 children 리스트를 기반으로 dict를 만들어 줌
            if not hasattr(node, 'child_map'):
                # body 값 → node index
                node.child_map = {self[ch].body: ch for ch in node.children}

            # O(1) lookup
            if element in node.child_map:
                curr = node.child_map[element]
            else:
                # 새 노드 생성
                self.append(TrieNode(body=element))
                new_idx = len(self) - 1
                node.children.append(new_idx)
                node.child_map[element] = new_idx
                curr = new_idx

        # 시퀀스 끝나면 단어 종료 표시
        self[curr].is_end = True



import sys


"""
TODO:
- 일단 lib.py의 Trie Class부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    input = sys.stdin.readline
    MOD = 10**9 + 7

    # 최장 공통 접두사(LCP)를 구하는 함수
    def lcp(a: str, b: str) -> int:
        i = 0
        m = min(len(a), len(b))
        while i < m and a[i] == b[i]:
            i += 1
        return i

    # 1) 입력 처리
    n = int(input())
    names = [input().rstrip() for _ in range(n)]
    names.sort()

    # 2) 각 이름마다 삽입할 최소 prefix 길이를 LCP로 계산
    prefix_lens = [0] * n
    for i in range(n):
        left = lcp(names[i], names[i-1]) if i > 0 else 0
        right = lcp(names[i], names[i+1]) if i < n-1 else 0
        prefix_lens[i] = min(max(left, right) + 1, len(names[i]))

    # 3) Trie 삽입 (generator로 메모리 절약)
    trie = Trie()
    for name, L in zip(names, prefix_lens):
        trie.push((ord(c) - ord('A') for c in name[:L]))

    # 4) iterative DFS로 결과 계산 (재귀 대신 스택)
    ans = 1
    stack = [0]
    while stack:
        idx = stack.pop()
        node = trie[idx]
        cnt = len(node.children) + (1 if node.is_end else 0)

        # factorial(cnt) % MOD
        f = 1
        for k in range(2, cnt + 1):
            f = f * k % MOD
        ans = ans * f % MOD

        stack.extend(node.children)

    print(ans)

if __name__ == "__main__":
    main()
