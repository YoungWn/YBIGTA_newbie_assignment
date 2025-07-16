from lib import Trie
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
