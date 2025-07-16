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
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = None # 구현하세요!

        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    # 구현하세요!
    pass


if __name__ == "__main__":
    main()