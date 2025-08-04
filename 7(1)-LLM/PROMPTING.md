## 📊 정답률 비교 (0-shot, 3-shot, 5-shot)
| Prompting 기법     | 0-shot | 3-shot | 5-shot |
| ---------------- | ------ | ------ | ------ |
| Direct Prompting | 18.00% | 20.00% | 16.00% |
| CoT Prompting    | 58.00% | 70.00% | 70.00% |
| My Prompting     | 68.00% | 72.00% | 72.00% |


## 🔍 CoT Prompting이 Direct Prompting보다 뛰어난 이유

Chain of Thought Prompting(CoT)은 문제 해결 과정을 단계적으로 서술하게 만들어, 단순한 정답 추론보다 더 나은 성능을 유도
특히 수학 문제처럼 중간 계산이 필요한 문제에서 다음과 같은 장점을 가짐
- 사고 과정을 분해하여 논리 전개 가능
- 중간 추론 오류를 방지할 수 있음
- 다단계 사고(Multi-hop reasoning)에 적합함

**예시 (CoT Prompting)**

```
Q: If a bag has 3 apples and each apple costs 2 dollars, how much is the total?
A: There are 3 apples. Each costs 2 dollars. So, 3 × 2 = 6.
#### 6
```

**예시 (Direct Prompting)**

```
Q: If a bag has 3 apples and each apple costs 2 dollars, how much is the total?
A: 6
```

## 🧪 My Prompting이 CoT보다 더 나은 이유
1. `→` 기호를 사용해 reasoning 단계 시각화
2. 설명을 수식 위주로 구성해 핵심 정보만 전달
3. 일관된 Q/A 형식 유지로 모델 혼란 최소화
4. Self-Consistency voting과 잘 어울리도록 구성

특히 정답을 하나만 생성하는 대신 같은 문제에 대해 여러 번 응답을 생성하고, 가장 많이 등장한 답을 최종 정답으로 채택하는 방식(Self-Consistency)을 도입함
이를 통해,
- 불확실한 추론에 대한 안정성 증가
- 복잡한 문제에 대한 평규적 사고 능력 강화
를 기대할 수 있다.

**예시 (My Prompting)**

```
Q: If a bag has 3 apples and each apple costs 2 dollars, how much is the total?
A: → Apples: 3  
→ Cost per apple: 2  
→ Total = 3 × 2 = 6  
#### 6
```