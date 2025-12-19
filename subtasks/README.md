### 01_odenet_mnist
```
python subtasks/01_odenet_mnist/exec.py
```
* <code>torchdiffeq</code>의 example에서 가져온 코드
* 한 번 실행시켜 보기 위해서 <code>requirements.txt</code>에서 <code>torchdiffeq</code>도 포함한 것

### 02_autograd_practice
```
python subtasks/02_autograd_practice/exec.py
```
* Neural ODE는 Backward를 ODE Solver로 하는 것에 기반하여(Sensitivity Adjoint Method),
* Backpropagation 때 모든 계산을 Autograd의 auto differentiation에 의존하면 안 된다. 

* adjoint state에 대한 부분(Adjoint dynamics)은 Autograd에 대한 조작이 필요하다. (논문에서도 언급한 Memory issue 때문이다.)
* 그래서, PyTorch에 내장되어있는 Autograd를 조금 다룰 줄 알아야 한다.
* 이를 위해 실습하는 Script를 작업했다.