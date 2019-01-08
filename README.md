## ML in Indoor Positioning

#### Stage 1
KNN + Gradient Descent

#### Stage 2
Feed Forward, single layer

#### Stage 3
keras 256_128_118, dropout = 0.2, multi-label(,118), epoch = 20, loss = binary_cross_entropy
```python
building accuracy:  100.0 %
building + floor prediction accuracy:  87.82608695652173 %
building + floor + place accuracy:  43.55555555555556 %
```

#### Stage 4
Autoencoder:
1. AE - 256 (single) (finished.)
```python
Without AE:
Test accuracy: 0.9893886969153687
For  5981  test data:
building accuracy:  99.73248620632002 %
building + floor prediction accuracy:  86.94198294599565 %
building + floor + place accuracy:  36.81658585520816 %

--------------------------------------------------------------
With AE:
Test accuracy: 0.9888630269303647
For  5981  test data:
building accuracy:  99.08042133422505 %
building + floor prediction accuracy:  85.45393746865072 %
building + floor + place accuracy:  30.245778297943488 %
```
2. AE - denoised
3. AE - stacked
4. AE - CNN

### Stage 5 

5. CNN
6. AE + CNN
7. AE_CNN + CNN
